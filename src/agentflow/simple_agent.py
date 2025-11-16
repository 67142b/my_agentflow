import concurrent
from typing import Tuple
from time import time
import json
"""
简化版AgentFlow - 仅保留核心工作流程
去除辅助功能、错误处理、日志记录、配置管理及其他非核心组件
"""
import os
current_dir = os.getcwd()
import asyncio
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入torch
import torch
import numpy as np

# 设置日志
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入核心组件
from src.agentflow.core.memory import Memory
from src.agentflow.core.planner import Planner
from src.agentflow.core.executor import Executor
from src.agentflow.core.verifier import Verifier
from src.agentflow.core.generator import Generator
from src.agentflow.core.memory_filter import MemoryFilter
from src.agentflow.core.adapter import interaction_manager

# 导入LLM引擎
from src.LLM_engine import get_llm_engine

# 导入工具
from src.agentflow.tools.python_executor import PythonExecutorTool
from src.agentflow.tools.web_search import DuckDuckGoSearchTool, GoogleCustomSearchTool
from src.agentflow.tools.wikipedia_search import WikipediaSearchTool, EnhancedWikipediaSearchTool
from src.agentflow.tools.base_generator import BaseGenerator


class SimpleAgent:
    """
    简化版AgentFlow实现
    
    主要组件：
    1. Memory - 记忆管理
    2. Planner - 任务规划
    3. Executor - 工具执行
    4. Verifier - 结果验证
    5. Generator - 答案生成
    """
    
    def __init__(self, config_path: str = "src/configs/config.yaml"):
        """
        初始化简化版AgentFlow
        
        Args:
            config_path: 配置文件路径
            enable_concurrency_control: 是否启用并发控制（锁机制）
        """
        self.config = self._load_config(config_path)
        self.config_path = config_path  # 保存配置路径，供后续使用
        # 初始化LLM引擎
        self.llm_engine = get_llm_engine(config_path)
        
        # 初始化模型缓存，加载母模型和分词器
        self.llm_engine.initialize_model_cache()
        
        # 获取模型配置
        model_config = self.config.get("model", {})
        device = model_config.get("device", "cuda")
        
        # 获取智能体配置
        agent_config = self.config.get("agent", {})
        self.max_turn = agent_config.get("max_tool_calls_per_turn", 3)  # 使用max_tool_calls_per_turn作为每条轨迹的最大尝试次数
        self.group_size = self.config.get("flow_group", {}).get("group_size", 8)  # 添加group_size参数
        
        # 初始化核心组件（不包含记忆，因为每条轨迹需要独立的记忆）
        self.planner = Planner(device=device, llm_engine=self.llm_engine)
        self.verifier = Verifier(config_path=config_path, llm_engine=self.llm_engine)
        self.generator = Generator(config_path=config_path, llm_engine=self.llm_engine)
        self.memory_filter = MemoryFilter(config_path=config_path, llm_engine=self.llm_engine)
        
        # 初始化工具
        self._init_tools(config=self.config)
        self.executor = Executor(config_path=config_path, tools=self.tools, llm_engine=self.llm_engine)
        self.query_id = 1
        self.k = self.config.get("flow_group", {}).get("copy_model_num", 1)  # 添加k参数
        
        # 添加并行/串行控制参数
        self.enable_concurrent = self.config.get("flow_group", {}).get("enable_concurrent", True)
        
        #logger.info("SimpleAgent初始化完成")
        
    def _init_tools(self, config: Dict[str, Any]):
        """初始化工具"""
        self.tools = {}
        
        # BaseGenerator工具 - 使用LLM引擎获取模型
        model_config = config.get("model", {})
        generator_config = model_config.get("base_generator", {})
        
        # 从LLM引擎获取模型配置
        planner_path = model_config.get("planner_path", "")
        generator_path = generator_config.get("model_path", "")
        
        # 判断是否共享模型
        share_model = (planner_path == generator_path) and planner_path != ""
        
        # 从LLM引擎获取模型
        if share_model:
            # 使用LLM引擎获取共享模型
            model, tokenizer = self.llm_engine.get_local_model("planner")
            #print(f"✓ 使用LLM引擎获取共享模型: {planner_path}")
        else:
            model, tokenizer = None, None
            #print("⚠ 不使用模型共享，BaseGenerator将加载独立模型实例")
        
        # 初始化BaseGenerator工具
        if share_model:
            self.tools["base_generator"] = BaseGenerator(
                model=model,
                tokenizer=tokenizer,
                device=model_config.get("device", generator_config.get("device", "cuda")),
                torch_dtype=generator_config.get("torch_dtype", model_config.get("torch_dtype", "float16")),
                max_length=generator_config.get("max_length", 2048),
                llm_engine=self.llm_engine
            )
        else:
            # 正常初始化BaseGenerator
            self.tools["base_generator"] = BaseGenerator(
                model_path=generator_config.get("model_path", "D:\\ModelScope\\cache\\modelscope\\hub\\models\\Qwen\\Qwen2.5-1.5B-Instruct"),
                device=model_config.get("device", generator_config.get("device", "cuda")),
                torch_dtype=generator_config.get("torch_dtype", model_config.get("torch_dtype", "float16")),
                max_length=generator_config.get("max_length", 2048),
                llm_engine=self.llm_engine
            )
        
        # Python执行器
        tool_config = config.get("tools", {})
        python_config = tool_config.get("python_executor", {})
        self.tools["python_executor"] = PythonExecutorTool(
            timeout=python_config.get("timeout", 30),
            max_output=python_config.get("max_output", 1000),
            allow_network=python_config.get("allow_network", False),
            api_key=model_config.get("dashscope_api_key", ""),
            base_url=model_config.get("dashscope_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model_name=model_config.get("python_executor_model_name", "qwen-plus"),
            llm_engine=self.llm_engine
        )
        
        # 搜索工具
        self.tools["duckduckgo_search"] = DuckDuckGoSearchTool(
            timeout=tool_config.get("duckduckgo_search", {}).get("timeout", 10),
            max_results=tool_config.get("search_top_k", 5)
        )
        
        # Wikipedia搜索
        wiki_config = tool_config.get("tools", {})
        
        self.tools["wikipedia_search"] = EnhancedWikipediaSearchTool(
            timeout=wiki_config.get("search_timeout", 10),
            language=wiki_config.get("wikipedia_language", "en"),
            max_results=tool_config.get("search_top_k", 5),
            include_images=wiki_config.get("include_images", False),
            include_references=wiki_config.get("include_references", False)
        )
        
        # Google Custom Search
        self.tools["google_custom_search"] = GoogleCustomSearchTool(
            timeout=tool_config.get("google_custom_search", {}).get("timeout", 10),
            max_results=tool_config.get("search_top_k", 5),
            api_key=tool_config.get("search_apis",{}).get("google_api_key", ""), 
            search_engine_id=tool_config.get("search_apis",{}).get("google_search_engine_id", "")
        )
        self.tools["web_search"] = self.tools[self.config.get("tools", {}).get("preferred_search_tool", "duckduckgo_search")]
    
    def singal_solve(self, query: str) -> Dict[str, Any]:
        """
        简化版解决方法，只调用generate_with_token_probs生成轨迹
        
        该方法用于测试当系统退化成普通GRPO时的显存占用情况
        
        Args:
            query: 用户查询
            
        Returns:
            Dict[str, Any]: 解决结果，格式与solve方法相同
        """
        # 记录开始时间
        time_start = time()
        # 存储所有轨迹的结果
        all_trajectories = []
        
        # 生成group_size个轨迹
        for i in range(self.group_size):
            try:
                # 直接调用generate_with_token_probs生成轨迹
                result = self.llm_engine.generate_with_token_probs(
                    model_name="planner",
                    prompt=query
                )
                
                # 构建与solve方法相同格式的轨迹结果
                trajectory_result = {
                    "trajectory_id": i + 1,
                    "success": True,  # 简化版总是成功
                    "final_answer": result,
                    "turns": [{
                        "turn": 1,
                        "plan": result,  # 将整个结果作为规划阶段
                        "execution": None,
                        "verification": None
                    }],
                    "error": None,
                    "trajectory_data": {
                        "query": query,
                        "turns": [{
                            "planning": {
                                "sub_goal": "Generate response",
                                "selected_tool": "planner",
                                "tool_context": query,
                                "confidence": 1.0,
                                "planner_input_ids": result.get("planner_input_ids", []),
                                "planner_output_ids": result.get("planner_output_ids", []),
                                "logprobs_old": result.get("logprobs_old", [])
                            },
                            "execution": None,
                            "verification": None,
                            "reasoning": None
                        }]
                    }
                }
                all_trajectories.append(trajectory_result)   
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"轨迹 {i} 处理失败: {str(e)}")
                trajectory_result = {
                    "trajectory_id": i + 1,
                    "success": False,
                    "error": str(e),
                    "final_answer": "",
                    "turns": [],
                    "trajectory_data": None
                }
                all_trajectories.append(trajectory_result)
        
        # 整理结果
        successful_trajectories = [t for t in all_trajectories if t.get("success", False)]
        failed_trajectories = [t for t in all_trajectories if not t.get("success", False)]
        
        # 计算成功率
        success_rate = len(successful_trajectories) / len(all_trajectories) if all_trajectories else 0
        
        # 准备返回结果
        result = {
            "query": query,
            "query_id": self.query_id,
            "total_trajectories": len(all_trajectories),
            "successful_trajectories": successful_trajectories,
            "failed_trajectories": failed_trajectories,
            "success_rate": success_rate,
            "time_taken": time() - time_start
        }
        
        # 更新查询ID
        self.query_id += 1
        
        return result
    
    def solve(self, query: str) -> Dict[str, Any]:
        """
        解决用户查询
        
        Args:
            query: 用户查询
            
        Returns:
            Dict[str, Any]: 解决结果
        """
        # 存储所有轨迹的结果
        time_start = time()
        all_trajectories = []
        
        # 创建结果目录前先删除该目录下的所有文件（如果有）
        result_dir = f"test_results/query_{self.query_id}"
        if os.path.exists(result_dir):
            for filename in os.listdir(result_dir):
                file_path = os.path.join(result_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"删除文件 {file_path} 失败: {e}")
        
        # 创建结果目录
        os.makedirs(result_dir, exist_ok=True)
        
        # 根据配置选择串行或并发模式
        if self.enable_concurrent:
            #logger.info(f"使用并发模式处理轨迹，并发数: {self.group_size}")
            all_trajectories = self._solve_concurrent(query)
        else:
            #logger.info("使用串行模式处理轨迹")
            # 串行处理
            results = []
            
            # 解决group_size个轨迹
            for i in range(self.group_size):
                try:
                    # 创建独立的记忆实例
                    memory = Memory(config_path=self.config_path)
                    
                    # 调用_solve_single_trajectory方法（串行模式不创建模型副本）
                    trajectory_result = self._solve_single_trajectory(query, memory, i)
                    results.append(trajectory_result)
                    
                except Exception as e:
                    logger.error(f"轨迹 {i} 处理失败: {str(e)}")
                    results.append({
                        "trajectory_id": i,
                        "success": False,
                        "error": str(e),
                        "final_answer": "",
                        "trajectory_data": None
                    })
            
            all_trajectories = results
        
        # 整理结果
        successful_trajectories = [t for t in all_trajectories if t.get("success", False)]
        failed_trajectories = [t for t in all_trajectories if not t.get("success", False)]
        
        
        # 计算成功率
        success_rate = len(successful_trajectories) / len(all_trajectories) if all_trajectories else 0
        
        # 准备返回结果
        result = {
            "query": query,
            "query_id": self.query_id,
            "total_trajectories": len(all_trajectories),
            "successful_trajectories": successful_trajectories,
            "failed_trajectories": failed_trajectories,
            "success_rate": success_rate,
            "time_taken": time() - time_start
        }
        
        # 更新查询ID
        self.query_id += 1
        end_time = time()
        #logger.info(f"问题解决完成: {len(successful_trajectories)}/{len(all_trajectories)} 条轨迹成功，用时：{end_time - time_start:.2f}秒")
        return result
    
    def _solve_concurrent(self, query: str) -> List[Dict[str, Any]]:
        """
        并发解决多条轨迹，使用模型副本和同步锁机制
        
        Args:
            query: 查询字符串
            
        Returns:
            List[Dict[str, Any]]: 轨迹结果列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
        import signal
        import os
        import time
        import ctypes
        import sys
        
        # 创建模型副本
        #logger.info(f"开始创建 {self.k} 个模型副本用于并发处理")
        model_copies = self.llm_engine.copy_model("planner", self.k)
        
        # 用于跟踪线程状态
        completed_count = 0
        total_count = self.group_size
        results = [None] * total_count  # 预分配结果数组
        active_threads = {}  # 跟踪活动线程
        thread_stop_flags = {}  # 用于通知线程停止
        thread_futures = {}  # 跟踪线程的Future对象
        
        def force_terminate_thread(thread_id):
            """强制终止线程"""
            try:
                # 在Windows上使用ctypes强制终止线程
                if sys.platform == "win32":
                    # 获取线程句柄
                    handle = ctypes.windll.kernel32.OpenThread(0x0001, False, thread_id)
                    if handle:
                        # 终止线程
                        ctypes.windll.kernel32.TerminateThread(handle, -1)
                        ctypes.windll.kernel32.CloseHandle(handle)
                        return True
                else:
                    # 在Unix-like系统上使用信号终止线程
                    os.kill(thread_id, signal.SIGTERM)
                    return True
            except Exception as e:
                logger.error(f"强制终止线程 {thread_id} 失败: {str(e)}")
            return False
        
        def process_trajectory(trajectory_index):
            """处理单个轨迹的内部函数"""
            nonlocal completed_count
            
            # 记录当前线程ID
            current_thread_id = threading.get_ident()
            active_threads[trajectory_index] = current_thread_id
            thread_stop_flags[trajectory_index] = threading.Event()
            
            try:
                #logger.info(f"开始处理轨迹 {trajectory_index + 1}/{total_count} (线程ID: {current_thread_id})")
                
                # 检查是否应该停止
                if thread_stop_flags[trajectory_index].is_set():
                    #logger.warning(f"轨迹 {trajectory_index + 1} 被标记为停止，提前退出")
                    return
                
                # 使用超时机制处理单个轨迹
                trajectory_result = self._solve_single_trajectory_with_timeout(
                    query,
                    Memory(config_path=self.config_path),  # 创建独立的记忆实例
                    trajectory_index,  # 轨迹ID从0开始，内部会+1
                    thread_stop_flags[trajectory_index]  # 传递停止标志
                )
                
                # 检查是否被超时中断
                if thread_stop_flags[trajectory_index].is_set():
                    #logger.warning(f"轨迹 {trajectory_index + 1} 因超时被中断")
                    results[trajectory_index] = {
                        "trajectory_id": trajectory_index + 1,
                        "success": False,
                        "error": f"处理超时",
                        "final_answer": "",
                        "trajectory_data": None
                    }
                else:
                    results[trajectory_index] = trajectory_result
                    #logger.info(f"轨迹 {trajectory_index + 1} 处理完成")
            except Exception as e:
                #logger.error(f"轨迹 {trajectory_index + 1} 处理失败: {str(e)}")
                results[trajectory_index] = {
                    "trajectory_id": trajectory_index + 1,
                    "success": False,
                    "error": str(e),
                    "final_answer": "",
                    "trajectory_data": None
                }
            finally:
                # 更新完成计数
                with threading.Lock():
                    completed_count += 1
                    # 从活动线程列表中移除
                    if trajectory_index in active_threads:
                        del active_threads[trajectory_index]
        
        try:
            # 使用线程池并发处理，限制并发数为可用的模型实例数量
            with ThreadPoolExecutor(max_workers=self.group_size) as executor:
                # 提交所有任务
                futures = []
                for i in range(self.group_size):
                    future = executor.submit(process_trajectory, i)
                    futures.append(future)
                    thread_futures[i] = future
                
                # 等待所有任务完成或超时
                try:
                    # 设置超时时间
                    timeout_seconds = 1200
                    #logger.info(f"等待所有轨迹完成，超时时间: {timeout_seconds}秒")
                    
                    # 使用as_completed和超时机制
                    completed_futures = 0
                    start_time = time.time()
                    
                    while completed_futures < len(futures):
                        # 检查是否超时
                        elapsed_time = time.time() - start_time
                        if elapsed_time > timeout_seconds:
                            #logger.warning(f"总体处理超时 ({timeout_seconds}秒)，开始终止未完成的轨迹")
                            
                            # 设置所有未完成线程的停止标志
                            for i, future in enumerate(futures):
                                if not future.done() and i in thread_stop_flags:
                                    #logger.warning(f"标记轨迹 {i + 1} 为停止状态")
                                    thread_stop_flags[i].set()
                                    
                                    # 设置超时结果
                                    if results[i] is None:
                                        results[i] = {
                                            "trajectory_id": i + 1,
                                            "success": False,
                                            "error": f"处理超时 (超过 {timeout_seconds} 秒)",
                                            "final_answer": "",
                                            "trajectory_data": None
                                        }
                            
                            # 等待线程响应停止信号
                            #logger.warning(f"等待 {len(active_threads)} 个活动线程响应停止信号...")
                            time.sleep(3)  # 给线程一些时间来响应停止信号
                            
                            # 强制终止仍然活动的线程
                            for i, thread_id in list(active_threads.items()):
                                if not futures[i].done():
                                    #logger.warning(f"强制终止轨迹 {i + 1} 的线程 {thread_id}")
                                    if force_terminate_thread(thread_id):
                                        #logger.warning(f"成功终止轨迹 {i + 1} 的线程")
                                        pass
                                    else:
                                        #logger.error(f"无法终止轨迹 {i + 1} 的线程，尝试取消Future")
                                        try:
                                            futures[i].cancel()
                                        except Exception as e:
                                            logger.error(f"取消Future失败: {str(e)}")
                            
                            # 强制标记未完成的轨迹为超时
                            for i in range(total_count):
                                if results[i] is None:
                                    #logger.warning(f"强制标记轨迹 {i + 1} 为超时")
                                    results[i] = {
                                        "trajectory_id": i + 1,
                                        "success": False,
                                        "error": f"处理超时 (超过 {timeout_seconds} 秒)",
                                        "final_answer": "",
                                        "trajectory_data": None
                                    }
                            break
                        
                        # 等待至少一个任务完成，但设置较短的超时以便定期检查总体超时
                        try:
                            for future in as_completed(futures, timeout=10):
                                completed_futures += 1
                                # 检查是否有异常
                                try:
                                    future.result()
                                except Exception as e:
                                    logger.error(f"轨迹执行异常: {str(e)}")
                                break
                        except FutureTimeoutError:
                            # 只是as_completed超时，继续循环检查总体超时
                            continue
                        
                except Exception as e:
                    logger.error(f"等待轨迹完成时发生错误: {str(e)}")
                
                # 确保所有结果都被填充
                for i in range(total_count):
                    if results[i] is None:
                        results[i] = {
                            "trajectory_id": i + 1,
                            "success": False,
                            "error": "未知错误",
                            "final_answer": "",
                            "trajectory_data": None
                        }
                
                return results
        finally:
            # 释放模型副本并清理缓存
            #logger.info("开始释放模型副本并清理GPU缓存")
            self.llm_engine.release_model_copies("planner")
            #logger.info("已释放所有模型副本并清理GPU缓存")
    
    def _solve_single_trajectory_with_timeout(self, query: str, memory: Memory, trajectory_id: int, stop_event: threading.Event) -> Dict[str, Any]:
        """
        解决单条轨迹，带超时控制
        
        Args:
            query: 用户查询
            memory: 当前轨迹的记忆实例
            trajectory_id: 轨迹ID
            stop_event: 用于通知线程停止的事件
            
        Returns:
            Dict: 轨迹解决结果
        """
        # 记录轨迹开始
        # 修复add_query调用，使用initialize方法
        memory.initialize(query)
        
        # 初始化轨迹结果
        trajectory_result = {
            "trajectory_id": trajectory_id + 1,  # 使用与日志一致的轨迹编号（从1开始）
            "success": False,
            "final_answer": "",
            "turns": [],
            "error": None
        }
        
        # 进行多轮尝试
        for turn in range(self.max_turn):
            # 检查是否应该停止
            if stop_event.is_set():
                logger.warning(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮被标记为停止")
                trajectory_result["error"] = "处理被中断"
                return trajectory_result
                
            #logger.info(f"轨迹 {trajectory_id + 1}, 轮次 {turn + 1}/{self.max_turn}")
            
            # 记录轮次开始 - 移到循环开始处
            memory.next_turn()
            
            # 检查是否应该停止
            if stop_event.is_set():
                logger.warning(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮被标记为停止")
                trajectory_result["error"] = "处理被中断"
                return trajectory_result
            
            # 规划阶段 - 添加停止检查
            try:
                memory_context = memory.get_context()
                
                # 在规划前检查停止信号
                if stop_event.is_set():
                    logger.warning(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮规划前被标记为停止")
                    trajectory_result["error"] = "处理被中断"
                    return trajectory_result
                    
                plan_result = self.planner.plan(query, memory_context, ["general"])
                
                # 在规划后检查停止信号
                if stop_event.is_set():
                    logger.warning(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮规划后被标记为停止")
                    trajectory_result["error"] = "处理被中断"
                    return trajectory_result
            except Exception as e:
                if stop_event.is_set():
                    trajectory_result["error"] = "处理被中断"
                else:
                    trajectory_result["error"] = f"规划阶段错误: {str(e)}"
                return trajectory_result
            
            # 修复add_planning调用，确保传递所有必需参数
            # 验证和处理logprobs_old
            logprobs_old = plan_result.get("logprobs_old", [])
            if logprobs_old is None:
                logprobs_old = []
            elif isinstance(logprobs_old, list):
                # 检查并替换无效值
                for i, val in enumerate(logprobs_old):
                    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                        logprobs_old[i] = -10.0  # 使用默认值替换无效值
            
            memory.add_planning(
                sub_goal=plan_result.get("sub_goal", ""),
                selected_tool=plan_result.get("selected_tool", ""),
                tool_context=plan_result.get("tool_context", ""),
                confidence=plan_result.get("confidence", 0.0),
                planner_input_ids=plan_result.get("planner_input_ids", []),
                planner_output_ids=plan_result.get("planner_output_ids", []),
                logprobs_old=logprobs_old
            )
            
            # 执行阶段 - 添加停止检查
            tool_name = plan_result.get("selected_tool", "")
            tool_context = plan_result.get("tool_context", "")
            sub_goal = plan_result.get("sub_goal", "")
            parameters = {
                "query": tool_context,
                "context": memory_context,
                "raw_query": query
            }
            
            # 在执行前检查停止信号
            if stop_event.is_set():
                logger.warning(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮执行前被标记为停止")
                trajectory_result["error"] = "处理被中断"
                return trajectory_result
            
            try:
                # 直接执行工具调用，支持返回token概率信息
                execution_result = self.executor.execute(tool_name, parameters, trajectory_id+1)
                
                # 在执行后检查停止信号
                if stop_event.is_set():
                    logger.warning(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮执行后被标记为停止")
                    trajectory_result["error"] = "处理被中断"
                    return trajectory_result
            except Exception as e:
                if stop_event.is_set():
                    trajectory_result["error"] = "处理被中断"
                else:
                    trajectory_result["error"] = f"执行阶段错误: {str(e)}"
                return trajectory_result
            
            # 修复add_execution调用，确保传递所有必需参数
            # 验证和处理logprobs_old
            logprobs_old = execution_result.get("logprobs_old", [])
            if logprobs_old is None:
                logprobs_old = []
            elif isinstance(logprobs_old, list):
                # 检查并替换无效值
                for i, val in enumerate(logprobs_old):
                    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                        logprobs_old[i] = -10.0  # 使用默认值替换无效值
                        
            memory.add_execution(
                sub_goal=sub_goal,
                selected_tool=tool_name,
                tool_context=tool_context,
                execution_result=str(execution_result.get("result", "")),
                success=execution_result.get("success", True) if isinstance(execution_result, dict) else True,
                metadata=execution_result.get("metadata", {}) if isinstance(execution_result, dict) else {},
                planner_input_ids=execution_result.get("planner_input_ids", []),
                planner_output_ids=execution_result.get("planner_output_ids", []),
                logprobs_old=logprobs_old
            )
            
            # 验证阶段 - 添加停止检查
            tool_info = {
                "name": tool_name,
                "context": tool_context
            }
            
            # 在验证前检查停止信号
            if stop_event.is_set():
                logger.warning(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮验证前被标记为停止")
                trajectory_result["error"] = "处理被中断"
                return trajectory_result
            
            try:
                verification_result = self.verifier.verify(
                    query=query,
                    sub_goal=sub_goal,
                    tool_info=tool_info,
                    execution_result=str(execution_result.get("result", "")),
                    memory_context=memory_context,
                    current_turn=turn + 1,
                    max_turn=self.max_turn
                )
                
                # 在验证后检查停止信号
                if stop_event.is_set():
                    logger.warning(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮验证后被标记为停止")
                    trajectory_result["error"] = "处理被中断"
                    return trajectory_result
            except Exception as e:
                if stop_event.is_set():
                    trajectory_result["error"] = "处理被中断"
                else:
                    trajectory_result["error"] = f"验证阶段错误: {str(e)}"
                return trajectory_result
            
            # 修复add_verification调用，确保传递所有必需参数
            result_data = verification_result.get("result", {}) if isinstance(verification_result, dict) else {}
            # 验证和处理logprobs_old
            logprobs_old = verification_result.get("logprobs_old", [])
            if logprobs_old is None:
                logprobs_old = []
            elif isinstance(logprobs_old, list):
                # 检查并替换无效值
                for i, val in enumerate(logprobs_old):
                    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                        logprobs_old[i] = -10.0  # 使用默认值替换无效值
                        
            memory.add_verification(
                sub_goal=sub_goal,
                selected_tool=tool_name,
                tool_context=verification_result.get("raw_output"),
                execution_result=str(execution_result.get("result", "")),
                verification_status=result_data.get("verifier_status", 0),
                confidence=result_data.get("confidence", 0.0),
                analysis=result_data.get("analysis", ""),
                planner_input_ids=verification_result.get("planner_input_ids", []),
                planner_output_ids=verification_result.get("planner_output_ids", []),
                logprobs_old=logprobs_old
            )
            
            # 记录本轮结果
            turn_result = {
                "turn": turn + 1,
                "plan": plan_result,
                "execution": execution_result,
                "verification": verification_result
            }
            trajectory_result["turns"].append(turn_result)
            
            # 如果验证通过，生成最终答案并结束
            result_data = verification_result.get("result", {}) if isinstance(verification_result, dict) else {}
            #print(f"轨迹：{trajectory_id + 1} 验证结果：\n",result_data.get("verifier_status", 0),"\n")
            if result_data.get("verifier_status", 0) == 1:
                # 在生成答案前检查停止信号
                if stop_event.is_set():
                    logger.warning(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮生成答案前被标记为停止")
                    trajectory_result["error"] = "处理被中断"
                    return trajectory_result
                
                try:
                    # 获取当前记忆上下文
                    memory_context = memory.get_context()
                    #print("开始过滤记忆上下文")
                    filtered_memory_context = self.memory_filter.filter_memory(query, memory_context)
                    #print("过滤完成")
                    
                    # 生成最终答案
                    final_answer = self.generator.generate(query, filtered_memory_context, temperature=0.5)
                    
                    # 在生成答案后检查停止信号
                    if stop_event.is_set():
                        logger.warning(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮生成答案后被标记为停止")
                        trajectory_result["error"] = "处理被中断"
                        return trajectory_result
                    
                    # 验证和处理logprobs_old
                    logprobs_old = final_answer.get("logprobs_old", [])
                    if logprobs_old is None:
                        logprobs_old = []
                    elif isinstance(logprobs_old, list):
                        # 检查并替换无效值
                        for i, val in enumerate(logprobs_old):
                            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                                logprobs_old[i] = -10.0  # 使用默认值替换无效值
                                
                    memory.add_reasoning(
                        sub_goal="Generate final answer",
                        tool_context=f"Query: {query}\nMemory context: {filtered_memory_context}",
                        reasoning_result=final_answer,
                        confidence=0.0,
                        planner_input_ids=final_answer.get("planner_input_ids", []),
                        planner_output_ids=final_answer.get("planner_output_ids", []),
                        logprobs_old=logprobs_old
                    )
                    trajectory_result["success"] = True
                    trajectory_result["final_answer"] = final_answer
                    
                    #logger.info(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮成功")
                    break
                except Exception as e:
                    if stop_event.is_set():
                        trajectory_result["error"] = "处理被中断"
                    else:
                        trajectory_result["error"] = f"生成答案阶段错误: {str(e)}"
                    return trajectory_result
            else:
                #logger.info(f"轨迹 {trajectory_id + 1} 第 {turn + 1} 轮验证失败，继续下一轮")
                if not trajectory_result["success"] and not trajectory_result["error"]:
                    trajectory_result["error"] = "所有轮次验证均失败"
                    #logger.info(f"轨迹 {trajectory_id + 1} 所有轮次均失败")
        
        # 从记忆中提取轨迹数据，包含token概率信息
        try:
            trajectory_data = memory.get_trajectory_data()
            # 将轨迹数据添加到结果中
            trajectory_result["trajectory_data"] = trajectory_data
        except Exception as e:
            logger.error(f"提取轨迹数据失败: {str(e)}")
            trajectory_result["trajectory_data"] = None
            
        try:
            memory.save_to_file(f"test_results/query_{self.query_id}/memory_{trajectory_id + 1}.json")
        except Exception as e:
            logger.error(f"保存记忆文件失败: {str(e)}")
            
        try:
            del memory
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")
            
        return trajectory_result
    
    def _solve_single_trajectory(self, query: str, memory: Memory, trajectory_id: int) -> Dict[str, Any]:
        """
        解决单条轨迹
        
        Args:
            query: 用户查询
            memory: 当前轨迹的记忆实例
            trajectory_id: 轨迹ID
            
        Returns:
            Dict: 轨迹解决结果
        """
        # 记录轨迹开始
        # 修复add_query调用，使用initialize方法
        memory.initialize(query)
        
        # 初始化轨迹结果
        trajectory_result = {
            "trajectory_id": trajectory_id + 1,  # 使用与日志一致的轨迹编号（从1开始）
            "success": False,
            "final_answer": "",
            "turns": [],
            "error": None
        }
        
        # 进行多轮尝试
        for turn in range(self.max_turn):
            #logger.info(f"轨迹 {trajectory_id + 1}, 轮次 {turn + 1}/{self.max_turn}")
            
            # 记录轮次开始 - 移到循环开始处
            memory.next_turn()
            
            # 规划阶段
            memory_context = memory.get_context()
            plan_result = self.planner.plan(query, memory_context, ["general"])
            #print(f"规划结果: {plan_result}")
            
            # 修复add_planning调用，确保传递所有必需参数
            # 验证和处理logprobs_old
            logprobs_old = plan_result.get("logprobs_old", [])
            if logprobs_old is None:
                logprobs_old = []
            elif isinstance(logprobs_old, list):
                # 检查并替换无效值
                for i, val in enumerate(logprobs_old):
                    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                        logprobs_old[i] = -10.0  # 使用默认值替换无效值
                        
            memory.add_planning(
                sub_goal=plan_result.get("sub_goal", ""),
                selected_tool=plan_result.get("selected_tool", ""),
                tool_context=plan_result.get("tool_context", ""),
                confidence=plan_result.get("confidence", 0.0),
                planner_input_ids=plan_result.get("planner_input_ids", []),
                planner_output_ids=plan_result.get("planner_output_ids", []),
                logprobs_old=logprobs_old
            )
            
            # 执行阶段
            tool_name = plan_result.get("selected_tool", "")
            tool_context = plan_result.get("tool_context", "")
            sub_goal = plan_result.get("sub_goal", "")
            parameters = {
                "query": tool_context,
                "context": memory_context,
                "raw_query": query
            }
            
            # 直接执行工具调用，支持返回token概率信息
            execution_result = self.executor.execute(tool_name, parameters, trajectory_id+1)
            
            # 修复add_execution调用，确保传递所有必需参数
            # 验证和处理logprobs_old
            logprobs_old = execution_result.get("logprobs_old", [])
            if logprobs_old is None:
                logprobs_old = []
            elif isinstance(logprobs_old, list):
                # 检查并替换无效值
                for i, val in enumerate(logprobs_old):
                    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                        logprobs_old[i] = -10.0  # 使用默认值替换无效值
                        
            memory.add_execution(
                sub_goal=sub_goal,
                selected_tool=tool_name,
                tool_context=tool_context,
                execution_result=str(execution_result.get("result", "")),
                success=execution_result.get("success", True) if isinstance(execution_result, dict) else True,
                metadata=execution_result.get("metadata", {}) if isinstance(execution_result, dict) else {},
                planner_input_ids=execution_result.get("planner_input_ids", []),
                planner_output_ids=execution_result.get("planner_output_ids", []),
                logprobs_old=logprobs_old
            )
            
            # 验证阶段
            tool_info = {
                "name": tool_name,
                "context": tool_context
            }
            
            verification_result = self.verifier.verify(
                query=query,
                sub_goal=sub_goal,
                tool_info=tool_info,
                execution_result=str(execution_result.get("result", "")),
                memory_context=memory_context,
                current_turn=turn + 1,
                max_turn=self.max_turn
            )
            
            # 修复add_verification调用，确保传递所有必需参数
            result_data = verification_result.get("result", {}) if isinstance(verification_result, dict) else {}
            # 验证和处理logprobs_old
            logprobs_old = verification_result.get("logprobs_old", [])
            if logprobs_old is None:
                logprobs_old = []
            elif isinstance(logprobs_old, list):
                # 检查并替换无效值
                for i, val in enumerate(logprobs_old):
                    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                        logprobs_old[i] = -10.0  # 使用默认值替换无效值
                        
            memory.add_verification(
                sub_goal=sub_goal,
                selected_tool=tool_name,
                tool_context=verification_result.get("raw_output"),
                execution_result=str(execution_result.get("result", "")),
                verification_status=result_data.get("verifier_status", 0),
                confidence=result_data.get("confidence", 0.0),
                analysis=result_data.get("analysis", ""),
                planner_input_ids=verification_result.get("planner_input_ids", []),
                planner_output_ids=verification_result.get("planner_output_ids", []),
                logprobs_old=logprobs_old
            )
            
            # 记录本轮结果
            turn_result = {
                "turn": turn + 1,
                "plan": plan_result,
                "execution": execution_result,
                "verification": verification_result
            }
            trajectory_result["turns"].append(turn_result)
            
            # 如果验证通过，生成最终答案并结束
            result_data = verification_result.get("result", {}) if isinstance(verification_result, dict) else {}
            #print(f"轨迹：{trajectory_id + 1} 验证结果：\n",result_data.get("verifier_status", 0),"\n")
            if result_data.get("verifier_status", 0) == 1:
                # 获取当前记忆上下文
                memory_context = memory.get_context()
                #print("开始过滤记忆上下文")
                filtered_memory_context = self.memory_filter.filter_memory(query, memory_context)
                #print(f"过滤完成")
                
                # 生成最终答案
                final_answer = self.generator.generate(query, filtered_memory_context, temperature=0.5)
                
                # 验证和处理logprobs_old
                logprobs_old = final_answer.get("logprobs_old", [])
                if logprobs_old is None:
                    logprobs_old = []
                elif isinstance(logprobs_old, list):
                    # 检查并替换无效值
                    for i, val in enumerate(logprobs_old):
                        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                            logprobs_old[i] = -10.0  # 使用默认值替换无效值
                            
                memory.add_reasoning(
                    sub_goal="Generate final answer",
                    tool_context=f"Query: {query}\nMemory context: {filtered_memory_context}",
                    reasoning_result=final_answer,
                    confidence=0.0,
                    planner_input_ids=final_answer.get("planner_input_ids", []),
                    planner_output_ids=final_answer.get("planner_output_ids", []),
                    logprobs_old=logprobs_old
                )
                trajectory_result["success"] = True
                trajectory_result["final_answer"] = final_answer
                
                #logger.info(f"轨迹 {trajectory_id + 1} 在第 {turn + 1} 轮成功")
                break
            else:
                #logger.info(f"轨迹 {trajectory_id + 1} 第 {turn + 1} 轮验证失败，继续下一轮")
                if not trajectory_result["success"] and not trajectory_result["error"]:
                    trajectory_result["error"] = "所有轮次验证均失败"
                    #logger.info(f"轨迹 {trajectory_id + 1} 所有轮次均失败")
        
        # 从记忆中提取轨迹数据，包含token概率信息
        trajectory_data = memory.get_trajectory_data()
        
        # 将轨迹数据添加到结果中
        trajectory_result["trajectory_data"] = trajectory_data
        
        memory.save_to_file(f"test_results/query_{self.query_id}/memory_{trajectory_id + 1}.json")
        del memory
        torch.cuda.empty_cache()
        return trajectory_result
    
    def cleanup_resources(self):
        """
        清理所有资源，包括模型副本和GPU内存
        """
        try:
            # 释放所有模型副本
            self.llm_engine.release_model_copies("planner")
            
            # 清理所有缓存
            self.llm_engine.clear_cache()
            
            #logger.info("已清理所有资源")
        except Exception as e:
            logger.error(f"清理资源时出错: {str(e)}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict: 配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
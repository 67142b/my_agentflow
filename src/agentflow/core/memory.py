from asyncio.log import logger
import uuid
from typing import Set
"""
Memory System for AgentFlow
记忆系统：记录和管理智能体的推理过程和轨迹数据
"""
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class MemoryEntryType(Enum):
    """记忆条目类型"""
    ACTION = "action"
    PLANNING = "planning"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    REASONING = "reasoning"
    OBSERVATION = "observation"


@dataclass
class MemoryEntry:
    """
    记忆条目数据类
    
    记录智能体在执行过程中的每一步操作和推理
    """
    turn: int                    # 回合数
    timestamp: float             # 时间戳
    entry_type: MemoryEntryType  # 条目类型
    sub_goal: str               # 子目标
    selected_tool: str          # 选择的工具
    tool_context: str           # 工具上下文
    execution_result: str       # 执行结果
    success: bool = True        # 执行是否成功
    evaluation_score: Optional[float] = None  # 评估分数
    confidence: Optional[float] = None        # 置信度
    metadata: Optional[Dict[str, Any]] = None # 元数据
    
    # GRPO算法所需的记忆系统信息
    planner_input_ids: Optional[List[int]] = None  # 当前轮次planner提示词的编码
    planner_output_ids: Optional[List[int]] = None  # 当前轮次planner输出的编码
    logprobs_old: Optional[List[float]] = None  # 当前轮次planner输出的逐token对数概率
    sampling_params: Optional[Dict[str, Any]] = None  # 采样参数
    
    # 多答案采样支持
    alternative_results: Optional[List[Dict[str, Any]]] = None  # 替代结果列表
    # 轨迹ID，用于区分不同轨迹
    trajectory_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['entry_type'] = self.entry_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """从字典创建实例"""
        data['entry_type'] = MemoryEntryType(data['entry_type'])
        return cls(**data)


class Memory:
    """
    记忆系统类
    
    职责：
    1. 记录智能体的推理过程和轨迹数据
    2. 提供上下文信息
    3. 支持状态跟踪和回溯
    4. 为强化学习提供训练数据
    """
    
    def __init__(self, config_path: str = "src/configs/config.yaml", trajectory_id: Optional[str] = None):
        """
        初始化记忆系统
        
        Args:
            config_path: 配置文件路径
            trajectory_id: 轨迹ID，用于区分不同轨迹
        """
        self.trajectory_id = trajectory_id or str(uuid.uuid4())  # 如果没有提供轨迹ID，则生成一个
        self.config = self._load_config(config_path)
        self.max_entries = self.config.get("memory", {}).get("max_entries", 100)
        self.entries: List[MemoryEntry] = []
        self.current_turn = 0
        self.initial_query = ""
        self.initial_context = ""
        self.required_skills: List[str] = []
        self.terminated = False
        self.termination_reason = ""
        self.used_tools: Set[str] = set()
        
        # 性能指标
        self.total_tokens = 0
        self.total_execution_time = 0.0
        self.successful_executions = 0
        self.failed_executions = 0
        
        # 回合管理
        self.turn_entry_count: Dict[int, int] = {0: 0}  # 每个turn的条目计数
        self.turn_start_time: Dict[int, float] = {0: time.time()}  # 每个turn的开始时间
    
    def initialize(self, query: str, context: str = "", required_skills: List[str] = None):
        """
        初始化记忆系统
        
        Args:
            query: 初始查询
            context: 初始上下文
            required_skills: 所需技能列表
        """
        self.initial_query = query
        self.initial_context = context
        self.required_skills = required_skills or []
        self.current_turn = 0
        self.terminated = False
        self.termination_reason = ""
        
        # 重置turn管理
        self.turn_entry_count = {0: 0}
        self.turn_start_time = {0: time.time()}
        
        # 添加初始观察条目
        self.add_observation(
            sub_goal="Process initial query",
            tool_context=f"Query: {query}\nContext: {context}\nRequired skills: {self.required_skills}",
            observation_result="Memory system initialized"
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict: 配置字典
        """
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def add_action(self, 
                   sub_goal: str, 
                   selected_tool: str, 
                   tool_context: str, 
                   execution_result: str,
                   entry_type: MemoryEntryType = MemoryEntryType.ACTION,
                   success: bool = True,
                   confidence: Optional[float] = None,
                   metadata: Dict[str, Any] = None,
                   evaluation_score: Optional[float] = None,
                   # GRPO算法所需的记忆系统信息
                   planner_input_ids: Optional[List[int]] = None,
                   planner_output_ids: Optional[List[int]] = None,
                   logprobs_old: Optional[List[float]] = None,
                   sampling_params: Optional[Dict[str, Any]] = None,
                   # 多答案采样支持
                   alternative_results: Optional[List[Dict[str, Any]]] = None):
        """
        添加动作条目到记忆中
        
        Args:
            sub_goal: 子目标
            selected_tool: 选择的工具
            tool_context: 工具上下文
            execution_result: 执行结果
            entry_type: 条目类型
            success: 执行是否成功
            confidence: 置信度
            metadata: 元数据
            evaluation_score: 评估分数
            planner_input_ids: 当前轮次planner提示词的编码
            planner_output_ids: 当前轮次planner输出的编码
            logprobs_old: 当前轮次planner输出的逐token对数概率
            sampling_params: 采样参数
            alternative_results: 替代结果列表
        """
        # 确保当前turn有计数器
        if self.current_turn not in self.turn_entry_count:
            self.turn_entry_count[self.current_turn] = 0
        
        # 创建记忆条目，包含轨迹ID
        entry = MemoryEntry(
            turn=self.current_turn,
            timestamp=time.time(),
            entry_type=entry_type,
            sub_goal=sub_goal,
            selected_tool=selected_tool,
            tool_context=tool_context,
            execution_result=execution_result,
            success=success,
            confidence=confidence,
            metadata=metadata or {},
            evaluation_score=evaluation_score,
            planner_input_ids=planner_input_ids,
            planner_output_ids=planner_output_ids,
            logprobs_old=logprobs_old,
            sampling_params=sampling_params,
            alternative_results=alternative_results,
            trajectory_id=self.trajectory_id  # 添加轨迹ID
        )
        
        # 添加到记忆列表
        self.entries.append(entry)
        
        # 更新turn条目计数
        self.turn_entry_count[self.current_turn] += 1
        
        # 更新使用的工具集合
        self.used_tools.add(selected_tool)
        
        # 更新性能指标
        if entry_type == MemoryEntryType.EXECUTION:
            if success:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
            
            # 更新执行时间和token数
            if metadata:
                if "execution_time" in metadata:
                    self.total_execution_time += metadata["execution_time"]
                if "tokens" in metadata:
                    self.total_tokens += metadata["tokens"]
        
        # 如果超过最大条目数，移除最旧的条目
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
    
    def add_planning(self, 
                    sub_goal: str, 
                    selected_tool: str, 
                    tool_context: str, 
                    confidence: Optional[float] = None,
                    # GRPO算法所需的记忆系统信息
                    planner_input_ids: Optional[List[int]] = None,
                    planner_output_ids: Optional[List[int]] = None,
                    logprobs_old: Optional[List[float]] = None):
        """
        添加规划条目
        
        Args:
            sub_goal: 子目标
            selected_tool: 选择的工具
            tool_context: 工具上下文
            confidence: 置信度
            planner_input_ids: 当前轮次planner提示词的编码
            planner_output_ids: 当前轮次planner输出的编码
            logprobs_old: 当前轮次planner输出的逐token对数概率
        """
        self.add_action(
            sub_goal=sub_goal,
            selected_tool=selected_tool,
            tool_context=tool_context,
            execution_result="Planning completed",
            entry_type=MemoryEntryType.PLANNING,
            confidence=confidence,
            planner_input_ids=planner_input_ids,
            planner_output_ids=planner_output_ids,
            logprobs_old=logprobs_old
        )
    
    def add_execution(self, 
                      sub_goal: str, 
                      selected_tool: str, 
                      tool_context: str, 
                      execution_result: str,
                      success: bool = True,
                      metadata: Optional[Dict[str, Any]] = None,
                      # GRPO算法所需的记忆系统信息
                      planner_input_ids: Optional[List[int]] = None,
                      planner_output_ids: Optional[List[int]] = None,
                      logprobs_old: Optional[List[float]] = None):
        """
        添加执行条目
        
        Args:
            sub_goal: 子目标
            selected_tool: 选择的工具
            tool_context: 工具上下文
            execution_result: 执行结果
            success: 是否成功
            metadata: 元数据
            planner_input_ids: 当前轮次planner提示词的编码
            planner_output_ids: 当前轮次planner输出的编码
            logprobs_old: 当前轮次planner输出的逐token对数概率
        """
        # 更新性能指标
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        # 处理execution_result，如果它是ToolResult对象，提取其中的metadata
        actual_metadata = metadata or {}
        
        # 如果是python_executor工具，确保metadata中包含代码信息
        if selected_tool == "python_executor" and "code" not in actual_metadata:
            # 尝试从execution_result中提取代码
            if isinstance(execution_result, str) and "```python" in execution_result:
                # 尝试提取代码块
                import re
                code_match = re.search(r'```python\n(.*?)\n```', execution_result, re.DOTALL)
                if code_match:
                    actual_metadata["code"] = code_match.group(1)
        
        # 如果execution_result是字符串，尝试解析其中的ToolResult信息
        if isinstance(execution_result, str):
            try:
                # 尝试解析ToolResult的字符串表示
                if "ToolResult(" in execution_result:
                    # 提取ToolResult的metadata部分
                    start = execution_result.find("metadata=") + 9
                    end = execution_result.find(")", start)
                    if start > 8 and end > start:
                        metadata_str = execution_result[start:end]
                        if metadata_str:
                            parsed_metadata = eval(metadata_str)
                            # 合并metadata，但保留actual_metadata中的值
                            for key, value in parsed_metadata.items():
                                if key not in actual_metadata:
                                    actual_metadata[key] = value
            except:
                pass
        
        # 更新执行时间和token数
        execution_time = actual_metadata.get("execution_time", 0.0)
        tokens = actual_metadata.get("tokens", 0)
        
        # 累加到总执行时间和token数
        self.total_execution_time += execution_time
        self.total_tokens += tokens
        
        # 确保metadata包含必要的信息
        if "execution_time" not in actual_metadata:
            actual_metadata["execution_time"] = execution_time
        
        if "tokens" not in actual_metadata:
            actual_metadata["tokens"] = tokens
        
        self.add_action(
            sub_goal=sub_goal,
            selected_tool=selected_tool,
            tool_context=tool_context,
            execution_result=execution_result,
            entry_type=MemoryEntryType.EXECUTION,
            metadata=actual_metadata,
            planner_input_ids=planner_input_ids,
            planner_output_ids=planner_output_ids,
            logprobs_old=logprobs_old
        )
    
    def add_verification(self, 
                         sub_goal: str, 
                         selected_tool: str, 
                         tool_context: str, 
                         execution_result: str,
                         verification_status: int,
                         confidence: Optional[float] = None,
                         analysis: Optional[str] = None,
                         # GRPO算法所需的记忆系统信息
                         planner_input_ids: Optional[List[int]] = None,
                         planner_output_ids: Optional[List[int]] = None,
                         logprobs_old: Optional[List[float]] = None):
        """
        添加验证条目
        
        Args:
            sub_goal: 子目标
            selected_tool: 选择的工具
            tool_context: 工具上下文
            execution_result: 执行结果
            verification_status: 验证状态 (0或1)
            confidence: 置信度
            analysis: 分析说明
            planner_input_ids: 当前轮次planner提示词的编码
            planner_output_ids: 当前轮次planner输出的编码
            logprobs_old: 当前轮次planner输出的逐token对数概率
        """
        # 构建验证结果描述
        verification_result = f"验证状态: {verification_status}"
        if confidence is not None:
            verification_result += f", 置信度: {confidence:.2f}"
        if analysis:
            verification_result += f"\n分析: {analysis}"
        
        self.add_action(
            sub_goal=sub_goal,
            selected_tool=selected_tool,
            tool_context=tool_context,
            execution_result=verification_result,
            entry_type=MemoryEntryType.EVALUATION,  # 使用EVALUATION类型，但内容是验证结果
            evaluation_score=verification_status,  # 使用verification_status作为评估分数
            confidence=confidence,
            planner_input_ids=planner_input_ids,
            planner_output_ids=planner_output_ids,
            logprobs_old=logprobs_old
        )
    
    def add_evaluation(self, 
                      sub_goal: str, 
                      selected_tool: str, 
                      tool_context: str, 
                      execution_result: str,
                      evaluation_score: float,
                      confidence: Optional[float] = None):
        """
        添加评估条目
        
        Args:
            sub_goal: 子目标
            selected_tool: 选择的工具
            tool_context: 工具上下文
            execution_result: 执行结果
            evaluation_score: 评估分数
            confidence: 置信度
        """
        self.add_action(
            sub_goal=sub_goal,
            selected_tool=selected_tool,
            tool_context=tool_context,
            execution_result=execution_result,
            entry_type=MemoryEntryType.EVALUATION,
            evaluation_score=evaluation_score,
            confidence=confidence
        )
    
    def add_reasoning(self, 
                     sub_goal: str, 
                     tool_context: str, 
                     reasoning_result: str,
                     confidence: Optional[float] = None,
                     # GRPO算法所需的记忆系统信息
                     planner_input_ids: Optional[List[int]] = None,
                     planner_output_ids: Optional[List[int]] = None,
                     logprobs_old: Optional[List[float]] = None,
                     sampling_params: Optional[Dict[str, Any]] = None,
                     # 多答案采样支持
                     alternative_results: Optional[List[Dict[str, Any]]] = None):
        """
        添加推理条目
        
        Args:
            sub_goal: 子目标
            tool_context: 工具上下文
            reasoning_result: 推理结果
            confidence: 置信度
            planner_input_ids: 当前轮次planner提示词的编码
            planner_output_ids: 当前轮次planner输出的编码
            logprobs_old: 当前轮次planner输出的逐token对数概率
            sampling_params: 采样参数
            alternative_results: 替代结果列表
        """
        self.add_action(
            sub_goal=sub_goal,
            selected_tool="base_generator",
            tool_context=tool_context,
            execution_result=reasoning_result,
            entry_type=MemoryEntryType.REASONING,
            confidence=confidence,
            planner_input_ids=planner_input_ids,
            planner_output_ids=planner_output_ids,
            logprobs_old=logprobs_old,
            sampling_params=sampling_params,
            alternative_results=alternative_results
        )
    
    def add_observation(self, 
                       sub_goal: str, 
                       tool_context: str, 
                       observation_result: str):
        """
        添加观察条目
        
        Args:
            sub_goal: 子目标
            tool_context: 工具上下文
            observation_result: 观察结果
        """
        self.add_action(
            sub_goal=sub_goal,
            selected_tool="observation",
            tool_context=tool_context,
            execution_result=observation_result,
            entry_type=MemoryEntryType.OBSERVATION
        )
    
    def get_context(self, max_turns: int = 3, max_context_length: int = 4096) -> str:
        """
        获取上下文信息
        
        Args:
            max_turns: 最大回合数（降低默认值）
            max_context_length: 最大上下文长度（字符数）
            
        Returns:
            str: 格式化的上下文信息
        """
        if not self.entries:
            return "没有可用的推理历史。"
        
        # 按turn和时间戳排序，确保执行顺序正确
        sorted_entries = sorted(self.entries, key=lambda x: (x.turn, x.timestamp))
        
        # 获取最近的条目，减少每个turn的条目数量
        recent_entries = sorted_entries[-max_turns*2:]  # 减少每个turn的条目数量
        
        # 构建上下文
        context_parts = []
        for entry in recent_entries:
            # 对于执行条目，特别强调工具执行结果
            if entry.entry_type == MemoryEntryType.EXECUTION:
                # 添加成功/失败状态
                status = "成功" if entry.success else "失败"
                # 限制每个条目的长度
                tool_context = entry.tool_context[:500] + "..." if len(entry.tool_context) > 500 else entry.tool_context
                execution_result = entry.execution_result[:500] + "..." if len(entry.execution_result) > 500 else entry.execution_result
                
                context_parts.append(
                    f"回合 {entry.turn} [{entry.entry_type.value.upper()}]:\n"
                    f"子目标: {entry.sub_goal}\n"
                    f"使用的工具: {entry.selected_tool}\n"
                    f"执行状态: {status}\n"
                    f"工具上下文: {tool_context}\n"
                    f"工具执行结果: {execution_result}\n"
                )
            elif entry.entry_type == MemoryEntryType.EVALUATION:
                # 对于评估条目，强调验证状态
                verification_status = entry.evaluation_score
                status = "验证通过" if verification_status == 1 else "验证失败"
                # 限制每个条目的长度
                tool_context = entry.tool_context[:500] + "..." if len(entry.tool_context) > 500 else entry.tool_context
                execution_result = entry.execution_result[:500] + "..." if len(entry.execution_result) > 500 else entry.execution_result
                
                context_parts.append(
                    f"回合 {entry.turn} [{entry.entry_type.value.upper()}]:\n"
                    f"子目标: {entry.sub_goal}\n"
                    f"使用的工具: {entry.selected_tool}\n"
                    f"验证状态: {status} (分数: {verification_status}, 置信度: {entry.confidence})\n"
                    f"工具上下文: {tool_context}\n"
                    f"验证结果: {execution_result}\n"
                )
            else:
                # 限制每个条目的长度
                tool_context = entry.tool_context[:500] + "..." if len(entry.tool_context) > 500 else entry.tool_context
                execution_result = entry.execution_result[:500] + "..." if len(entry.execution_result) > 500 else entry.execution_result
                
                context_parts.append(
                    f"回合 {entry.turn} [{entry.entry_type.value.upper()}]:\n"
                    f"子目标: {entry.sub_goal}\n"
                    f"工具: {entry.selected_tool}\n"
                    f"上下文: {tool_context}\n"
                    f"结果: {execution_result}\n"
                )
        
        context = "\n".join(context_parts)
        
        # 如果上下文仍然太长，进一步截断
        if len(context) > max_context_length:
            # 保留开头和结尾
            keep_start = max_context_length // 2
            keep_end = max_context_length - keep_start
            context = context[:keep_start] + "\n...[Context truncated]...\n" + context[-keep_end:]
            #logger.warning(f"Context too long, truncated to {max_context_length} characters")
        
        return context
    
    def get_formatted_context(self, max_turns: int = 3, max_context_length: int = 4096) -> str:
        """
        获取格式化的上下文信息（用于提示词）
        
        Args:
            max_turns: 最大回合数（降低默认值）
            max_context_length: 最大上下文长度（字符数）
            
        Returns:
            str: 格式化的上下文信息
        """
        if not self.entries:
            return "No previous actions."
        
        # 按turn和时间戳排序，确保执行顺序正确
        sorted_entries = sorted(self.entries, key=lambda x: (x.turn, x.timestamp))
        
        # 获取最近的条目，减少每个turn的条目数量
        recent_entries = sorted_entries[-max_turns*2:]  # 减少每个turn的条目数量
        
        # 构建上下文
        context_parts = ["Previous Actions:"]
        for entry in recent_entries:
            # 限制每个条目的长度
            tool_context = entry.tool_context[:300] + "..." if len(entry.tool_context) > 300 else entry.tool_context
            execution_result = entry.execution_result[:300] + "..." if len(entry.execution_result) > 300 else entry.execution_result
            
            context_parts.append(
                f"Turn {entry.turn} [{entry.entry_type.value.upper()}]:\n"
                f"Sub-Goal: {entry.sub_goal}\n"
                f"Tool: {entry.selected_tool}\n"
                f"Context: {tool_context}\n"
                f"Result: {execution_result}\n"
            )
        
        context = "\n".join(context_parts)
        
        # 如果上下文仍然太长，进一步截断
        if len(context) > max_context_length:
            # 保留开头和结尾
            keep_start = max_context_length // 2
            keep_end = max_context_length - keep_start
            context = context[:keep_start] + "\n...[Context truncated]...\n" + context[-keep_end:]
            #logger.warning(f"Context too long, truncated to {max_context_length} characters")
        
        return context
    
    def get_last_entry(self) -> Optional[MemoryEntry]:
        """获取最后一个条目"""
        return self.entries[-1] if self.entries else None
    
    def get_last_execution_result(self) -> str:
        """获取最后一个执行结果"""
        last_entry = self.get_last_entry()
        return last_entry.execution_result if last_entry else ""
    
    def get_required_skills(self) -> List[str]:
        """获取所需技能列表"""
        return self.required_skills
    
    def get_used_tools(self) -> List[str]:
        """获取已使用的工具列表"""
        return list(self.used_tools)
    
    def is_terminated(self) -> bool:
        """检查是否已终止"""
        return self.terminated
    
    def set_terminated(self, reason: str = ""):
        """
        设置终止状态
        
        Args:
            reason: 终止原因
        """
        self.terminated = True
        self.termination_reason = reason
        
        # 添加终止观察
        self.add_observation(
            sub_goal="Process termination",
            tool_context=f"Termination reason: {reason}",
            observation_result="Process terminated"
        )
    
    def next_turn(self):
        """进入下一回合"""
        self.current_turn += 1
        # 确保新turn有计数器和开始时间
        if self.current_turn not in self.turn_entry_count:
            self.turn_entry_count[self.current_turn] = 0
        self.turn_start_time[self.current_turn] = time.time()
        print(f"进入第 {self.current_turn} 回合")
    
    def get_turn_duration(self, turn: Optional[int] = None) -> float:
        """
        获取指定turn的持续时间
        
        Args:
            turn: 回合数，如果为None则返回当前turn
            
        Returns:
            float: 持续时间（秒）
        """
        if turn is None:
            turn = self.current_turn
            
        if turn not in self.turn_start_time:
            return 0.0
            
        # 如果是当前turn，计算到现在的持续时间
        if turn == self.current_turn:
            return time.time() - self.turn_start_time[turn]
            
        # 如果是过去的turn，需要找到下一个turn的开始时间
        if turn + 1 in self.turn_start_time:
            return self.turn_start_time[turn + 1] - self.turn_start_time[turn]
            
        # 如果没有下一个turn，使用最后一个条目的时间戳
        turn_entries = [e for e in self.entries if e.turn == turn]
        if turn_entries:
            last_entry = max(turn_entries, key=lambda x: x.timestamp)
            return last_entry.timestamp - self.turn_start_time[turn]
            
        return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        total_executions = self.successful_executions + self.failed_executions
        success_rate = self.successful_executions / total_executions if total_executions > 0 else 0
        
        # 计算平均每turn的持续时间
        turn_durations = []
        for turn in range(self.current_turn + 1):
            duration = self.get_turn_duration(turn)
            if duration > 0:
                turn_durations.append(duration)
        
        avg_turn_duration = sum(turn_durations) / len(turn_durations) if turn_durations else 0
        
        return {
            "total_turns": self.current_turn,
            "total_entries": len(self.entries),
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "total_tokens": self.total_tokens,
            "total_execution_time": self.total_execution_time,
            "used_tools": self.get_used_tools(),
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "turn_entry_count": self.turn_entry_count,
            "avg_turn_duration": avg_turn_duration
        }
    
    def save_to_file(self, file_path: str):
        """
        保存记忆到文件
        
        Args:
            file_path: 文件路径
        """
        import os
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        data = {
            "initial_query": self.initial_query,
            "initial_context": self.initial_context,
            "required_skills": self.required_skills,
            "current_turn": self.current_turn,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "performance_metrics": self.get_performance_metrics(),
            "turn_entry_count": self.turn_entry_count,
            "turn_start_time": self.turn_start_time,
            "entries": [entry.to_dict() for entry in self.entries]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, file_path: str):
        """
        从文件加载记忆
        
        Args:
            file_path: 文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.initial_query = data.get("initial_query", "")
        self.initial_context = data.get("initial_context", "")
        self.required_skills = data.get("required_skills", [])
        self.current_turn = data.get("current_turn", 0)
        self.terminated = data.get("terminated", False)
        self.termination_reason = data.get("termination_reason", "")
        self.turn_entry_count = data.get("turn_entry_count", {0: 0})
        self.turn_start_time = data.get("turn_start_time", {0: time.time()})
        
        # 加载条目
        self.entries = []
        for entry_data in data.get("entries", []):
            self.entries.append(MemoryEntry.from_dict(entry_data))
        
        # 更新使用的工具集合
        self.used_tools = set(entry.selected_tool for entry in self.entries)
    
    def get_training_data(self) -> Dict[str, Any]:
        """
        获取用于强化学习的训练数据
        
        Returns:
            Dict[str, Any]: 训练数据
        """
        # 按turn分组条目
        turn_groups = {}
        for entry in self.entries:
            if entry.turn not in turn_groups:
                turn_groups[entry.turn] = {}
            turn_groups[entry.turn][entry.entry_type.value] = entry
        
        # 构建轨迹数据
        trajectory = []
        for turn in sorted(turn_groups.keys()):
            turn_data = turn_groups[turn]
            
            # 确保有完整的planning和execution数据
            if "planning" in turn_data and "execution" in turn_data:
                planning = turn_data["planning"]
                execution = turn_data["execution"]
                
                turn_trajectory = {
                    "turn": turn,
                    "planning": {
                        "sub_goal": planning.sub_goal,
                        "selected_tool": planning.selected_tool,
                        "tool_context": planning.tool_context,
                        "confidence": planning.confidence
                    },
                    "execution": {
                        "result": execution.execution_result,
                        "success": execution.success,
                        "metadata": execution.metadata
                    }
                }
                
                # 添加evaluation数据（如果有）
                if "evaluation" in turn_data:
                    evaluation = turn_data["evaluation"]
                    turn_trajectory["evaluation"] = {
                        "score": evaluation.evaluation_score,
                        "confidence": evaluation.confidence
                    }
                    
                trajectory.append(turn_trajectory)
        
        return {
            "query": self.initial_query,
            "context": self.initial_context,
            "trajectory": trajectory,
            "performance": self.get_performance_metrics(),
            "success": self.terminated and "success" in self.termination_reason.lower()
        }
    
    def get_recent_entries(self, max_entries: int = 10) -> List[MemoryEntry]:
        """
        获取最近的记忆条目
        
        Args:
            max_entries: 最大条目数
            
        Returns:
            List[MemoryEntry]: 最近的记忆条目列表
        """
        if not self.entries:
            return []
        
        # 按时间戳排序，获取最近的条目
        sorted_entries = sorted(self.entries, key=lambda x: x.timestamp)
        return sorted_entries[-max_entries:]
    
    def get_failed_entries(self, max_entries: int = 10) -> List[MemoryEntry]:
        """
        获取失败的记忆条目
        
        Args:
            max_entries: 最大条目数
            
        Returns:
            List[MemoryEntry]: 失败的记忆条目列表
        """
        # 筛选失败的执行条目
        failed_entries = [
            entry for entry in self.entries 
            if entry.entry_type == MemoryEntryType.EXECUTION and not entry.success
        ]
        
        # 按时间戳排序，获取最近的失败条目
        failed_entries.sort(key=lambda x: x.timestamp)
        return failed_entries[-max_entries:]
    
    def get_failure_patterns(self, max_entries: int = 10) -> Dict[str, Any]:
        """
        分析失败模式
        
        Args:
            max_entries: 最大分析条目数
            
        Returns:
            Dict[str, Any]: 失败模式分析结果
        """
        failed_entries = self.get_failed_entries(max_entries)
        
        if not failed_entries:
            return {
                "total_failures": 0,
                "failed_tools": {},
                "failure_reasons": {},
                "recommendations": []
            }
        
        # 统计失败工具
        failed_tools = {}
        failure_reasons = {}
        
        for entry in failed_entries:
            # 统计失败的工具
            tool = entry.selected_tool
            if tool not in failed_tools:
                failed_tools[tool] = 0
            failed_tools[tool] += 1
            
            # 尝试从执行结果中提取失败原因
            result = entry.execution_result
            if isinstance(result, str):
                # 简单的关键词匹配来识别失败原因
                if "error" in result.lower() or "exception" in result.lower():
                    reason = "error"
                elif "timeout" in result.lower():
                    reason = "timeout"
                elif "permission" in result.lower() or "denied" in result.lower():
                    reason = "permission"
                elif "not found" in result.lower() or "does not exist" in result.lower():
                    reason = "not_found"
                else:
                    reason = "unknown"
                
                if reason not in failure_reasons:
                    failure_reasons[reason] = 0
                failure_reasons[reason] += 1
        
        # 生成建议
        recommendations = []
        for tool, count in failed_tools.items():
            if count >= 2:  # 同一工具失败多次
                recommendations.append(f"工具 '{tool}' 失败了 {count} 次，考虑使用替代工具")
        
        for reason, count in failure_reasons.items():
            if reason == "error" and count >= 2:
                recommendations.append("多次出现错误，建议检查输入参数和工具使用方式")
            elif reason == "timeout" and count >= 2:
                recommendations.append("多次超时，建议增加超时时间或优化操作")
            elif reason == "permission" and count >= 1:
                recommendations.append("出现权限问题，建议检查文件或资源访问权限")
            elif reason == "not_found" and count >= 2:
                recommendations.append("多次找不到资源，建议检查路径和资源名称")
        
        return {
            "total_failures": len(failed_entries),
            "failed_tools": failed_tools,
            "failure_reasons": failure_reasons,
            "recommendations": recommendations
        }
    
    def get_alternative_tools(self, failed_tool: str) -> List[str]:
        """
        获取失败工具的替代工具建议
        
        Args:
            failed_tool: 失败的工具名称
            
        Returns:
            List[str]: 替代工具列表
        """
        # 定义工具替代映射
        tool_alternatives = {
            "python_executor": ["base_generator", "tool_manager"],
            "base_generator": ["python_executor"],
            "tool_manager": ["base_generator", "python_executor"],
            "observation": ["base_generator"]
        }
        
        return tool_alternatives.get(failed_tool, [])
    
    def get_trajectory_data(self) -> Dict[str, Any]:
        """
        获取轨迹数据，包含token概率信息，用于GRPO算法训练
        
        Returns:
            Dict[str, Any]: 轨迹数据，包含token概率信息
        """
        # 按turn分组条目
        turn_groups = {}
        for entry in self.entries:
            if entry.turn not in turn_groups:
                turn_groups[entry.turn] = {}
            turn_groups[entry.turn][entry.entry_type.value] = entry
        
        # 构建轨迹数据
        trajectory = []
        for turn in sorted(turn_groups.keys()):
            turn_data = turn_groups[turn]
            
            # 确保有完整的planning和execution数据
            if "planning" in turn_data and "execution" in turn_data:
                planning = turn_data["planning"]
                execution = turn_data["execution"]
                
                turn_trajectory = {
                    "turn": turn,
                    "planning": {
                        "sub_goal": planning.sub_goal,
                        "selected_tool": planning.selected_tool,
                        "tool_context": planning.tool_context,
                        "confidence": planning.confidence,
                        # 添加GRPO算法所需的记忆系统信息
                        "planner_input_ids": planning.planner_input_ids,
                        "planner_output_ids": planning.planner_output_ids,
                        "logprobs_old": planning.logprobs_old,
                        "sampling_params": planning.sampling_params
                    },
                    "execution": {
                        "result": execution.execution_result,
                        "success": execution.success,
                        "metadata": execution.metadata,
                        # 添加GRPO算法所需的记忆系统信息
                        "planner_input_ids": execution.planner_input_ids,
                        "planner_output_ids": execution.planner_output_ids,
                        "logprobs_old": execution.logprobs_old,
                        "sampling_params": execution.sampling_params
                    }
                }
                
                # 添加evaluation数据（如果有）
                if "evaluation" in turn_data:
                    evaluation = turn_data["evaluation"]
                    turn_trajectory["evaluation"] = {
                        "score": evaluation.evaluation_score,
                        "confidence": evaluation.confidence,
                        # 添加GRPO算法所需的记忆系统信息
                        "planner_input_ids": evaluation.planner_input_ids,
                        "planner_output_ids": evaluation.planner_output_ids,
                        "logprobs_old": evaluation.logprobs_old,
                        "sampling_params": evaluation.sampling_params
                    }
                
                # 添加reasoning数据（如果有）
                if "reasoning" in turn_data:
                    reasoning = turn_data["reasoning"]
                    turn_trajectory["reasoning"] = {
                        "result": reasoning.execution_result,
                        "confidence": reasoning.confidence,
                        # 添加GRPO算法所需的记忆系统信息
                        "planner_input_ids": reasoning.planner_input_ids,
                        "planner_output_ids": reasoning.planner_output_ids,
                        "logprobs_old": reasoning.logprobs_old,
                        "sampling_params": reasoning.sampling_params
                    }
                    
                trajectory.append(turn_trajectory)
        
        return {
            "query": self.initial_query,
            "context": self.initial_context,
            "trajectory": trajectory,
            "performance": self.get_performance_metrics(),
            "success": self.terminated and "success" in self.termination_reason.lower(),
            "trajectory_id": self.trajectory_id
        }
    
    def clear(self):
        """清空记忆"""
        self.entries.clear()
        self.current_turn = 0
        self.terminated = False
        self.termination_reason = ""
        self.used_tools.clear()
        self.total_tokens = 0
        self.total_execution_time = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.turn_entry_count = {0: 0}
        self.turn_start_time = {0: time.time()}
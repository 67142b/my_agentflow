from numpy import extract
import uuid
"""
Executor Module for AgentFlow
执行器：执行工具，处理结果，验证执行
"""

import time
import json
import re
import os
import yaml
from typing import Dict, Any, Optional, List
from dashscope import Generation
import dashscope
from .prompt_templates import PromptTemplates
from ..tools.python_executor import PythonExecutorTool
from ..tools.base_generator import BaseGenerator
from ..tools.web_search import DuckDuckGoSearchTool, GoogleCustomSearchTool
from ..tools.wikipedia_search import WikipediaSearchTool
from ...LLM_engine import get_llm_engine
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Executor:
    """
    执行器类
    
    职责：
    1. 执行工具调用
    2. 处理执行结果
    3. 生成执行报告
    4. 管理执行状态和错误
    """
    
    def __init__(self, config_path: str = "config.yaml", tools: Optional[Dict[str, Any]] = None, llm_engine=None):
        """
        初始化执行器
        
        Args:
            config_path: 配置文件路径
            tools: 外部传入的工具字典
        """
        self.config = self._load_config(config_path)
        
        # 初始化API配置
        self.api_key = self.config.get("model",{}).get("dashscope_api_key", "")
        self.api_url = self.config.get("model",{}).get("dashscope_base_url", "")
        self.model_name = self.config.get("model",{}).get("executor_model_name", "")
        self.timeout = 60
        
        # 初始化LLM引擎
        self.llm_engine = llm_engine
        
        # 初始化执行状态跟踪（在工具初始化之前）
        self.execution_history = []
        self.consecutive_failures = 0
        self.max_consecutive_failures = self.config.get("max_consecutive_failures", 3)
        self.critical_errors = []
        self.is_shared_planner = self.config.get("model", {}).get("executor_is_shared_planner", False)
        
        # 初始化工具
        self.tools = tools if tools is not None else {}
        if tools is None:
            # 只有在没有提供外部工具时才初始化工具
            self._initialize_tools()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _initialize_tools(self):
        """初始化所有工具"""
        # 初始化Python执行器
        self.tools["python_executor"] = PythonExecutorTool()
        #logger.info("Python执行器初始化成功")
        
        # 初始化基础生成器
        self.tools["base_generator"] = BaseGenerator()
        #logger.info("基础生成器初始化成功")
        
        # 初始化搜索工具
        wiki_language = self.config.get("tools", {}).get("wikipedia_language", "en")
        self.tools["wikipedia_search"] = WikipediaSearchTool(language=wiki_language)
        #logger.info(f"维基百科搜索工具初始化成功（语言：{wiki_language}）")
        
        self.tools["duckduckgo_search"] = DuckDuckGoSearchTool()
        #logger.info("DuckDuckGo搜索工具初始化成功")
        
        self.tools["google_search"] = GoogleCustomSearchTool()
        #logger.info("Google搜索工具初始化成功")
        
        # 添加web_search作为duckduckgo_search的别名
        self.tools["web_search"] = self.tools[self.config.get("tools", {}).get("preferred_search_tool", "duckduckgo_search")]
        #logger.info(f"web_search别名设置为: {self.config.get('tools', {}).get('preferred_search_tool', 'duckduckgo_search')}")
        
        #logger.info(f"所有工具初始化完成，共初始化 {len(self.tools)} 个工具")
    
    def get_tool_metadata(self) -> Dict[str, Dict[str, Any]]:
        """获取所有工具的元数据"""
        metadata = {}
        for name, tool in self.tools.items():
            metadata[name] = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_parameters()
            }
        return metadata
    
    def execute(self, tool_name: str, parameters: Dict[str, Any],trajectory_id: int) -> Dict[str, Any]:
        """
        执行工具调用
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        
        # 记录执行开始
        execution_record = {
            "id": execution_id,
            "tool_name": tool_name,
            "parameters": parameters,
            "start_time": start_time,
            "status": "running"
        }
        
        # 检查工具是否存在
        if tool_name not in self.tools:
            raise ValueError(f"工具 {tool_name} 不存在")
        
        tool = self.tools[tool_name]
        
        # 验证参数
        self._validate_parameters(tool, parameters)
        
        # 执行工具
        #logger.info(f"执行工具 {tool_name}，轨迹ID: {trajectory_id}")
        
        # 根据工具类型处理参数
        if tool_name == "python_executor":
            # python_executor工具需要单独的query和context参数
            #print(f"代码工具原始查询: \n=======================================================\n{query}\n=======================================================")
            query = parameters.get("query", "")
            context = parameters.get("context", "")
            result = tool.execute(query, context)
        elif tool_name == "base_generator":
            # base_generator工具需要单独的query和context参数
            #print(f"大模型工具原始查询: \n=======================================================\n{query}\n=======================================================")
            query = parameters.get("query", "")
            context = parameters.get("context", "")
            #print("=======================================调用了工具base_generator=======================================")
            result = tool.execute(query, context)
        elif tool_name in ["duckduckgo_search", "enhanced_wikipedia_search", "wikipedia_search", "web_search","google_custom_search"]:
            # 搜索工具只需要query参数
            query = parameters.get("raw_query", "")
            context = parameters.get("context", None)
            #print(f"搜索工具原始查询: \n=======================================================\n{query}\n=======================================================")
            
            # 对于wikipedia_search和web_search工具，使用大模型提取出真正的查询关键词
            if tool_name in ["wikipedia_search", "web_search","duckduckgo_search","google_custom_search"]:
                # 对于wikipedia_search，移除特定前缀和后缀
                if tool_name == "wikipedia_search":
                    # 移除"Search Wikipedia for:"前缀
                    if query.startswith("Search Wikipedia for:"):
                        query = query.replace("Search Wikipedia for:", "").strip()
                        # 移除",工具强制选择wikipedia_search"后缀
                        if ",工具强制选择wikipedia_search" in query:
                            query = query.replace(",工具强制选择wikipedia_search", "").strip()
                
                # 对于web_search，也进行类似的清理
                if tool_name in ["web_search","duckduckgo_search","google_custom_search"]:
                    # 移除"Search the web for:"前缀
                    if query.startswith("Search the web for:"):
                        query = query.replace("Search the web for:", "").strip()
                        # 移除",工具强制选择web_search"后缀
                        if ",工具强制选择web_search" in query:
                            query = query.replace(",工具强制选择web_search", "").strip()
                
                # 使用大模型从查询和上下文中提取关键词
                tool_type = "wikipedia" if tool_name == "wikipedia_search" else "web"
                #query = self._extract_search_keywords(query, context, tool_type)
                #print(f"提取后的关键词: \n=======================================================\n{query}\n=======================================================")
            
            result = tool.execute(query, context)
        else:
            # 其他工具使用原始参数字典
            result = tool.execute(parameters)
        
        # 确保执行结果被正确格式化
        #logger.info(f"工具 {tool_name} 执行成功\n")
        if hasattr(result, 'success') and hasattr(result, 'result'):
            # 如果是ToolResult对象，提取相关信息
            formatted_result = {
                "success": result.success,
                "result": result.result,
                "error": getattr(result, 'error', None),
                "metadata": getattr(result, 'metadata', {})
            }
        elif isinstance(result, dict):
            # 如果是字典，确保包含必要的字段
            formatted_result = result.copy()
            if "success" not in formatted_result:
                formatted_result["success"] = True
        else:
            # 其他类型，包装为标准格式
            formatted_result = {
                "success": True,
                "result": str(result),
                "metadata": {}
            }
        
        # 验证执行结果
        validated_result = self._validate_result(tool_name, formatted_result)

        # 如果是搜索工具，应用搜索增强
        if tool_name in ["web_search", "wikipedia_search", "duckduckgo_search", "google_search"]:
            original_query = parameters.get("raw_query", "")
            print("使用搜索增强")
            enhanced_result = self._search_enhancement(original_query, tool_name,validated_result.get("result", ""))
            
            # 如果搜索增强成功，使用增强后的结果
            if enhanced_result.get("success", False):
                validated_result = enhanced_result
                #logger.info(f"✓ 搜索增强已应用，工具: {tool_name}")
            else:
                logger.warning(f"搜索增强失败，使用原始结果，工具: {tool_name}")

        # 记录执行成功
        execution_time = time.time() - start_time
        execution_record.update({
            "status": "success",
            "result": validated_result,
            "execution_time": execution_time,
            "end_time": time.time()
        })
        
        # 重置连续失败计数
        self.consecutive_failures = 0
        
        #logger.info(f"工具 {tool_name} 执行成功，结果: \n{validated_result.get('result', '')} \n耗时: {execution_time:.2f}秒")
        
        # 记录执行历史
        self.execution_history.append(execution_record)
        
        # 限制执行历史长度
        if len(self.execution_history) > 100:
            self.execution_history.pop(0)
        
        return {
            "success": True,
            "result": validated_result.get("result", ""),
            "execution_time": execution_time,
            "execution_id": execution_id,
            "metadata": validated_result.get("metadata", {})
        }
    
    def _extract_search_keywords(self, query: str, context: str = None, tool_type: str = "wikipedia") -> str:
        """
        使用大模型从查询和上下文中提取搜索关键词
        
        Args:
            query: 原始查询
            context: 上下文信息
            tool_type: 工具类型，可以是"wikipedia"或"web"
            
        Returns:
            提取的关键词
        """
            # 首先移除工具强制选择的前缀和后缀
        clean_query = query
        
        # 构建提示词
        if tool_type == "wikipedia":
            prompt = f"""Please extract a suitable keyword from the following query for Wikipedia search. The keyword should be concise, clear, and directly usable for searching.

Original query: {clean_query}
Please return only the extracted keyword, without any explanation or additional content. The keyword should be english, no more than 20 words, and preferably one or two sentences."""
        else:  # web_search
            prompt = f"""Please extract a suitable keyword from the following query for web search engine. The keyword should be concise, clear, and directly usable for searching.

Original query: {clean_query}
Please return only the extracted keyword, without any explanation or additional content. The keyword should be english, no more than 20 words, and preferably one or two sentences."""
        
        # 调用LLM引擎生成关键词，共享planner的权重和分词器
        response = self.llm_engine.generate(
            model_source="local" if self.is_shared_planner else "dashscope",
            model_name=self.model_name,  # 使用executor作为模型类型，会自动共享planner的权重
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.5
        )
        keyword = response
        
        # 移除可能的引号
        keyword = keyword.strip('"\'')
        # 确保关键词不为空
        if keyword and len(keyword) > 0:
            #logger.info(f"使用LLM引擎提取的关键词: {keyword}")
            return keyword
        else:
            logger.warning("LLM引擎返回空关键词，使用原始查询")
            return query  # 返回查询的前20个字符
                

    
    def _search_enhancement(self, original_query: str, search_tool: str, search_results: Any) -> Dict[str, Any]:
        """
        搜索增强方法：从搜索结果中总结与解决原始问题相关的信息
        
        Args:
            original_query: 原始查询
            search_tool: 使用的搜索工具
            search_results: 搜索结果
            
        Returns:
            Dict[str, Any]: 包含增强后的搜索结果的字典
        """
        if len(search_results) > 5000:
            search_results = search_results[:5000]
        
        # 获取搜索增强提示词
        prompt = f"""You are a search result analyzer in an agentic reasoning system.

Your task is to Summarise the search results and extract the key information.

IMPORTANT: Output ONLY the key information related to the original query,do not response your own thoughts or reasoning.

[Original Query]
{original_query}

[Search Results]
{search_results}

Now,Summarise the search results:"""

        #print(f"搜索增强提示词:\n{prompt}")
        
        # 调用LLM引擎进行搜索增强，共享planner的权重和分词器
        response = self.llm_engine.generate(
            model_source="local" if self.is_shared_planner else "dashscope",
            model_name=self.model_name,  # 使用executor作为模型类型，会自动共享planner的权重
            prompt=prompt,
            max_new_tokens=1000,
            temperature=0.5
        )
        enhanced_result = response
        
        #logger.info(f"搜索增强成功，\n工具: {search_tool},结果: {enhanced_result}\n")
        
        # 返回增强后的结果
        return {
            "success": True,
            "result": enhanced_result,
            "original_results": search_results,
            "enhancement_applied": True,
            "metadata": {
                "search_tool": search_tool,
                "original_query": original_query,
                "enhancement_method": "llm_summarization"
            }
        }
                
    
    def _validate_parameters(self, tool: Any, parameters: Dict[str, Any]):
        """验证工具参数"""
        tool_params = tool.get_parameters()
        
        # 检查必需参数
        required_params = tool_params.get("required", [])
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"缺少必需参数: {param}")
        
        # 检查参数类型
        properties = tool_params.get("properties", {})
        for param_name, param_value in parameters.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                if expected_type and not self._check_type(param_value, expected_type):
                    raise ValueError(f"参数 {param_name} 类型错误，期望 {expected_type}")
                        

    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """检查值类型"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # 未知类型，跳过检查
            
        return isinstance(value, expected_python_type)
    
    def _validate_result(self, tool_name: str, result: Any) -> Any:
        """验证执行结果"""
        if result is None:
            raise ValueError("执行结果为空")
            
        # 根据工具类型进行特定验证
        if tool_name == "python_executor":
            # 检查是否是ToolResult对象
            if hasattr(result, 'success') and hasattr(result, 'result'):
                # 如果是ToolResult对象，检查success字段
                if not result.success:
                    raise ValueError(f"Python执行器执行失败: {result.error}")
                # 转换为标准字典格式
                return {
                    "success": result.success,
                    "result": result.result,
                    "error": getattr(result, 'error', None),
                    "metadata": getattr(result, 'metadata', {})
                }
            # 如果是字典，检查是否有output或result字段
            elif isinstance(result, dict):
                if "output" not in result and "result" not in result:
                    raise ValueError("Python执行器结果格式错误")
                # 确保字典有必要的字段
                if "success" not in result:
                    result["success"] = True
                if "result" not in result and "output" in result:
                    result["result"] = result["output"]
                return result
            else:
                raise ValueError("Python执行器结果格式错误")
                
        elif tool_name == "base_generator":
            # 检查是否是ToolResult对象
            if hasattr(result, 'success') and hasattr(result, 'result'):
                if not result.success:
                    raise ValueError(f"生成器执行失败: {result.error}")
                if not result.result or not result.result.strip():
                    raise ValueError("生成器结果为空")
                # 转换为标准字典格式
                return {
                    "success": result.success,
                    "result": result.result,
                    "error": getattr(result, 'error', None),
                    "metadata": getattr(result, 'metadata', {})
                }
            # 如果是字符串，检查是否为空
            elif isinstance(result, str):
                if not result.strip():
                    raise ValueError("生成器结果为空")
                # 包装为标准格式
                return {
                    "success": True,
                    "result": result,
                    "metadata": {}
                }
            # 如果是字典，检查是否有result字段
            elif isinstance(result, dict):
                if "result" not in result:
                    raise ValueError("生成器结果格式错误")
                # 确保字典有必要的字段
                if "success" not in result:
                    result["success"] = True
                return result
            else:
                raise ValueError("生成器结果格式错误")
                
        elif "search" in tool_name:
            # 检查是否是ToolResult对象
            if hasattr(result, 'success') and hasattr(result, 'result'):
                if not result.success:
                    raise ValueError(f"搜索工具执行失败: {result.error}")
                # 转换为标准字典格式
                return {
                    "success": result.success,
                    "result": result.result,
                    "error": getattr(result, 'error', None),
                    "metadata": getattr(result, 'metadata', {})
                }
            # 如果是字典，检查是否有results字段
            elif isinstance(result, dict):
                if "result" not in result and "results" not in result:
                    raise ValueError("搜索结果格式错误")
                # 确保字典有必要的字段
                if "success" not in result:
                    result["success"] = True
                return result
            # 如果是列表，包装为标准格式
            elif isinstance(result, list):
                return {
                    "success": True,
                    "result": result,
                    "metadata": {}
                }
            else:
                raise ValueError("搜索结果应为列表或ToolResult对象")
        
        # 默认情况，返回结果本身
        return result
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history[-limit:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0,
                "average_execution_time": 0,
                "consecutive_failures": 0,
                "critical_errors": len(self.critical_errors)
            }
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for e in self.execution_history if e["status"] == "success")
        failed_executions = total_executions - successful_executions
        success_rate = successful_executions / total_executions
        
        execution_times = [e["execution_time"] for e in self.execution_history if "execution_time" in e]
        average_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": success_rate,
            "average_execution_time": average_execution_time,
            "consecutive_failures": self.consecutive_failures,
            "critical_errors": len(self.critical_errors),
            "last_errors": self.critical_errors[-5:]  # 最近5个关键错误
        }
    
    def reset_error_counters(self):
        """重置错误计数器"""
        self.consecutive_failures = 0
        self.critical_errors = []
        #logger.info("错误计数器已重置")
    
    def is_healthy(self) -> bool:
        """检查执行器健康状态"""
        return (
            self.consecutive_failures < self.max_consecutive_failures and
            len(self.critical_errors) == 0
        )
    
    def _call_dashscope_api(self, prompt: str) -> str:
        """调用DashScope API"""
        # 使用LLM引擎的generate方法
        response = self.llm_engine.generate(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=self.model_name,
            max_tokens=8192,
            temperature=0.5
        )
        
        if response.choices:
            return response.choices[0].message.content
        else:
            raise Exception("API调用失败: 无响应")
                


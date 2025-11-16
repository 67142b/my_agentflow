"""
组件交互适配器
提供统一的接口，解决不同组件间的兼容性问题
"""

import logging
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ComponentAdapter(ABC):
    """组件适配器基类"""
    
    @abstractmethod
    def adapt_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """适配输入数据"""
        pass
    
    @abstractmethod
    def adapt_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """适配输出数据"""
        pass


class PlannerToExecutorAdapter(ComponentAdapter):
    """规划器到执行器的适配器"""
    
    def adapt_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将规划器输出适配为执行器输入
        
        Args:
            data: 规划器输出数据
            
        Returns:
            执行器输入数据
        """
        try:
            # 提取规划器输出中的工具和参数
            plan = data.get("plan", {})
            
            # 尝试从不同位置获取工具信息
            # 1. 首先尝试从plan对象中获取
            selected_tool = plan.get("selected_tool", "") if plan else ""
            tool_context = plan.get("tool_context", "") if plan else ""
            sub_goal = plan.get("sub_goal", "") if plan else ""
            
            # 2. 如果plan中没有，尝试从plan中获取tool_name
            if not selected_tool and plan:
                selected_tool = plan.get("tool_name", "")
                if not tool_context:
                    tool_context = plan.get("context", "")
                if not sub_goal:
                    sub_goal = plan.get("sub_goal", "")
            
            # 3. 如果仍然没有，直接从data中获取
            if not selected_tool:
                selected_tool = data.get("selected_tool", "")
                if not selected_tool:
                    selected_tool = data.get("tool_name", "")
            
            if not tool_context:
                tool_context = data.get("tool_context", "")
                if not tool_context:
                    tool_context = data.get("context", "")
            
            if not sub_goal:
                sub_goal = data.get("sub_goal", "")
            
            # 解析工具参数
            parameters = {
                "query": tool_context,
                "context": sub_goal
            }
            
            # 构建执行器输入
            return {
                "tool_name": selected_tool,
                "parameters": parameters,
                "sub_goal": sub_goal,
                "selected_tool": selected_tool,
                "tool_context": tool_context
            }
        except Exception as e:
            logger.error(f"规划器到执行器适配失败: {e}")
            return {
                "tool_name": "",
                "parameters": {},
                "sub_goal": "",
                "selected_tool": "",
                "tool_context": ""
            }
    
    def adapt_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将执行器输出适配为规划器可理解的格式
        
        Args:
            data: 执行器输出数据
            
        Returns:
            规划器可理解的格式
        """
        try:
            # 执行器输出已经是标准格式，直接返回
            return data
        except Exception as e:
            logger.error(f"执行器输出适配失败: {e}")
            return {
                "success": False,
                "error": f"适配失败: {e}",
                "result": ""
            }


class ExecutorToVerifierAdapter(ComponentAdapter):
    """执行器到验证器的适配器"""
    
    def adapt_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将执行器输出适配为验证器输入
        
        Args:
            data: 执行器输出数据
            
        Returns:
            验证器输入数据
        """
        try:
            # 提取执行器输出，增加多层嵌套数据结构的支持
            execution_result = data.get("result", "")
            if not execution_result and "data" in data:
                execution_result = data["data"].get("result", "")
            
            # 提取工具名称，增加更多备选字段
            tool_name = (data.get("tool_name", "") or 
                        data.get("selected_tool", "") or
                        data.get("tool", ""))
            
            # 提取子目标，支持嵌套结构
            sub_goal = data.get("sub_goal", "")
            if not sub_goal and "context" in data:
                sub_goal = data["context"].get("sub_goal", "")
            
            # 提取额外的执行信息
            execution_info = {
                "success": data.get("success", True),
                "error": data.get("error", ""),
                "parameters": data.get("parameters", {})
            }
            
            # 构建工具信息
            tool_info = {
                "name": tool_name,
                "description": f"工具 {tool_name} 的执行结果",
                "execution_info": execution_info
            }
            
            # 构建验证器输入
            return {
                "sub_goal": sub_goal,
                "tool_info": tool_info,
                "execution_result": execution_result,
                "original_data": data  # 保留原始数据以备后续使用
            }
        except Exception as e:
            logger.error(f"执行器到验证器适配失败: {e}")
            return {
                "sub_goal": "",
                "tool_info": {"name": "", "description": "", "execution_info": {}},
                "execution_result": "",
                "original_data": data
            }
    
    def adapt_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将验证器输出适配为执行器可理解的格式
        
        Args:
            data: 验证器输出数据
            
        Returns:
            执行器可理解的格式
        """
        try:
            # 验证器输出已经是标准格式，直接返回
            return data
        except Exception as e:
            logger.error(f"验证器输出适配失败: {e}")
            return {
                "success": False,
                "error": f"适配失败: {e}",
                "result": ""
            }


class VerifierToPlannerAdapter(ComponentAdapter):
    """验证器到规划器的适配器"""
    
    def adapt_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将验证器输出适配为规划器输入
        
        Args:
            data: 验证器输出数据
            
        Returns:
            规划器输入数据
        """
        try:
            # 提取验证器输出
            result = data.get("result", {})
            verification_status = result.get("verification_status", {})
            
            # 构建规划器输入
            return {
                "verification_result": result,
                "continue": verification_status.get("continue", True),
                "reason": verification_status.get("reason", ""),
                "next_focus": verification_status.get("next_focus", "")
            }
        except Exception as e:
            logger.error(f"验证器到规划器适配失败: {e}")
            return {
                "verification_result": {},
                "continue": True,
                "reason": f"适配失败: {e}",
                "next_focus": ""
            }
    
    def adapt_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将规划器输出适配为验证器可理解的格式
        
        Args:
            data: 规划器输出数据
            
        Returns:
            验证器可理解的格式
        """
        try:
            # 规划器输出已经是标准格式，直接返回
            return data
        except Exception as e:
            logger.error(f"规划器输出适配失败: {e}")
            return {
                "success": False,
                "error": f"适配失败: {e}",
                "result": ""
            }


class ComponentInteractionManager:
    """组件交互管理器"""
    
    def __init__(self):
        """初始化组件交互管理器"""
        self.adapters = {
            "planner_to_executor": PlannerToExecutorAdapter(),
            "executor_to_verifier": ExecutorToVerifierAdapter(),
            "verifier_to_planner": VerifierToPlannerAdapter()
        }
        self.interaction_history = []
    
    def adapt(self, adapter_key: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用指定适配器转换数据
        
        Args:
            adapter_key: 适配器键名，如 "planner_to_executor"
            data: 要转换的数据
            
        Returns:
            转换后的数据
        """
        if adapter_key not in self.adapters:
            logger.warning(f"未找到适配器: {adapter_key}，使用原始数据")
            return data
        
        try:
            adapter = self.adapters[adapter_key]
            adapted_data = adapter.adapt_input(data)
            
            # 记录交互历史
            source, target = adapter_key.split("_to_")
            self.interaction_history.append({
                "source": source,
                "target": target,
                "adapter": adapter_key,
                "original_data": data,
                "adapted_data": adapted_data,
                "success": True
            })
            
            return adapted_data
        except Exception as e:
            logger.error(f"数据适配失败: {e}")
            
            # 记录失败交互
            source, target = adapter_key.split("_to_")
            self.interaction_history.append({
                "source": source,
                "target": target,
                "adapter": adapter_key,
                "original_data": data,
                "adapted_data": {},
                "success": False,
                "error": str(e)
            })
            
            return data
    
    def adapt_data(self, 
                   source_component: str, 
                   target_component: str, 
                   data: Dict[str, Any]) -> Dict[str, Any]:
        """
        适配组件间的数据传递
        
        Args:
            source_component: 源组件名称
            target_component: 目标组件名称
            data: 要传递的数据
            
        Returns:
            适配后的数据
        """
        adapter_key = f"{source_component}_to_{target_component}"
        return self.adapt(adapter_key, data)
    
    def register_adapter(self, 
                         source_component: str, 
                         target_component: str, 
                         adapter: ComponentAdapter):
        """
        注册自定义适配器
        
        Args:
            source_component: 源组件名称
            target_component: 目标组件名称
            adapter: 适配器实例
        """
        adapter_key = f"{source_component}_to_{target_component}"
        self.adapters[adapter_key] = adapter
        #logger.info(f"注册适配器: {adapter_key}")
    
    def get_interaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取交互历史"""
        return self.interaction_history[-limit:]
    
    def get_adapter_metrics(self) -> Dict[str, Any]:
        """获取适配器性能指标"""
        if not self.interaction_history:
            return {
                "total_interactions": 0,
                "successful_interactions": 0,
                "failed_interactions": 0,
                "success_rate": 0,
                "adapter_usage": {}
            }
        
        total_interactions = len(self.interaction_history)
        successful_interactions = sum(1 for i in self.interaction_history if i.get("success", False))
        failed_interactions = total_interactions - successful_interactions
        success_rate = successful_interactions / total_interactions if total_interactions > 0 else 0
        
        # 统计适配器使用情况
        adapter_usage = {}
        for interaction in self.interaction_history:
            adapter = interaction.get("adapter", "")
            if adapter not in adapter_usage:
                adapter_usage[adapter] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0
                }
            
            adapter_usage[adapter]["total"] += 1
            if interaction.get("success", False):
                adapter_usage[adapter]["successful"] += 1
            else:
                adapter_usage[adapter]["failed"] += 1
        
        return {
            "total_interactions": total_interactions,
            "successful_interactions": successful_interactions,
            "failed_interactions": failed_interactions,
            "success_rate": success_rate,
            "adapter_usage": adapter_usage
        }


# 创建全局交互管理器实例
interaction_manager = ComponentInteractionManager()
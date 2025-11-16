"""
Base tool interface for AgentFlow
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    result: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    @abstractmethod
    def execute(self, query: str, context: Optional[str] = None) -> ToolResult:
        """
        执行工具
        
        Args:
            query: 执行查询
            context: 上下文信息
            
        Returns:
            ToolResult: 执行结果
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取工具元数据"""
        return {
            "name": self.name,
            "description": self.description
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        获取工具参数定义
        
        Returns:
            Dict: 参数定义，包括必需参数和可选参数
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "执行查询或命令"
                }
            },
            "required": ["query"]
        }
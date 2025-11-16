"""
Tools for AgentFlow system
"""

from .base import BaseTool, ToolResult
from .web_search import (
    DuckDuckGoSearchTool,
    GoogleCustomSearchTool
)
from .wikipedia_search import WikipediaSearchTool
from .python_executor import PythonExecutorTool
from .base_generator import BaseGenerator

# 尝试导入DashScope工具（如果文件存在）
from .dashscope_api import DashScopeAPITool, DashScopeSearchTool
_HAS_DASHSCOPE = True

__all__ = [
    "BaseTool",
    "ToolResult", 
    "DuckDuckGoSearchTool",
    "GoogleCustomSearchTool",
    "WikipediaSearchTool",
    "PythonExecutorTool",
    "BaseGenerator",
    "DashScopeAPITool",
    "DashScopeSearchTool"
]
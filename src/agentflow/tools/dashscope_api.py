"""
DashScope API工具实现
通义千问API调用工具
"""

import requests
import json
import os
from typing import Dict, Any, Optional
from .base import BaseTool, ToolResult


class DashScopeAPITool(BaseTool):
    """DashScope API工具 - 调用通义千问模型"""
    
    def __init__(self, api_key: str, model: str = "qwen-max", timeout: int = 30):
        super().__init__(
            name="dashscope_api",
            description="Call DashScope API to use Qwen models (requires API key)"
        )
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def execute(self, query: str, context: Optional[str] = None) -> ToolResult:
        """执行DashScope API调用"""
        # 构建API URL
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
        # 构建请求负载
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            },
            "parameters": {
                "temperature": 0.8,
                "max_tokens": 1500,
                "result_format": "message"
            }
        }
        
        # 发送请求
        response = self.session.post(url, json=payload, timeout=self.timeout)
        
        # 检查响应状态
        if response.status_code != 200:
            return ToolResult(
                success=False,
                result="",
                error=f"DashScope API request failed with status code {response.status_code}: {response.text}"
            )
        
        # 解析响应
        data = response.json()
        
        # 检查是否有错误
        if "code" in data and data["code"] != "200":
            return ToolResult(
                success=False,
                result="",
                error=f"DashScope API error: {data.get('message', 'Unknown error')}"
            )
        
        # 提取生成结果
        if "output" in data and "choices" in data["output"]:
            choices = data["output"]["choices"]
            if choices:
                generated_text = choices[0]["message"]["content"]
                
                return ToolResult(
                    success=True,
                    result=generated_text,
                    metadata={
                        "query": query,
                        "model": self.model,
                        "tool": "dashscope_api",
                        "usage": data.get("usage", {})
                    }
                )
        
        return ToolResult(
            success=False,
            result="",
            error="No valid response from DashScope API"
        )


class MoonshotAPITool(BaseTool):
    """Moonshot API工具 - 调用Moonshot-Kimi-K2-Instruct模型"""
    
    def __init__(self, api_key: str, model: str = "moonshot-v1-8k", timeout: int = 30):
        super().__init__(
            name="moonshot_api",
            description="Call Moonshot API to use Kimi models (requires API key)"
        )
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def execute(self, query: str, context: Optional[str] = None) -> ToolResult:
        """执行Moonshot API调用"""
        # 构建API URL
        url = "https://api.moonshot.cn/v1/chat/completions"
        
        # 构建请求负载
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        
        # 发送请求
        response = self.session.post(url, json=payload, timeout=self.timeout)
        
        # 检查响应状态
        if response.status_code != 200:
            return ToolResult(
                success=False,
                result="",
                error=f"Moonshot API request failed with status code {response.status_code}: {response.text}"
            )
        
        # 解析响应
        data = response.json()
        
        # 检查是否有错误
        if "error" in data:
            return ToolResult(
                success=False,
                result="",
                error=f"Moonshot API error: {data['error'].get('message', 'Unknown error')}"
            )
        
        # 提取生成结果
        if "choices" in data and data["choices"]:
            generated_text = data["choices"][0]["message"]["content"]
            
            return ToolResult(
                success=True,
                result=generated_text,
                metadata={
                    "query": query,
                    "model": self.model,
                    "tool": "moonshot_api",
                    "usage": data.get("usage", {})
                }
            )
        
        return ToolResult(
            success=False,
            result="",
            error="No valid response from Moonshot API"
        )


class DashScopeSearchTool(BaseTool):
    """DashScope搜索工具 - 使用通义千问进行搜索"""
    
    def __init__(self, api_key: str, model: str = "qwen-max", timeout: int = 30):
        super().__init__(
            name="dashscope_search",
            description="Search using DashScope API with Qwen models"
        )
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def execute(self, query: str, context: Optional[str] = None) -> ToolResult:
        """执行DashScope搜索"""
        # 构建搜索提示
        search_prompt = f"Please search for information about '{query}' and provide a comprehensive answer with sources if possible."
        
        # 构建API URL
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
        # 构建请求负载
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": search_prompt
                    }
                ]
            },
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 2000,
                "result_format": "message"
            }
        }
        
        # 发送请求
        response = self.session.post(url, json=payload, timeout=self.timeout)
        
        # 检查响应状态
        if response.status_code != 200:
            return ToolResult(
                success=False,
                result="",
                error=f"DashScope API request failed with status code {response.status_code}: {response.text}"
            )
        
        # 解析响应
        data = response.json()
        
        # 检查是否有错误
        if "code" in data and data["code"] != "200":
            return ToolResult(
                success=False,
                result="",
                error=f"DashScope API error: {data.get('message', 'Unknown error')}"
            )
        
        # 提取生成结果
        if "output" in data and "choices" in data["output"]:
            choices = data["output"]["choices"]
            if choices:
                generated_text = choices[0]["message"]["content"]
                
                return ToolResult(
                    success=True,
                    result=f"DashScope search results for '{query}':\n\n{generated_text}",
                    metadata={
                        "query": query,
                        "model": self.model,
                        "tool": "dashscope_search",
                        "usage": data.get("usage", {})
                    }
                )
        
        return ToolResult(
            success=False,
            result="",
            error="No valid response from DashScope API"
        )
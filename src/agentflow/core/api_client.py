"""
API Client Module for AgentFlow
API客户端：通过API调用外部模型服务
"""

import json
import time
import requests
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class APIClient:
    """
    API客户端，用于调用外部模型服务
    
    支持多种API格式：
    1. OpenAI兼容格式
    2. DashScope格式
    3. 自定义格式
    """
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.openai.com/v1",
                 model_name: str = "gpt-3.5-turbo",
                 timeout: int = 60,
                 api_format: str = "openai"):
        """
        初始化API客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称
            timeout: 请求超时时间
            api_format: API格式 ("openai", "dashscope", "custom")
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.api_format = api_format
        
        # 设置请求头
        self.headers = self._get_headers()
        
        #logger.info(f"✅ 初始化API客户端: {self.base_url} ({self.model_name})")
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        if self.api_format == "openai":
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        elif self.api_format == "dashscope":
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        else:  # custom
            return {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key
            }
    
    def _format_messages(self, prompt: str) -> List[Dict[str, str]]:
        """格式化消息"""
        return [{"role": "user", "content": prompt}]
    
    def _make_request(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024) -> Dict[str, Any]:
        """发送API请求"""
        # 构建请求体
        if self.api_format == "openai":
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            endpoint = f"{self.base_url}/chat/completions"
        elif self.api_format == "dashscope":
            data = {
                "model": self.model_name,
                "input": {
                    "messages": messages
                },
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            }
            endpoint = f"{self.base_url}/services/aigc/text-generation/generation"
        else:  # custom
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            endpoint = f"{self.base_url}/generate"
        
        # 发送请求
        response = requests.post(
            endpoint,
            headers=self.headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def _parse_response(self, response: Dict[str, Any]) -> str:
        """解析API响应"""
        if self.api_format == "openai":
            return response["choices"][0]["message"]["content"]
        elif self.api_format == "dashscope":
            return response["output"]["text"]
        else:  # custom
            return response.get("text", "")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示词
            temperature: 采样温度
            max_tokens: 最大生成令牌数
            
        Returns:
            str: 生成的文本
        """
        # 格式化消息
        messages = self._format_messages(prompt)
        
        # 发送请求
        start_time = time.time()
        response = self._make_request(messages, temperature, max_tokens)
        generated_text = self._parse_response(response)
        
        execution_time = time.time() - start_time
        #logger.info(f"✅ API调用成功，耗时: {execution_time:.2f}秒")
        
        return generated_text
    
    def generate_with_history(self, 
                             messages: List[Dict[str, str]], 
                             temperature: float = 0.7, 
                             max_tokens: int = 1024) -> str:
        """
        基于历史消息生成文本
        
        Args:
            messages: 消息历史
            temperature: 采样温度
            max_tokens: 最大生成令牌数
            
        Returns:
            str: 生成的文本
        """
        # 发送请求
        start_time = time.time()
        response = self._make_request(messages, temperature, max_tokens)
        generated_text = self._parse_response(response)
        
        execution_time = time.time() - start_time
        #logger.info(f"✅ API调用成功，耗时: {execution_time:.2f}秒")
        
        return generated_text
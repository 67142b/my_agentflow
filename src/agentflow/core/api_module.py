"""
API Module for AgentFlow
通过API调用实现executor、verifier和generator角色
"""

import requests
import json
import os
from typing import Dict, Any, Optional
from ..tools.base import ToolResult


class APIModule:
    """
    API模块 - 通过调用DashScope API实现executor、verifier和generator角色
    """
    
    def __init__(self, api_key: str, model: str = "qwen-max", timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        # 角色特定的prompt模板
        self.role_prompts = {
            "executor": self._get_executor_prompt(), 
            "verifier": self._get_verifier_prompt(),
            "generator": self._get_generator_prompt()
        }
    
    def _get_executor_prompt(self):
        """Executor角色prompt"""
        return """You are a TOOL EXECUTOR. Your task is to execute the selected tool with the given context.

[Current Sub-Goal]
{sub_goal}

[Selected Tool]
{selected_tool}

[Tool Context]
{tool_context}

Execute the tool and provide the result:"""
    
    def _get_verifier_prompt(self):
        """Verifier角色prompt"""
        return """You are a VERIFIER in an agentic reasoning system. Your task is to analyze the execution result and decide whether to continue.

[Original Query]
{query}

[Current Sub-Goal]
{sub_goal}

[Tool Used]
{selected_tool}

[Execution Result]
{execution_result}

{memory_context}

Output:
[Execution Analysis]
{{Analyze the execution result}}

[Memory Analysis]  
{{Assess if current information is sufficient}}

[Verification Status]
{{0: continue, 1: terminate}}

Provide your analysis:"""
    
    def _get_generator_prompt(self):
        """Generator角色prompt"""
        return """You are a SOLUTION GENERATOR. Your task is to synthesize all information and provide the final answer.

[Original Query]
{query}

[Complete Reasoning History]
{memory_context}

Based on all the information above, provide a clear final answer in <answer> tags.

<answer>
{{Your final answer}}
</answer>"""
    
    def _call_api(self, prompt: str) -> Dict[str, Any]:
        """调用DashScope API"""
        try:
            # 构建API URL
            url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
            
            # 构建请求负载
            payload = {
                "model": self.model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1500,
                    "result_format": "message"
                }
            }
            
            # 发送请求
            response = self.session.post(url, json=payload, timeout=self.timeout)
            
            # 检查响应状态
            if response.status_code != 200:
                return {
                    "success": False,
                    "result": "",
                    "error": f"DashScope API request failed with status code {response.status_code}: {response.text}"
                }
            
            # 解析响应
            data = response.json()
            
            # 检查是否有错误
            if "code" in data and data["code"] != "200":
                return {
                    "success": False,
                    "result": "",
                    "error": f"DashScope API error: {data.get('message', 'Unknown error')}"
                }
            
            # 提取生成结果
            if "output" in data and "choices" in data["output"]:
                choices = data["output"]["choices"]
                if choices:
                    generated_text = choices[0]["message"]["content"]
                    
                    return {
                        "success": True,
                        "result": generated_text,
                        "metadata": {
                            "model": self.model,
                            "usage": data.get("usage", {})
                        }
                    }
            
            return {
                "success": False,
                "result": "",
                "error": "No valid response from DashScope API"
            }
            
        except Exception as e:
            return {
                "success": False,
                "result": "",
                "error": f"DashScope API call failed: {str(e)}"
            }
    
    def forward(self, 
                role_type: str,
                inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        统一的前向传播接口
        
        Args:
            role_type: 角色类型 ("executor", "verifier", "generator")
            inputs: 输入参数字典
            
        Returns:
            Dict: 包含生成结果的字典
        """
        # 检查角色类型是否有效
        if role_type not in self.role_prompts:
            return {
                "success": False,
                "result": "",
                "error": f"Invalid role type: {role_type}"
            }
        
        # 构建角色特定的prompt
        prompt = self.role_prompts[role_type].format(**inputs)
        
        # 调用API
        result = self._call_api(prompt)
        
        return {
            "generated_text": result.get("result", ""),
            "prompt": prompt,
            "role": role_type,
            "success": result.get("success", False),
            "error": result.get("error", "")
        }
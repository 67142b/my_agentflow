"""
Evaluator Module for AgentFlow
评估器：评估工具执行结果，决定是否继续或终止
"""

import json
import re
from typing import Dict, Any, Optional, Tuple
from ..tools.dashscope_api import DashScopeAPITool
from .prompt_templates import PromptTemplates


class Evaluator:
    """
    评估器
    
    职责：
    1. 评估工具执行结果
    2. 决定是否继续或终止
    3. 验证上下文完整性
    4. 提供终止信号
    
    输入：
    - Current Sub-Goal
    - Tool Execution Result
    - Memory Context
    
    输出：
    - Verification Status (0: continue, 1: terminate)
    - Reasoning
    - Error Signal (if any)
    """
    
    def __init__(self, api_key: str, model: str = "qwen-max"):
        self.api_tool = DashScopeAPITool(api_key=api_key, model=model)
        
    def evaluate(self, 
                 sub_goal: str,
                 tool_name: str,
                 tool_result: str,
                 memory_context: str,
                 query: str) -> Dict[str, Any]:
        """
        评估工具执行结果
        
        Args:
            sub_goal: 当前子目标
            tool_name: 使用的工具名称
            tool_result: 工具执行结果
            memory_context: 记忆上下文
            query: 原始查询
            
        Returns:
            Dict: 包含评估结果的字典
        """
        # 使用统一的提示词模板
        prompt = PromptTemplates.get_evaluator_prompt(
            "result_evaluation",
            query=query,
            sub_goal=sub_goal,
            tool_name=tool_name,
            tool_result=tool_result,
            memory_context=memory_context
        )
        
        # 调用API
        result = self.api_tool.execute(prompt)
        
        if not result.success:
            # 如果API调用失败，默认继续
            return {
                "verification_status": 0,
                "reasoning": f"API call failed: {result.error}. Defaulting to continue.",
                "error_signal": result.error
            }
        
        # 解析响应
        verification_status, reasoning, error_signal, confidence = self._parse_evaluation(result.result)
        
        return {
            "verification_status": verification_status,
            "reasoning": reasoning,
            "error_signal": error_signal,
            "confidence": confidence,
            "raw_output": result.result
        }
    
    def _parse_evaluation(self, response: str) -> Tuple[int, str, str, float]:
        """解析评估响应"""
        try:
            # 提取验证状态
            status_match = re.search(r'\[Verification Status\]\s*(.*?)(?=\n\[|$)', response, re.DOTALL)
            status_text = status_match.group(1).strip() if status_match else "0"
            
            # 转换为整数
            try:
                verification_status = int(status_text)
                if verification_status not in [0, 1]:
                    verification_status = 0
            except ValueError:
                verification_status = 0
            
            # 提取推理
            reasoning_match = re.search(r'\[Reasoning\]\s*(.*?)(?=\n\[|$)', response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            # 提取错误信号
            error_match = re.search(r'\[Error Signal\]\s*(.*?)(?=\n\[|$)', response, re.DOTALL)
            error_signal = error_match.group(1).strip() if error_match else "None"
            
            # 提取置信度
            confidence_match = re.search(r'\[Confidence\]\s*(.*?)(?=\n\[|$)', response, re.DOTALL)
            confidence_text = confidence_match.group(1).strip() if confidence_match else "0.5"
            
            # 转换置信度为浮点数
            try:
                confidence = float(confidence_text)
                if confidence < 0 or confidence > 1:
                    confidence = 0.5  # 默认中等置信度
            except ValueError:
                confidence = 0.5
            
            # 如果错误信号是"None"或"none"，设为None
            if error_signal.lower() == "none":
                error_signal = None
            
            return verification_status, reasoning, error_signal, confidence
            
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            return 0, f"Parsing error: {str(e)}", None, 0.5
    
    def verify_context(self, 
                      context: str, 
                      query: str) -> Dict[str, Any]:
        """
        验证上下文完整性
        
        Args:
            context: 当前上下文
            query: 原始查询
            
        Returns:
            Dict: 包含验证结果的字典
        """
        prompt = f"""You are an evaluator in an agentic reasoning system. Your task is to verify if the current context is complete and consistent for answering the query.

[Query]
{query}

[Current Context]
{context}

Your task is to evaluate the context and determine if it's sufficient and consistent. Consider:
1. Does the context contain all necessary information to answer the query?
2. Is the information consistent and coherent?
3. Are there any contradictions or gaps?
4. Is additional information needed?

Provide your evaluation in the following format:

[Completeness]
{{Complete/Partial/Incomplete}}

[Consistency]
{{Consistent/Mostly Consistent/Inconsistent}}

[Additional Information Needed]
{{Yes/No}}

[Reasoning]
{{Explain your evaluation}}

Now, provide your evaluation:"""
        
        # 调用API
        result = self.api_tool.execute(prompt)
        
        if not result.success:
            return {
                "completeness": "Incomplete",
                "consistency": "Inconsistent",
                "additional_info_needed": True,
                "reasoning": f"API call failed: {result.error}",
                "raw_output": ""
            }
        
        # 解析响应
        completeness, consistency, additional_info, reasoning = self._parse_context_verification(result.result)
        
        return {
            "completeness": completeness,
            "consistency": consistency,
            "additional_info_needed": additional_info,
            "reasoning": reasoning,
            "raw_output": result.result
        }
    
    def _parse_context_verification(self, response: str) -> Tuple[str, str, bool, str]:
        """解析上下文验证响应"""
        try:
            # 提取完整性
            completeness_match = re.search(r'\[Completeness\]\s*(.*?)(?=\n\[|$)', response, re.DOTALL)
            completeness = completeness_match.group(1).strip() if completeness_match else "Incomplete"
            
            # 提取一致性
            consistency_match = re.search(r'\[Consistency\]\s*(.*?)(?=\n\[|$)', response, re.DOTALL)
            consistency = consistency_match.group(1).strip() if consistency_match else "Inconsistent"
            
            # 提取是否需要额外信息
            additional_match = re.search(r'\[Additional Information Needed\]\s*(.*?)(?=\n\[|$)', response, re.DOTALL)
            additional_text = additional_match.group(1).strip() if additional_match else "Yes"
            additional_info = additional_text.lower() == "yes"
            
            # 提取推理
            reasoning_match = re.search(r'\[Reasoning\]\s*(.*?)(?=\n\[|$)', response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            return completeness, consistency, additional_info, reasoning
            
        except Exception as e:
            print(f"Error parsing context verification: {e}")
            return "Incomplete", "Inconsistent", True, f"Parsing error: {str(e)}"
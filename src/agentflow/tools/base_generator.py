"""
Base Generator Tool for AgentFlow
基础生成器工具：提供通用的文本生成和推理能力
"""

import json
import re
import logging
import torch
from typing import Dict, Any, Optional, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from .dashscope_api import DashScopeAPITool
from ..core.planner import Planner
from ..core.memory import Memory, MemoryEntryType
from .base import BaseTool, ToolResult
import torch.nn.functional as F

# 导入LLM引擎
from src.LLM_engine import get_llm_engine

logger = logging.getLogger(__name__)


class BaseGenerator(BaseTool):
    """
    基础生成器工具
    
    职责：
    1. 提供默认推理引擎
    2. 处理一般性问题解决
    3. 支持持续推理
    4. 与Planner协同工作
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 device: str = "cuda",
                 torch_dtype: str = "float16",
                 max_length: int = 1024*16,
                 config_path: str = "src/configs/config.yaml",
                 share_model: bool = False,
                 llm_engine=None):
        """
        初始化基础生成器工具
        
        Args:
            model_path: 模型路径（如果提供model和tokenizer则不需要）
            model: 已加载的模型实例
            tokenizer: 已加载的分词器实例
            device: 设备
            torch_dtype: 数据类型
            max_length: 最大生成长度
            config_path: 配置文件路径
            share_model: 是否使用共享模型
            llm_engine: 外部传入的LLM引擎实例（可选）
        """
        super().__init__(
            name="base_generator",
            description="基础生成器工具，使用本地模型进行文本生成"
        )
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.max_length = max_length
        self.config_path = config_path
        self.share_model = share_model
        
        # 使用传入的llm_engine实例或初始化新的实例
        if llm_engine is not None:
            self.llm_engine = llm_engine
        else:
            self.llm_engine = get_llm_engine(config_path)
        
        # 使用LLM引擎，无需直接加载模型
        self.model_path = model_path
    
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
                    "description": "需要生成或推理的文本"
                },
                "context": {
                    "type": "string",
                    "description": "生成或推理的上下文信息"
                }
            },
            "required": ["query"]
        }
    
    def execute(self, 
               input_text: str, 
               context: Optional[str] = None,
               planner: Optional[Any] = None,
               memory: Optional[Any] = None,
               return_token_probs: bool = False) -> ToolResult:
        """
        执行基础生成器工具
        
        Args:
            input_text: 输入文本
            context: 上下文信息
            planner: 规划器实例（可选）
            memory: 记忆实例（可选）
            return_token_probs: 是否返回token概率信息
            
        Returns:
            ToolResult: 工具执行结果
        """
        #logger.info(f"=======================================base_generator.execute开始=======================================")
        #logger.info(f"输入文本: {input_text}")
        #logger.info(f"上下文: {context}")
        #logger.info(f"规划器: {planner}")
        #logger.info(f"记忆: {memory}")
        
        # 如果有规划器和记忆，则进行持续推理
        if planner is not None and memory is not None:
            #logger.info("使用持续推理模式")
            return self._continuous_reasoning(input_text, context, planner, memory, return_token_probs)
        else:
            # 否则进行单次推理
            #logger.info("===================================================进行单次推理===================================================")
            result = self._single_reasoning(input_text, context, return_token_probs)
            #logger.info(f"单次推理结果: {result}")
            return result
    
    def _single_reasoning(self, 
                         input_text: str, 
                         context: Optional[str] = None,
                         return_token_probs: bool = False) -> ToolResult:
        """
        单次推理
        
        Args:
            input_text: 输入文本
            context: 上下文信息
            return_token_probs: 是否返回token概率信息
            
        Returns:
            ToolResult: 工具执行结果
        """
        # 使用统一的提示词模板（不需要锁保护）
        prompt = self._build_single_reasoning_prompt(input_text, context)
        #logger.info(" =================================单次推理提示词=====================================")
        #logger.info(f"{prompt}")
        #logger.info("=======================================单次推理提示词====================================")
        
        # 使用LLM引擎生成输出，共享planner的权重和分词器
        response = self.llm_engine.generate(
            model_source="local",
            model_name="base_generator",  # 使用base_generator作为模型类型，会自动共享planner的权重
            prompt=prompt,
            max_new_tokens=1024,  # 使用max_new_tokens而不是max_tokens
            temperature=0.5,
            return_token_probs=return_token_probs
        )
        
        # 根据返回类型处理结果
        if return_token_probs and isinstance(response, dict):
            generated_text = response.get("response", "")
            token_probs = response.get("token_probs", [])
            token_logprobs = response.get("token_logprobs", [])
            token_texts = response.get("token_texts", [])
        else:
            generated_text = response
            token_probs = []
            token_logprobs = []
            token_texts = []
        
        # 检查生成的文本是否为空
        if not generated_text.strip():
            logger.warning("警告：模型生成了空文本，尝试使用更简单的生成参数")
            # 获取模型和分词器
            model, tokenizer = self.llm_engine.get_local_model("base_generator")
            
            # 编码输入
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=8192
            ).to("cuda")
            
            # 尝试使用更简单的参数重新生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 解码重新生成的文本
            generated_text = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            #logger.info("第二次生成的文本:\n===============================================")
            #logger.info(f"{generated_text}")
            #logger.info("===============================================")
            
            # 保存input_length，避免在删除inputs后无法访问
            input_length = inputs.input_ids.shape[1]
            
            # 清理显存 - 立即释放不再需要的张量
            del inputs, outputs
            torch.cuda.empty_cache()
        
        # 计算输出长度
        
        return ToolResult(
            success=True,
            result=generated_text,
            metadata={
                "reasoning_type": "single",
                "token_probs": token_probs,
                "token_logprobs": token_logprobs,
                "token_texts": token_texts
            }
        )
    
    def _continuous_reasoning(self, 
                              input_text: str, 
                              context: Optional[str],
                              planner: Any,
                              memory: Any,
                              return_token_probs: bool = False) -> ToolResult:
        """
        持续推理
        
        Args:
            input_text: 输入文本
            context: 上下文信息
            planner: 规划器实例
            memory: 记忆实例
            return_token_probs: 是否返回token概率信息
            
        Returns:
            ToolResult: 工具执行结果
        """
        # 获取记忆上下文
        memory_context = memory.get_formatted_context()
        
        # 获取所需技能
        required_skills = memory.get_required_skills()
        
        # 调用规划器制定计划
        plan_result = planner.plan(
            query=input_text,
            memory_context=memory_context,
            required_skills=required_skills
        )
        
        # 记录规划结果
        memory.add_action(
            sub_goal=plan_result["sub_goal"],
            selected_tool=plan_result["selected_tool"],
            tool_context=plan_result["tool_context"],
            execution_result="Planning completed",
            token_probs=plan_result.get("token_probs", []),
            token_logprobs=plan_result.get("token_logprobs", []),
            token_texts=plan_result.get("token_texts", [])
        )
        
        # 构建推理提示词
        prompt = self._build_continuous_reasoning_prompt(
            input_text, context, plan_result, memory_context
        )
        
        # 使用LLM引擎生成输出，共享planner的权重和分词器
        try:
            response = self.llm_engine.generate(
                model_source="local",
                model_name="base_generator",  # 使用base_generator作为模型类型，会自动共享planner的权重
                prompt=prompt,
                max_new_tokens=1024,  # 使用max_new_tokens而不是max_tokens
                temperature=0.3,
                return_token_probs=return_token_probs
            )
            
            # 根据返回类型处理结果
            if return_token_probs and isinstance(response, dict):
                generated_text = response.get("response", "")
                token_probs_list = response.get("token_probs", [])
                token_logprobs_list = response.get("token_logprobs", [])
                token_texts_list = response.get("token_texts", [])
            else:
                generated_text = response
                token_probs_list = []
                token_logprobs_list = []
                token_texts_list = []
        except Exception as e:
            logger.error(f"生成失败: {str(e)}")
            generated_text = ""
            token_probs_list = []
            token_logprobs_list = []
            token_texts_list = []
        
        # 解析结果
        result = self._parse_continuous_reasoning_result(generated_text)
        
        # 记录推理结果
        memory.add_action(
            sub_goal=plan_result["sub_goal"],
            selected_tool="base_generator",
            tool_context=plan_result["tool_context"],
            execution_result=result,
            token_probs=token_probs_list,
            token_logprobs=token_logprobs_list,
            token_texts=token_texts_list
        )
        
        # 判断是否需要继续推理
        conclusion = planner.extract_conclusion(generated_text)
        if conclusion == "STOP":
            memory.set_terminated()
        
        return ToolResult(
            success=True,
            result=result,
            metadata={
                "reasoning_type": "continuous",
                "sub_goal": plan_result["sub_goal"],
                "selected_tool": plan_result["selected_tool"],
                "conclusion": conclusion,
                "token_probs": token_probs_list,
                "token_logprobs": token_logprobs_list,
                "token_texts": token_texts_list
            }
        )
    
    def _build_single_reasoning_prompt(self, 
                                     input_text: str, 
                                     context: Optional[str] = None) -> str:
        """
        构建单次推理提示词
        
        Args:
            input_text: 输入文本
            context: 上下文信息
            
        Returns:
            str: 构建的提示词
        """
        # 构建明确的英文提示词
        prompt = f"""You are in a multi-agent collaborative system, and your identity is a professional Q&A assistant. Please answer the question based on the following information, and respond in the same language as the question.

Question: {input_text}"""
        
        if context:
            prompt += f"\n\nContext from memory system: {context}"
            
        prompt += "\n\nRequirements:\n1. Respond only in English\n2. Answer should be concise and accurate\n3. Do not include any formatting marks or instruction descriptions\n4. Provide the answer content directly\n\nAnswer:"
        
        return prompt
    
    def _build_continuous_reasoning_prompt(self, 
                                         input_text: str, 
                                         context: Optional[str],
                                         plan_result: Dict[str, str],
                                         memory_context: str) -> str:
        """
        构建持续推理提示词
        
        Args:
            input_text: 输入文本
            context: 上下文信息
            plan_result: 规划结果
            memory_context: 记忆上下文
            
        Returns:
            str: 构建的提示词
        """
        # 构建简洁、直接的英文提示词
        prompt = f"""You are a reasoning engine in an agent system. Please analyze and reason based on the following information.

Original Question: {input_text}"""
        
        if memory_context:
            prompt += f"\n\nMemory Context: {memory_context}"
            
        if plan_result.get("sub_goal"):
            prompt += f"\n\nSub-goal: {plan_result['sub_goal']}"
            
        if plan_result.get("selected_tool"):
            prompt += f"\n\nSelected Tool: {plan_result['selected_tool']}"
            
        if plan_result.get("tool_context"):
            prompt += f"\n\nTool Context: {plan_result['tool_context']}"
            
        if context:
            prompt += f"\n\nAdditional Context: {context}"
            
        prompt += "\n\nPlease provide concise and logical analysis and reasoning in English:"
        
        return prompt
    
    def _remove_repetitions(self, text: str) -> str:
        """
        去除文本中的重复内容
        
        Args:
            text: 原始文本
            
        Returns:
            str: 去重后的文本
        """
        if not text:
            return text
            
        # 按行分割并去重
        lines = text.strip().split('\n')
        unique_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                unique_lines.append(line)
                seen_lines.add(line)
        
        # 重新组合
        result = '\n'.join(unique_lines)
        
        # 检查句子级别的重复
        sentences = result.split('。')
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen_sentences:
                unique_sentences.append(sentence)
                seen_sentences.add(sentence)
        
        return '。'.join(unique_sentences)
    
    def _parse_single_reasoning_result(self, result: str) -> str:
        """
        解析单次推理结果
        
        Args:
            result: 原始结果
            
        Returns:
            str: 解析后的结果
        """
        # 去除可能的重复内容
        lines = result.strip().split('\n')
        unique_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                unique_lines.append(line)
                seen_lines.add(line)
        
        # 重新组合并限制长度
        parsed_result = '\n'.join(unique_lines)
        
        # 如果结果过长，进行截断
        if len(parsed_result) > 400:  # 从800减少到400
            parsed_result = parsed_result[:400] + "..."
            
        return parsed_result
    
    def _parse_continuous_reasoning_result(self, result: str) -> str:
        """
        解析持续推理结果
        
        Args:
            result: 原始结果
            
        Returns:
            str: 解析后的结果
        """
        # 尝试提取推理部分
        reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=\nConclusion:|\nConfidence:|$)', result, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # 如果没有找到特定格式，直接使用整个结果
            reasoning = result.strip()
        
        # 去除重复内容
        lines = reasoning.split('\n')
        unique_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                unique_lines.append(line)
                seen_lines.add(line)
        
        # 重新组合并限制长度
        parsed_result = '\n'.join(unique_lines)
        
        # 如果结果过长，进行截断
        if len(parsed_result) > 400:  # 从800减少到400
            parsed_result = parsed_result[:400] + "..."
            
        return parsed_result
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取工具元数据"""
        return {
            "name": "base_generator",
            "description": "Default reasoning engine for general problem solving",
            "parameters": ["input_text", "context"],
            "model_path": getattr(self, 'model_path', 'Unknown'),
            "device": self.device,
            "max_length": self.max_length
        }
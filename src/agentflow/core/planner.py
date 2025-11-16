"""
Planner Module for AgentFlow
动作规划器：制定子目标，选择工具，提供上下文
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import yaml
import re
import os
from PIL import Image
from .prompt_templates import PromptTemplates

# 导入LLM引擎
from src.LLM_engine import get_llm_engine


class Planner:
    def __init__(self, device: str = "cuda", config_path: str = "src/configs/config.yaml", llm_engine=None):
        """
        初始化Planner
        
        Args:
            device: 设备
            config_path: 配置文件路径
            llm_engine: 外部传入的LLM引擎实例（可选）
            enable_concurrency_control: 是否启用并发控制（锁机制）
        """
        self.device = device
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # 使用传入的llm_engine实例或初始化新的实例
        if llm_engine is not None:
            self.llm_engine = llm_engine
        else:
            self.llm_engine = get_llm_engine(config_path)
        
        # 从LLM引擎获取模型
        self.model, self.tokenizer = self.llm_engine.get_local_model(model_type="planner")
        
        # 直接使用模型和tokenizer
        
        # 可用工具列表
        self.available_tools = [
            "base_generator",
            "python_executor", 
            "web_search",
            "wikipedia_search"
        ]
        
        # 工具元数据
        self.tool_metadata = {
            "base_generator": {
                "description": "Generate text responses for general queries, explanations, and content creation, The result may not be good, if it fails please try another tool.",
                "capabilities": ["text_generation", "reasoning", "explanation", "analysis"]
            },
            "python_executor": {
                "description": "Generate Python code and execute it for programming tasks, data processing, and computational tasks",
                "capabilities": ["computation", "data_analysis", "algorithm_execution", "mathematical_operations"]
            },
            "web_search": {
                "description": "Search the web for current information, facts, and recent developments ,For most encyclopedia query tasks, For most encyclopedia query tasks, please chosen this tool",
                "capabilities": ["information_retrieval", "fact_checking", "current_events", "web_content"]
            },
            "wikipedia_search": {
                "description": "Search Wikipedia for encyclopedic knowledge and established facts",
                "capabilities": ["encyclopedic_knowledge", "historical_facts", "scientific_concepts", "definitions"]
            }
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """获取图像信息"""
        from PIL import Image
        image = Image.open(image_path)
        return {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode
        }
    
    def generate_base_response(self, prompt: str, max_length: int = 1024, temperature: float = 0.7) -> str:
        """生成基础响应"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # 直接生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询，确定所需技能和工具"""
        prompt = f"""Analyze the following query to determine the required skills and appropriate tools:

Query: {query}

Available Tools:
- base_generator: For general text generation and reasoning
- python_executor: For calculations and code execution
- web_search: For current information and web content
- wikipedia_search: For encyclopedic knowledge

Provide your analysis in this format:
Required Skills: [list of skills]
Suggested Tools: [list of tools]
Reasoning: [brief explanation]"""

        response = self.generate_base_response(prompt, max_length=512, temperature=0.3)
        
        # 解析响应
        skills = []
        tools = []
        reasoning = ""
        
        for line in response.split('\n'):
            if line.startswith('Required Skills:'):
                skills = [s.strip() for s in line.split(':', 1)[1].strip().split(',')]
            elif line.startswith('Suggested Tools:'):
                tools = [t.strip() for t in line.split(':', 1)[1].strip().split(',')]
            elif line.startswith('Reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
        
        return {
            "required_skills": skills,
            "suggested_tools": tools,
            "reasoning": reasoning
        }
    
    def extract_context_subgoal_and_tool(self, response: str, query: str) -> Tuple[str, str, str]:
        """
        从响应中提取上下文、子目标和工具名称
        修复版本：增强容错性和格式兼容性
        """
        # 初始化默认值
        context = ""
        subgoal = ""
        tool = "base_generator"  # 默认工具
        
        # 尝试多种格式解析
        lines = response.strip().split('\n')
        
        # 方法1: 标准格式解析
        for line in lines:
            line = line.strip()
            if line.startswith('Justification:'):
                continue  # 跳过，只用于调试
            elif line.startswith('Context:'):
                context = line[len('Context:'):].strip()
            elif line.startswith('Sub-Goal:'):
                subgoal = line[len('Sub-Goal:'):].strip()
            elif line.startswith('Tool Name:'):
                tool = line[len('Tool Name:'):].strip()
        
        # 方法2: 如果标准格式失败，尝试简化格式
        if not context or not subgoal or tool == "base_generator":
            # 尝试匹配冒号分隔的格式
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if 'context' in key and not context:
                        context = value
                    elif ('sub' in key and 'goal' in key) or ('subgoal' in key):
                        subgoal = value
                    elif 'tool' in key and not tool:
                        tool = value
        
        # 方法3: 如果仍然失败，使用默认值生成
        if not context:
            context = query  # 使用原始查询作为上下文
        if not subgoal:
            subgoal = f"Address the query: {query}"
        if not tool or tool not in self.available_tools:
            # 基于查询内容智能选择工具
            if self._is_math_expression(query):
                tool = "python_executor"
            elif any(keyword in query.lower() for keyword in ["current", "recent", "news", "today"]):
                tool = "web_search"
            elif any(keyword in query.lower() for keyword in ["define", "what is", "explain", "history"]):
                tool = "wikipedia_search"
            else:
                tool = "base_generator"
        
        # 验证工具名称
        if tool not in self.available_tools:
            tool = "base_generator"  # 最终回退
        
        return context, subgoal, tool
    
    def _get_default_values(self, query: str) -> Tuple[str, str, str]:
        """获取默认的上下文、子目标和工具"""
        context = query
        subgoal = f"Address the query: {query}"
        
        # 基于查询内容智能选择工具
        if self._is_math_expression(query):
            tool = "python_executor"
        elif any(keyword in query.lower() for keyword in ["current", "recent", "news", "today"]):
            tool = "web_search"
        elif any(keyword in query.lower() for keyword in ["define", "what is", "explain", "history"]):
            tool = "wikipedia_search"
        else:
            tool = "base_generator"
        
        return context, subgoal, tool
    
    def _extract_justification(self, response: str) -> str:
        """提取推理说明"""
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Justification:'):
                return line[len('Justification:'):].strip()
        return ""
    
    def verify_memory_completeness(self, query: str, memory_context: str, image: str = "") -> Dict[str, Any]:
        """
        验证记忆完整性，基于论文源码的实现
        
        Args:
            query: 用户查询
            context: 当前上下文
            image: 图像路径（可选）
            
        Returns:
            包含验证结果的字典
        """
        image_info = ""
        if image:
            # 简单的图像信息处理
            image_info = f"Image: {image}"
        
        # 基于论文源码的提示词
        prompt = f"""You are an action planner in an agentic reasoning system. Your task is to verify if the current memory context is complete and sufficient to answer the user's query.

Query: {query}
{image_info}
Current Memory Context:
{memory_context}

Response Format:

If the memory is complete, accurate, AND verified:
Explanation:
<Provide a detailed explanation of why the memory is sufficient. Reference specific information from the memory and explain its relevance to each aspect of the task. Address how each main point of the query has been satisfied.>

Conclusion: STOP

If the memory is incomplete, insufficient, or requires further verification:
Explanation:
<Explain in detail why the memory is incomplete. Identify specific information gaps or unaddressed aspects of the query. Suggest which additional tools could be used, how they might contribute, and why their input is necessary for a comprehensive response.>

Conclusion: CONTINUE

IMPORTANT: Your response MUST end with either 'Conclusion: STOP' or 'Conclusion: CONTINUE' and nothing else. Ensure your explanation thoroughly justifies this conclusion."""

        # 使用LLM引擎生成输出
        generated_text = self.llm_engine.generate(
            model_source="local",
            model_name="planner",
            prompt=prompt,
            max_new_tokens=1024,
            temperature=0.3,  # 较低温度以获得更确定性的结果
            do_sample=True
        )
        
        # 解析验证结果
        return self._parse_memory_verification_result(generated_text)
    
    def _parse_memory_verification_result(self, response: str) -> Dict[str, Any]:
        """解析记忆验证结果，基于论文源码的解析逻辑"""
        # 提取解释部分
        explanation_match = re.search(r'Explanation:\s*(.*?)(?=Conclusion:|$)', response, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        # 提取结论
        conclusion_match = re.search(r'Conclusion:\s*(STOP|CONTINUE)', response, re.IGNORECASE)
        conclusion = conclusion_match.group(1).upper() if conclusion_match else "CONTINUE"
        
        # 验证结论值
        if conclusion not in ["STOP", "CONTINUE"]:
            conclusion = "CONTINUE"  # 默认继续
            
        return {
            "explanation": explanation,
            "conclusion": conclusion,
            "raw_output": response
        }
        """验证上下文完整性"""
        prompt = f"""You are an action planner in an agentic reasoning system. Verify if the current context is complete and sufficient to answer the query.

Query: {query}

Current Context: {context}

Evaluate the context based on:
1. Completeness: Does it contain all necessary information?
2. Consistency: Is the information coherent and non-contradictory?
3. Relevance: Is it directly related to answering the query?

Provide your evaluation in this format:

Completeness: [Complete/Partial/Incomplete]
Consistency: [Consistent/Mostly Consistent/Inconsistent]
Relevance: [Highly Relevant/Somewhat Relevant/Not Relevant]
Verification Needed: [Yes/No]

If verification is needed, explain what additional information is required.

Now, provide your evaluation:"""

        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # 生成输出
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析验证结果
        return self._parse_verification_result(generated_text)
    
    def _parse_verification_result(self, response: str) -> Dict[str, Any]:
        """解析验证结果"""
        # 提取完整性
        completeness_match = re.search(r'Completeness:\s*(.*?)(?=\n|$)', response)
        completeness = completeness_match.group(1).strip() if completeness_match else "Incomplete"
        
        # 提取一致性
        consistency_match = re.search(r'Consistency:\s*(.*?)(?=\n|$)', response)
        consistency = consistency_match.group(1).strip() if consistency_match else "Inconsistent"
        
        # 提取相关性
        relevance_match = re.search(r'Relevance:\s*(.*?)(?=\n|$)', response)
        relevance = relevance_match.group(1).strip() if relevance_match else "Not Relevant"
        
        # 提取是否需要验证
        verification_match = re.search(r'Verification Needed:\s*(.*?)(?=\n|$)', response)
        verification_text = verification_match.group(1).strip() if verification_match else "Yes"
        verification_needed = verification_text.lower() == "yes"
        
        return {
            "completeness": completeness,
            "consistency": consistency,
            "relevance": relevance,
            "verification_needed": verification_needed,
            "raw_output": response
        }
    
    def extract_conclusion(self, response: str) -> str:
        """提取结论（STOP/CONTINUE信号）"""
        # 查找STOP或CONTINUE信号
        if "STOP" in response.upper():
            return "STOP"
        elif "CONTINUE" in response.upper():
            return "CONTINUE"
        else:
            # 如果没有明确信号，根据内容判断
            if any(word in response.lower() for word in ["complete", "finished", "done", "answered"]):
                return "STOP"
            else:
                return "CONTINUE"
    
    def generate_final_output(self, query: str, memory_context: str, image: str = "") -> str:
        """
        生成最终输出，基于论文源码的结构化输出格式
        
        Args:
            query: 用户查询
            memory_context: 记忆上下文
            image: 图像路径（可选）
            
        Returns:
            结构化的最终输出
        """
        image_info = ""
        if image:
            # 简单的图像信息处理
            image_info = f"Image: {image}"
        
        prompt = f"""Task: Generate a concise final answer to the query based on all provided context.

Context:
- **Query:** {query}
{image_info}
- **Memory Context:** {memory_context}

Instructions:
1. Review the query and the results from all actions.
2. Synthesize the key findings into a clear, step-by-step summary of the process.
3. Provide a direct, precise answer to the original query.

Output Structure:
1. **Process Summary:** A clear, step-by-step breakdown of how the query was addressed, including the purpose and key results of each action.
2. **Answer:** A direct and concise final answer to the query."""

        # 使用LLM引擎生成输出
        generated_text = self.llm_engine.generate(
            model_source="local",
            model_name="planner",
            prompt=prompt,
            max_new_tokens=1024,
            temperature=0.5,
            do_sample=True
        )
        
        # 返回文本结果，保持与原接口兼容
        return {
            "text": generated_text.strip(),
            "planner_input_ids": [],  # LLM_engine接口不返回token概率信息
            "planner_output_ids": [],
            "logprobs_old": []
        }
    
    def generate_direct_output(self, query: str) -> str:
        """生成直接输出（无需工具）"""
        prompt = f"""You are a helpful AI assistant. Provide a clear and accurate answer to the following question.

Question: {query}

Please provide a comprehensive answer:"""

        # 使用LLM引擎生成输出
        generated_text = self.llm_engine.generate(
            model_source="local",
            model_name="planner",
            prompt=prompt,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True
        )
        
        # 返回文本结果，保持与原接口兼容
        return {
            "text": generated_text.strip(),
            "planner_input_ids": [],  # LLM_engine接口不返回token概率信息
            "planner_output_ids": [],
            "logprobs_old": []
        }

    def _is_math_expression(self, query: str) -> bool:
        """检测查询是否为数学表达式"""
        # 更精确的数学表达式检测，包含中文数学术语
        math_indicators = [
            # 英文数学术语
            r'calculate', r'compute', r'evaluate', r'solve', r'expression',
            r'equation', r'formula', r'math', r'arithmetic', r'addition',
            r'subtraction', r'multiplication', r'division', r'modulo',
            # 中文数学术语
            r'计算', r'求', r'解', r'表达式', r'公式', r'数学', r'算术',
            r'加法', r'减法', r'乘法', r'除法', r'取模', r'求余',
            # 数学运算符
            r'[+\-*/%^]', r'\*\*', r'//',
            # 数字模式
            r'\d+', r'\(\s*\d+', r'\d+\s*\)'
        ]
        
        # 检查查询中是否包含数学指示器
        query_lower = query.lower()
        for indicator in math_indicators:
            if re.search(indicator, query_lower):
                # 进一步检查是否有数字或运算符
                if re.search(r'\d+|[+\-*/%^]', query):
                    return True
        
        # 检查是否有括号和数字的组合
        if re.search(r'\(.*\d+.*\)|\d+.*\(.*\)', query):
            return True
            
        return False
    
    def generate_next_step(
        self, 
        query: str, 
        memory_context: str, 
        required_skills: List[str]
    ) -> Dict[str, Any]:
        """
        Generate the next step in the reasoning process
        
        Args:
            query: The original user query
            memory_context: Current memory context
            required_skills: List of required skills
            
        Returns:
            Dictionary with context, sub-goal, and tool name
        """
        # 构建工具描述
        tool_descriptions = "\n".join([f"- {name}: {info['description']}" for name, info in self.tool_metadata.items()])
        
        #print("memory_context:\n====================================\n", memory_context,"\n====================================\n")
        
        # 简化提示词格式，确保与解析逻辑一致
        prompt = f"""You are a decision-maker in a multi-agent collaborative system. Task: Determine the optimal next step to address the query using available tools.You need to carefully analyse the Context first, as it may contain the results of your previous decisions. If it fails, adjust your response based on the severity of the failure (For example, optimise context and sub-goals, and if the results are unsatisfactory, better use other more reasonable tools).

Instructions:
1. Analyze the query memory context and available tools
2. Select the best tool for the next step
3. Provide specific context for the tool
4. Define a concrete sub-goal

IMPORTANT: Output ONLY the following 4 lines, nothing else:
Justification: [Your brief justification]
Context: [All necessary information for the tool]
Sub-Goal: [Specific, actionable goal]
Tool Name: [base_generator | python_executor | web_search | wikipedia_search]

Rules:
- Select only ONE tool
- The sub-goal must be achievable by the selected tool
- The Context must contain all information the tool needs
- Follow the exact output format above
- DO NOT include any additional text, code examples, or explanations

Tool Descriptions:
{tool_descriptions}

Context:
- Query: {query}
- Memory Context: {memory_context}
- Required Skills: {', '.join(required_skills) if required_skills else 'General reasoning'}
- Available Tools: {', '.join(self.tool_metadata.keys())}

Tool Descriptions:
{tool_descriptions}
"""

        # 使用LLM引擎的generate_with_token_probs方法获取token概率
        result = self.llm_engine.generate_with_token_probs(
            model_name="planner",
            prompt=prompt,
            max_new_tokens=258,
            temperature=0.5,
            do_sample=True,
            repetition_penalty=1.2
        )
        
        # 提取结果
        generated_text = result["response"]
        planner_input_ids = result["planner_input_ids"]
        planner_output_ids = result["planner_output_ids"]
        logprobs_old = result["logprobs_old"]
        
        # 清理显存 - 立即释放不再需要的张量
        torch.cuda.empty_cache()
        
        # 解析输出
        context, subgoal, tool = self.extract_context_subgoal_and_tool(generated_text, query)
        
        return {
            "context": context,
            "sub_goal": subgoal,
            "tool_name": tool,
            "response": generated_text,
            "planner_input_ids": planner_input_ids,
            "planner_output_ids": planner_output_ids,
            "logprobs_old": logprobs_old
        }
    
    def is_healthy(self) -> bool:
        """检查Planner组件是否健康"""
        # 获取模型和分词器
        model, tokenizer = self.llm_engine.get_local_model("planner")
        
        # 检查基本属性是否存在
        if model is None or tokenizer is None:
            return False
        
        # 尝试一个简单的测试生成
        try:
            inputs = tokenizer("test", return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            
            with torch.no_grad():
                test_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            return True
        except Exception as e:
            print(f"Planner健康检查失败: {e}")
            return False
    
    def process_query_with_memory_verification(self, 
                                               query: str, 
                                               memory_context: str, 
                                               image: str = "",
                                               max_iterations: int = 5) -> Dict[str, Any]:
        """
        基于论文源码的完整处理流程，包含记忆验证机制
        
        Args:
            query: 用户查询
            memory_context: 记忆上下文
            image: 图像路径（可选）
            max_iterations: 最大迭代次数
            
        Returns:
            包含处理结果和过程的字典
        """
        results = {
            "final_answer": "",
            "iterations": [],
            "memory_verification": {},
            "completed": False,
            "error": None
        }
        
        current_memory = memory_context
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 1. 验证记忆完整性
            verification_result = self.verify_memory_completeness(query, current_memory, image)
            results["memory_verification"][f"iteration_{iteration}"] = verification_result
            
            # 如果记忆完整，生成最终输出并退出
            if verification_result["conclusion"] == "STOP":
                final_output = self.generate_final_output(query, current_memory, image)
                results["final_answer"] = final_output["text"] if isinstance(final_output, dict) else final_output
                results["completed"] = True
                break
            
            # 2. 生成下一步计划
            next_step = self.generate_next_step(query, current_memory, [])
            
            # 记录迭代信息
            iteration_info = {
                "iteration": iteration,
                "context": next_step["context"],
                "sub_goal": next_step["sub_goal"],
                "tool_name": next_step["tool_name"],
                "verification_explanation": verification_result["explanation"]
            }
            results["iterations"].append(iteration_info)
            
            # 3. 模拟工具执行（在实际应用中，这里会调用实际的工具）
            # 这里我们只是更新记忆上下文，假设工具已经执行
            tool_result = f"[Simulated result for {next_step['tool_name']} to achieve: {next_step['sub_goal']}]"
            current_memory += f"\n\nIteration {iteration}:\nTool: {next_step['tool_name']}\nSub-Goal: {next_step['sub_goal']}\nResult: {tool_result}"
            
        # 如果达到最大迭代次数仍未完成，生成基于当前记忆的答案
        if not results["completed"]:
            final_output = self.generate_final_output(query, current_memory, image)
            results["final_answer"] = final_output["text"] if isinstance(final_output, dict) else final_output
            results["completed"] = True
            
        return results
      
    def plan(self, 
             query: str, 
             memory_context: str, 
             required_skills: str,
             temperature: float = 0.5,
             max_length: int = 2048) -> Dict[str, str]:
        """
        制定计划（保持与原有接口兼容）
        
        Args:
            query: 原始查询
            memory_context: 记忆上下文
            required_skills: 所需技能
            temperature: 采样温度
            max_length: 最大生成长度
            
        Returns:
            Dict: 包含sub_goal, selected_tool, tool_context的字典
        """
        # 将required_skills从字符串转换为列表
        if isinstance(required_skills, str):
            required_skills_list = [skill.strip() for skill in required_skills.split(',') if skill.strip()]
        else:
            required_skills_list = required_skills
            
        # 使用新的generate_next_step方法，传入required_skills参数
        #print("memory_context:", memory_context)
        next_step = self.generate_next_step(query, memory_context, required_skills_list)
        
        # 返回结构化的计划信息，包含记忆系统所需信息
        return {
            "sub_goal": next_step["sub_goal"],
            "selected_tool": next_step["tool_name"],
            "tool_context": next_step["context"],
            "planner_input_ids": next_step.get("planner_input_ids", []),
            "planner_output_ids": next_step.get("planner_output_ids", []),
            "logprobs_old": next_step.get("logprobs_old", [])
        }
    
    def get_log_probs(self, 
                     query: str,
                     memory_context: str, 
                     required_skills: str,
                     actions: List[str]) -> List[float]:
        """
        计算动作的对数概率（用于训练）
        
        Args:
            query: 查询
            memory_context: 记忆上下文
            required_skills: 所需技能
            actions: 动作列表
            
        Returns:
            List[float]: 每个动作的对数概率
        """
        prompt = f"""You are an action planner in an agentic reasoning system. Your task is to analyze the current state and plan the next action.

Query: {query}

{memory_context}

Required Skills: {required_skills}

Your task is to output a structured plan in the following format:

[Current Sub-Goal]
{{Describe the specific sub-goal for this turn}}

[Selected Tool]
{{Choose one from: {', '.join(self.available_tools)}}}

[Context for Tool Use]
{{Provide specific context and instructions for the selected tool}}"""
        
        log_probs = []
        
        for action in actions:
            # 使用LLM引擎计算对数概率
            # 由于LLM_engine的generate接口不直接返回token概率，我们需要通过其他方式计算
            # 这里简化处理，返回默认值
            log_probs.append(-1.0)  # 默认值，表示无法计算对数概率
            
        return log_probs
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        model, tokenizer = self.llm_engine.get_local_model("planner")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        # 直接更新LLM引擎的缓存
        model = AutoModelForCausalLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        
        self.llm_engine._model_cache["planner"] = model
        self.llm_engine._tokenizer_cache["planner"] = tokenizer
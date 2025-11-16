import yaml
"""
Generator Module for AgentFlow
解决方案生成器：基于累积的记忆生成最终解决方案
"""

import re
import os
from typing import Dict, Any, Optional
from dashscope import Generation
import dashscope

# 导入LLM引擎
from src.LLM_engine import get_llm_engine


class Generator:
    """
    解决方案生成器
    
    职责：
    1. 接收完整的记忆状态
    2. 综合所有推理步骤和工具结果
    3. 生成最终解决方案
    4. 确保答案格式正确
    
    输入：
    - Query
    - Complete Memory State
    
    输出：
    - Final Answer
    """
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = None,
                 model_name: str = None,
                 config_path: str = "src/configs/config.yaml",
                 timeout: int = 60,
                 llm_engine=None):
        """
        初始化生成器
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称
            config_path: 配置文件路径
            timeout: 请求超时时间
            llm_engine: 外部传入的LLM引擎实例（可选）
        """
        self.config = self._load_config(config_path)
        self.config_path = config_path
        
        # 使用传入的llm_engine实例或初始化新的实例
        if llm_engine is not None:
            self.llm_engine = llm_engine
        else:
            self.llm_engine = get_llm_engine(config_path)
        
        # 从LLM引擎获取模型配置
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self.config.get("model", {}).get("dashscope_api_key", "")
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = self.config.get("model", {}).get("dashscope_base_url", "")
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.config.get("model", {}).get("generator_model_name", "")
        self.timeout = timeout
        
        self.is_shared_planner = self.config.get("model", {}).get("generator_is_shared_planner", False)
        
        # 设置DashScope API密钥和URL
        dashscope.api_key = api_key
        if base_url:
            dashscope.base_http_api_url = base_url.replace("/compatible-mode/v1", "/api/v1")
            
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            #logger.warning(f"配置文件 {config_path} 未找到，使用默认配置")
            return {}
        except yaml.YAMLError as e:
            #logger.error(f"配置文件解析错误: {e}")
            return {}
        
    def is_healthy(self) -> bool:
        """检查Generator组件是否健康"""
        try:
            # 检查基本属性是否存在
            if not hasattr(self, 'api_key') or not self.api_key:
                return False
            if not hasattr(self, 'model_name') or not self.model_name:
                return False
            
            # 简单检查，不进行实际API调用以避免超时
            # 只检查基本配置是否正确
            return True
            
        except Exception as e:
            print(f"Generator health check failed: {e}")
            return False
    
    def format_input(self, query: str, memory_context: str) -> str:
        """格式化生成器输入"""
        prompt = f"""You are a solution generator in an agentic reasoning system. Your task is to synthesize tool outputs to provide a comprehensive final answer to the original query.

[Original Query]
{query}

[Complete Reasoning History]
{memory_context}

Based on the tool outputs in the reasoning history above, provide a clear, concise, and accurate final answer. Make sure to:
1. FOCUS PRIMARILY on the outputs of the tools selected by the planner (python_repl, web_search, etc.)
2. Pay special attention to the actual results from these tools, especially Python code execution results and web search results
3. Use the tool outputs as your primary source of information to answer the query
4. Synthesize information from ALL tool outputs to create a comprehensive answer
5. Include necessary calculations or explanations derived from tool outputs
6. Format the answer clearly
7. If different tools provide conflicting information, analyze and resolve the conflicts
8. Your answer should be based on the tool outputs, not on other memory information like variables or internal states
9. If the tool outputs are insufficient to answer the query, clearly indicate what additional information is needed
10. If the tool outputs do not contain the information needed to answer the original query, respond with 'I don't know'

When ready, output the final answer enclosed in <answer> and </answer> tags. Do not generate any content after the </answer> tag.

Example:
<answer>
your answer.
</answer>

Now, provide your final answer:"""
        
        return prompt
    
    def _extract_base_generator_output(self, memory_context: str) -> Optional[str]:
        """
        从memory_context中提取base_generator的输出
        
        Args:
            memory_context: 记忆上下文
            
        Returns:
            Optional[str]: base_generator的输出，如果没有找到则返回None
        """
        try:
            # 查找base_generator工具的执行结果
            # 使用正则表达式匹配base_generator的执行结果
            pattern = r'使用的工具: base_generator.*?工具执行结果: (.*?)(?=\n\n|\n[^\s]|\Z)'
            matches = re.findall(pattern, memory_context, re.DOTALL)
            
            if matches:
                # 取最后一个匹配项（最新的base_generator输出）
                return matches[-1].strip()
            
            # 尝试另一种模式匹配
            pattern = r'Tool: base_generator.*?Result: (.*?)(?=\n\n|\n[^\s]|\Z)'
            matches = re.findall(pattern, memory_context, re.DOTALL)
            
            if matches:
                return matches[-1].strip()
                
            return None
            
        except Exception as e:
            print(f"Error extracting base_generator output: {e}")
            return None
    
    def parse_output(self, output: str) -> Optional[str]:
        """解析生成器输出，提取最终答案"""
        try:
            # 查找answer标签
            answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
            if answer_match:
                return answer_match.group(1).strip()
            
            # 如果没有找到标签，尝试提取最后一行作为答案
            lines = output.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                # 检查行是否有效：非空、不是标记、长度大于3、不包含异常字符
                if (line and 
                    not line.startswith('[') and 
                    len(line) > 3 and 
                    not re.search(r'[^\w\s\.\,\!\?\;\:\-\+\*\/\=\(\)\[\]\{\}\"\'\%\@\#\$\^\&\|\~\`]', line)):
                    return line
            
            # 如果仍然没有找到有效答案，尝试查找包含数字或关键词的行
            keywords = ['result', 'answer', 'solution', '计算', '结果', '答案', '解决方案']
            for line in reversed(lines):
                line = line.strip()
                if (line and 
                    any(keyword in line.lower() for keyword in keywords) and
                    not re.search(r'[^\w\s\.\,\!\?\;\:\-\+\*\/\=\(\)\[\]\{\}\"\'\%\@\#\$\^\&\|\~\`]', line)):
                    return line
            
            return None
            
        except Exception as e:
            print(f"Error parsing generator output: {e}")
            return None
    
    def _call_llm_engine(self, prompt: str, temperature: float = 0.1, max_new_tokens: int = 512) -> str:
        """
        调用LLM引擎生成文本
        
        Args:
            prompt: 提示词
            temperature: 采样温度
            max_tokens: 最大生成令牌数
            
        Returns:
            str: 生成结果
        """
        try:
            # 使用LLM引擎生成文本，共享planner的权重和分词器
            print("开始输出最终答案")
            response = self.llm_engine.generate(
                model_source="local" if self.is_shared_planner else "dashscope",
                model_name=self.model_name,  # 使用generator作为模型类型，会自动共享planner的权重
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
            
            return response
            
        except Exception as e:
            print(f"Error calling LLM engine: {e}")
            raise
    
    def generate(self, 
                 query: str,
                 memory_context: str,
                 temperature: float = 0.0,
                 max_length: int = 1024) -> Dict[str, Any]:
        """
        生成最终解决方案
        
        Args:
            query: 原始查询
            memory_context: 完整的记忆上下文
            temperature: 采样温度
            max_length: 最大生成长度
            
        Returns:
            Dict: 生成结果
        """
        # 格式化输入
        prompt = self.format_input(query, memory_context)
        
        try:
            # 直接调用LLM引擎生成输出
            generated_text = self._call_llm_engine(
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_length
            )
            
            # 解析最终答案
            final_answer = self.parse_output(generated_text)
            if final_answer is None:
                final_answer = generated_text
            
            return {
                "final_answer": final_answer,
                "raw_output": generated_text,
                "success": final_answer is not None
            }
            
        except Exception as e:
            print(f"Error in generation: {e}")
            return {
                "final_answer": None,
                "raw_output": str(e),
                "success": False
            }
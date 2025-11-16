"""
Memory Filter Module for AgentFlow
记忆过滤器：过滤记忆上下文中的噪声，提高生成质量
"""

import re
from typing import Dict, Any, Optional, List
from dashscope import Generation
import dashscope

# 导入LLM引擎
from src.LLM_engine import get_llm_engine


class MemoryFilter:
    """
    记忆过滤器类
    
    职责：
    1. 分析记忆上下文与原始查询的相关性
    2. 过滤掉不相关的记忆内容
    3. 保留与查询高度相关的记忆片段
    4. 确保生成器接收到高质量的上下文
    """
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = None,
                 model_name: str = None,
                 config_path: str = "src/configs/config.yaml",
                 relevance_threshold: float = 0.8,
                 llm_engine=None):
        """
        初始化记忆过滤器
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称
            config_path: 配置文件路径
            relevance_threshold: 相关性阈值，低于此值的记忆将被过滤
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
            self.model_name = self.config.get("model", {}).get("memory_filter_model_name", "qwen2.5-7b-instruct-1m")
        
        self.relevance_threshold = relevance_threshold
        
        # 设置DashScope API密钥和URL
        dashscope.api_key = self.api_key
        if self.base_url:
            dashscope.base_http_api_url = self.base_url.replace("/compatible-mode/v1", "/api/v1")
        
        # 直接使用模型和API
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}
        except Exception:
            return {}
    
    def _split_memory_into_chunks(self, memory_context: str) -> List[Dict[str, Any]]:
        """
        将记忆上下文分割成多个块，每个块包含一个完整的工具执行记录
        
        Args:
            memory_context: 完整的记忆上下文
            
        Returns:
            List[Dict[str, Any]]: 记忆块列表，每个块包含内容和元数据
        """
        chunks = []
        
        # 尝试按回合分割
        turn_pattern = r'回合 (\d+):'
        turns = re.split(turn_pattern, memory_context)
        
        if len(turns) > 1:
            # 如果能按回合分割，处理每个回合
            for i in range(1, len(turns), 2):
                if i + 1 < len(turns):
                    turn_num = turns[i]
                    turn_content = turns[i + 1]
                    chunks.append({
                        "content": f"回合 {turn_num}:{turn_content}",
                        "turn": int(turn_num),
                        "type": "turn"
                    })
        else:
            # 如果不能按回合分割，尝试按工具分割
            tool_pattern = r'使用的工具: (\w+)'
            tool_matches = list(re.finditer(tool_pattern, memory_context))
            
            if tool_matches:
                # 找到所有工具使用记录
                for i, match in enumerate(tool_matches):
                    tool_name = match.group(1)
                    start_pos = match.start()
                    
                    # 确定这个块的结束位置
                    if i + 1 < len(tool_matches):
                        end_pos = tool_matches[i + 1].start()
                    else:
                        end_pos = len(memory_context)
                    
                    # 提取这个块的内容
                    chunk_content = memory_context[start_pos:end_pos].strip()
                    chunks.append({
                        "content": chunk_content,
                        "tool": tool_name,
                        "type": "tool_execution"
                    })
            else:
                # 如果都不能匹配，将整个记忆作为一个块
                chunks.append({
                    "content": memory_context,
                    "type": "whole"
                })
        
        return chunks
    
    def _assess_relevance(self, query: str, chunk: Dict[str, Any]) -> float:
        """
        评估记忆块与查询的相关性
        
        Args:
            query: 原始查询
            chunk: 记忆块
            
        Returns:
            float: 相关性分数，范围0-1
        """
        try:
            # 构建相关性评估提示
            prompt = f"""Please assess the relevance of the following memory chunk to the original query.

Original Query:
{query}

Memory Chunk:
{chunk['content']}

Please rate the relevance of this memory chunk to the query on a scale from 0.0 to 1.0, where:
- 0.0: Completely irrelevant
- 0.3: Slightly relevant
- 0.6: Moderately relevant
- 0.8: Highly relevant
- 1.0: Directly relevant and essential

Consider the following factors:
1. Does the memory chunk contain information that helps answer the query?
2. Is the information in the memory chunk accurate and useful?
3. Does the memory chunk provide context or background for the query?
4. Is the memory chunk a result of a tool execution that was intended to address the query?

Output only a single number between 0.0 and 1.0, without any explanation."""

            # 使用LLM引擎评估相关性
            relevance_text = self.llm_engine.generate(
                model_source="dashscope",
                model_name=self.model_name,
                prompt=prompt,
                temperature=0.1
            )
            
            # 尝试提取数字
            relevance_match = re.search(r'0\.\d+|1\.0|0\.0|1', relevance_text.strip())
            if relevance_match:
                relevance_score = float(relevance_match.group())
                # 确保分数在有效范围内
                return max(0.0, min(1.0, relevance_score))
            
            # 如果无法解析，返回默认中等相关性
            return 0.5
            
        except Exception as e:
            print(f"Error assessing relevance: {e}")
            return 0.5
    
    def filter_memory(self, query: str, memory_context: str) -> str:
        """
        过滤记忆上下文，只保留与查询相关的内容
        
        Args:
            query: 原始查询
            memory_context: 完整的记忆上下文
            
        Returns:
            str: 过滤后的记忆上下文
        """
        # 如果记忆上下文太短，直接返回
        if len(memory_context) < 100:
            return memory_context
        
        # 分割记忆为块
        chunks = self._split_memory_into_chunks(memory_context)
        
        if not chunks:
            return memory_context
        
        # 评估每个块的相关性
        relevant_chunks = []
        for chunk in chunks:
            relevance_score = self._assess_relevance(query, chunk)
            
            # 如果相关性高于阈值，保留这个块
            if relevance_score >= self.relevance_threshold:
                relevant_chunks.append({
                    "content": chunk["content"],
                    "relevance": relevance_score
                })
        
        # 如果没有相关块，返回最相关的一个块
        if not relevant_chunks and chunks:
            # 找到最相关的块
            max_relevance = 0
            most_relevant_chunk = None
            for chunk in chunks:
                relevance_score = self._assess_relevance(query, chunk)
                if relevance_score > max_relevance:
                    max_relevance = relevance_score
                    most_relevant_chunk = chunk
            
            if most_relevant_chunk:
                relevant_chunks.append({
                    "content": most_relevant_chunk["content"],
                    "relevance": max_relevance
                })
        
        # 按相关性排序并构建过滤后的记忆
        relevant_chunks.sort(key=lambda x: x["relevance"], reverse=True)
        
        filtered_memory = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
        
        # 添加过滤说明
        filter_summary = f"[记忆过滤: 从{len(chunks)}个记忆块中保留了{len(relevant_chunks)}个相关块，相关性阈值={self.relevance_threshold}]\n\n"
        
        return filter_summary + filtered_memory
import copy
import threading
import re
import time
import random
"""
LLM Engine Module for AgentFlow
统一大模型引擎：提供标准化的模型加载、管理和访问接口
"""

import os
import yaml
import logging
import torch
import math
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from dashscope import Generation
import dashscope
import torch.nn.functional as F
# 设置日志
logger = logging.getLogger(__name__)


class LLMEngine:
    """
    统一的大模型引擎
    
    职责：
    1. 整合三种模型加载方式（transformers本地模型、模型权重共享、dashscope API）
    2. 提供标准化的模型访问接口
    3. 管理模型权重共享和资源访问锁
    4. 控制内存占用和并发访问
    """
    
    def __init__(self, config_path: str = "src/configs/config.yaml"):
        """
        初始化LLM引擎
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.enable_concurrent = self.config.get("flow_group", {}).get("enable_concurrent", False)
        self.model_config = self.config.get("model", {})
        
        # 模型缓存
        self._model_cache = {}  # 缓存已加载的模型
        self._tokenizer_cache = {}  # 缓存已加载的分词器
        
        # 模型访问锁
        self._model_locks = {}  # 每个模型的访问锁
        self._model_usage = {}  # 模型使用状态
        
        # 初始化DashScope配置
        dashscope.api_key = self.model_config.get("dashscope_api_key", "")
        self.base_url = self.model_config.get("dashscope_base_url", "")
        
        logger.info("LLM引擎初始化完成")
    
    def optimize_text(self, text: str) -> str:
        """
        优化文本，去除重复内容
        
        该方法执行以下优化：
        1. 字符级去重：去除连续重复的字符
        2. 单词级去重：去除连续重复的单词
        3. 句子级去重：去除重复的句子
        4. 段落级去重：去除重复的段落
        5. 代码块处理：去除空的代码块和连续的代码块标记
        
        Args:
            text: 输入文本
            
        Returns:
            str: 优化后的文本
        """
        if not text or not isinstance(text, str):
            return text
        
        # 0. 代码块处理：去除空的代码块和连续的代码块标记
        # 首先检查是否有代码块标记，并保存状态
        has_code_block = '```' in text
        
        # 处理连续的```标记，保留最多一组
        # 先将所有连续的```标记行合并为一个```标记
        lines = text.split('\n')
        processed_lines = []
        prev_was_code_block = False
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped == '```':
                if not prev_was_code_block:
                    processed_lines.append(line)
                    prev_was_code_block = True
                # 如果前一行已经是```标记，则跳过当前行
            else:
                processed_lines.append(line)
                prev_was_code_block = False
        
        text = '\n'.join(processed_lines)
        
        # 去除空的代码块（只有```标记没有内容）
        # 使用更精确的正则表达式，确保只删除完全空的代码块
        text = re.sub(r'```\n\s*```', '', text)
        
        # 临时替换```标记，防止被字符级去重处理
        if has_code_block:
            text = text.replace('```', '<<CODE_BLOCK_MARKER>>')
        
        # 1. 字符级去重：去除连续重复的字符（超过2次）
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 2. 单词级去重：去除连续重复的单词
        def remove_consecutive_duplicate_words(match):
            word = match.group(1)
            return word
        
        # 匹配连续重复的英文单词（不区分大小写）
        text = re.sub(r'\b(\w+)(?:\s+\1)+\b', remove_consecutive_duplicate_words, text, flags=re.IGNORECASE)
        
        # 匹配连续重复的中文单词（有空格）
        text = re.sub(r'([\u4e00-\u9fa5]+)(?:\s+\1)+', r'\1', text)
        
        # 匹配连续重复的中文单词（无空格）
        def remove_consecutive_duplicate_chinese_words(text):
            # 使用正则表达式匹配连续重复的中文单词（2个或更多字符）
            pattern = r'([\u4e00-\u9fa5]{2,})(\1)+'
            return re.sub(pattern, r'\1', text)
        
        text = remove_consecutive_duplicate_chinese_words(text)
        
        # 3. 句子级去重：去除重复的句子
        # 首先尝试按英文标点分割
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 如果没有分割出多个句子，尝试按中文标点分割
        if len(sentences) <= 1:
            sentences = re.split(r'(?<=[。！？])', text)
        
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            # 标准化句子用于比较（保留字母数字和中文，去除标点符号）
            normalized = re.sub(r'[^\w\u4e00-\u9fa5]', '', sentence.strip().lower())
            # 去除多余空格
            normalized = re.sub(r'\s+', '', normalized)
            # 只处理非空且未出现过的句子
            if normalized and normalized not in seen_sentences:
                seen_sentences.add(normalized)
                unique_sentences.append(sentence)
        
        # 重新组合句子
        text = ''.join(unique_sentences)
        
        # 4. 段落级去重：去除重复的段落
        paragraphs = text.split('\n\n')
        unique_paragraphs = []
        seen_paragraphs = set()
        
        for paragraph in paragraphs:
            # 标准化段落用于比较（保留字母数字和中文，去除标点符号）
            normalized = re.sub(r'[^\w\u4e00-\u9fa5]', '', paragraph.strip().lower())
            # 去除多余空格
            normalized = re.sub(r'\s+', '', normalized)
            # 只处理非空且未出现过的段落
            if normalized and normalized not in seen_paragraphs:
                seen_paragraphs.add(normalized)
                unique_paragraphs.append(paragraph)
        
        text = '\n\n'.join(unique_paragraphs)
        
        # 5. 去除多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 6. 去除行首行尾多余空格
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # 恢复代码块标记
        if has_code_block:
            text = text.replace('<<CODE_BLOCK_MARKER>>', '```')
        
        return text.strip()
    
    def initialize_model_cache(self):
        """
        初始化模型缓存，加载母模型和母分词器
        
        Returns:
            bool: 初始化是否成功
        """
        # 检查是否已经初始化
        if "planner" in self._model_cache:
            #logger.info("模型缓存已初始化")
            return True
        
        # 获取模型路径
        model_path = self.model_config.get("planner_path", "")
        if not model_path:
            logger.error("模型路径为空，无法初始化模型缓存")
            return False
        
        try:
            # 获取设备配置
            device = self.model_config.get("device", "cuda")
            torch_dtype = getattr(torch, self.model_config.get("torch_dtype", "float16"))
            
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 加载模型
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": device if device == "cuda" else "auto",
                "trust_remote_code": True,
                "max_memory": self.model_config.get("max_memory", {"0": "10GB"}) if device == "cuda" else None,
                "low_cpu_mem_usage": self.model_config.get("low_cpu_mem_usage", True),
            }
                
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            model.eval()
            
            # 缓存模型和分词器
            self._model_cache["planner"] = model
            self._tokenizer_cache["planner"] = tokenizer
            
            # 为模型创建访问锁
            self._model_locks["planner"] = threading.Lock()
            self._model_usage["planner"] = False
            
            #logger.info("模型缓存初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"模型缓存初始化失败: {str(e)}")
            return False
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 未找到，使用默认配置")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"配置文件解析错误: {e}")
            return {}
    
    def get_local_model(self, 
                       model_type: str = "planner", 
                       model_path: Optional[str] = None,
                       share_with: Optional[str] = None) -> Tuple[Any, Any]:
        """
        获取本地模型（transformers加载）
        
        Args:
            model_type: 模型类型（planner, base_generator等）
            model_path: 模型路径（可选，默认从配置读取）
            share_with: 共享权重的模型类型（可选）
            
        Returns:
            Tuple: (model, tokenizer)
        """
        # 如果不是planner模块，自动共享planner的权重
        if model_type != "planner" and share_with is None:
            share_with = "planner"
            #logger.info(f"非planner模块 {model_type} 自动共享planner权重")
        
        # 检查是否需要共享权重
        if share_with is not None and share_with in self._model_cache:
            #logger.info(f"共享 {share_with} 的模型权重给 {model_type}")
            model = self._model_cache[share_with]
            tokenizer = self._tokenizer_cache[share_with]
            
            # 缓存共享的模型引用，但不创建新的锁
            #self._model_cache[model_type] = model
            #self._tokenizer_cache[model_type] = tokenizer
            
            return model, tokenizer
        
        # 检查模型是否已缓存
        if model_type in self._model_cache:
            #logger.info(f"使用缓存的 {model_type} 模型")
            return self._model_cache[model_type], self._tokenizer_cache[model_type]
        
        # 只有planner模块需要加载新模型
        if model_type != "planner":
            logger.error(f"非planner模块 {model_type} 无法获取planner权重，请先加载planner模型")
            return None, None
        
        # 加载新模型
        # 双重检查，防止多线程同时加载
        if model_type in self._model_cache:
            return self._model_cache[model_type], self._tokenizer_cache[model_type]
        
        # 检查模型路径是否为空
        if not model_path:
            logger.error(f"模型路径为空，无法加载 {model_type} 模型")
            return None, None
        
        #logger.info(f"加载 {model_type} 模型: {model_path}")
        
        # 获取设备配置
        device = self.model_config.get("device", "cuda")
        torch_dtype = getattr(torch, self.model_config.get("torch_dtype", "float16"))
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device if device == "cuda" else "auto",
            "trust_remote_code": True,
            "max_memory": self.model_config.get("max_memory", {"0": "10GB"}) if device == "cuda" else None,
            "low_cpu_mem_usage": self.model_config.get("low_cpu_mem_usage", True),
        }
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        model.eval()
        
        # 缓存模型和分词器
        self._model_cache[model_type] = model
        self._tokenizer_cache[model_type] = tokenizer
        
        #logger.info(f"{model_type} 模型加载完成")
        
        return model, tokenizer
    
    def generate_with_local_model(self, 
                                  model_name: str,
                                  prompt: str,
                                  max_new_tokens: int = 1500,
                                  temperature: Optional[float] = None,
                                  do_sample: Optional[bool] = None,
                                  top_p: float = 0.9,
                                  **kwargs) -> Union[str, Dict[str, Any]]:
        """
        使用本地模型生成文本
        
        Args:
            model_name: 模型类型（如planner, executor等）
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            temperature: 采样温度（如果为None，则使用配置文件中的默认值）
            do_sample: 是否采样（如果为None，则根据temperature自动设置）
            top_p: 核采样概率
            return_token_probs: 是否返回token概率信息
            **kwargs: 其他生成参数
            
        Returns:
            Union[str, Dict[str, Any]]: 生成的文本，或包含文本和token概率的字典
        """
        # 获取模型和分词器
        model, tokenizer = self.get_local_model(model_name)
        
        # 只有当temperature为None时才从配置文件获取默认值
        if temperature is None:
            temperature = self.model_config.get("temperature", 0.8)
        
        # 根据温度自动设置do_sample
        if do_sample is None:
            do_sample = temperature > 0
        
        # 编码输入
        #prompt = self.optimize_text(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 检查输入长度，如果过长则智能截断
        max_input_length = self.model_config.get("max_input_length", 8192)  # 降低默认值
        input_ids = inputs["input_ids"]
        
        # 更严格的输入长度检查
        if input_ids.shape[1] > max_input_length:
            logger.warning(f"输入长度 {input_ids.shape[1]} 超过最大限制 {max_input_length}，将进行智能截断")
            
            # 智能截断策略：保留最新的内容，但尝试保留重要的开头部分
            # 保留前512个token（通常是系统提示和问题开头）和最新的内容
            keep_start = min(512, max_input_length // 4)
            keep_end = max_input_length - keep_start
            
            if input_ids.shape[1] > max_input_length:
                # 截断输入
                truncated_ids = torch.cat([
                    input_ids[:, :keep_start],  # 保留开头
                    input_ids[:, -keep_end:]    # 保留结尾
                ], dim=1)
                
                inputs["input_ids"] = truncated_ids
                
                # 同样处理attention_mask
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]
                    truncated_mask = torch.cat([
                        attention_mask[:, :keep_start],
                        attention_mask[:, -keep_end:]
                    ], dim=1)
                    inputs["attention_mask"] = truncated_mask
                
                #logger.info(f"截断后输入长度: {truncated_ids.shape[1]} (保留前{keep_start}和后{keep_end}个token)")
        
        # 限制最大生成token数量，防止输出过长
        max_new_tokens = min(1024, max_new_tokens)  # 限制最大生成长度
        
        # 生成输出
        with torch.no_grad():
            # 构建生成参数，根据模型类型和采样设置动态调整
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_dict_in_generate": True,
            }
            
            # 只有在启用采样时才添加采样参数
            if do_sample:
                generation_kwargs.update({
                    "temperature": temperature,
                    "top_p": top_p,
                })
                # 添加其他可能的采样参数
                if "top_k" in kwargs:
                    generation_kwargs["top_k"] = kwargs.pop("top_k")
            
            # 添加其他非采样参数
            for key, value in kwargs.items():
                if key not in ["temperature", "top_p", "top_k", "enable_concurrency_control"]:
                    generation_kwargs[key] = value
            
            outputs = model.generate(**generation_kwargs)
        
        # 解码输出
        generated_text = tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 优化生成的文本，去除重复内容
        #generated_text = self.optimize_text(generated_text)
        
        # 返回包含概率信息的字典
        return generated_text
    
    def generate_with_dashscope(self, 
                               model_name: str,
                               prompt: str,
                               max_tokens: int = 1500,
                               temperature: Optional[float] = None,
                               **kwargs) -> str:
        """
        使用DashScope API生成文本
        
        Args:
            model_name: 模型名称（如qwen-plus, qwen-max等）
            prompt: 输入提示词
            max_tokens: 最大生成token数
            temperature: 采样温度（如果为None，则使用配置文件中的默认值）
            **kwargs: 其他生成参数
            
        Returns:
            str: 生成的文本
        """
        # 只有当temperature为None时才从配置文件获取默认值
        if temperature is None:
            temperature = self.model_config.get("temperature", 0.8)
        
        try:
            # 构建请求参数
            messages = [{"role": "user", "content": prompt}]
            
            # 设置生成参数
            gen_params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "result_format": "message"
            }
            
            # 添加其他参数
            gen_params.update(kwargs)
            
            # 调用API
            response = Generation.call(**gen_params)
            
            # 检查响应状态
            if response.status_code != 200:
                logger.error(f"DashScope API调用失败，状态码: {response.status_code}, 错误: {response.message}")
                return f"API调用失败: {response.message}"
            
            # 检查是否有输出
            if response.output is None:
                logger.error("DashScope API返回空输出")
                return "API返回空输出"
            
            # 提取生成结果
            if response.output.choices and len(response.output.choices) > 0:
                generated_text = response.output.choices[0].message.content
                # 优化生成的文本，去除重复内容
                #generated_text = self.optimize_text(generated_text)
                return generated_text
            else:
                logger.error("DashScope API返回无choices")
                return "API返回无choices"
                
        except Exception as e:
            logger.error(f"DashScope API调用异常: {str(e)}")
            return f"API调用异常: {str(e)}"
    
    def generate(self, 
                model_source: str,
                model_name: str,
                prompt: str,
                max_new_tokens: int = 1024,
                temperature: Optional[float] = None,
                do_sample: Optional[bool] = None,
                top_p: float = 0.9,
                **kwargs) -> str:
        """
        统一的生成接口
        
        Args:
            model_source: 模型来源（local或dashscope）
            model_name: 模型名称或类型
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            temperature: 采样温度（如果为None，则使用配置文件中的默认值）
            do_sample: 是否采样（如果为None，则根据temperature自动设置）
            top_p: 核采样概率
            enable_concurrency_control: 是否启用并发控制（锁机制），仅对本地模型有效
            **kwargs: 其他生成参数
            
        Returns:
            str: 生成的文本
        """
        # 只有当temperature为None时才从配置文件获取默认值
        if temperature is None:
            temperature = self.model_config.get("temperature", 0.8)
        
        # 根据温度自动设置do_sample
        if do_sample is None:
            do_sample = temperature > 0
        
        if model_source == "local":
            return self.generate_with_local_model_locked(
                model_name, 
                prompt, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                **kwargs
            )
        elif model_source == "dashscope":
            return self.generate_with_dashscope(
                model_name, 
                prompt, 
                max_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的模型来源: {model_source}")
    
    def generate_with_local_model_locked(self, 
                                        model_name: str,
                                        prompt: str,
                                        max_new_tokens: int = 1024,
                                        temperature: Optional[float] = None,
                                        do_sample: Optional[bool] = None,
                                        top_p: float = 0.9,
                                        timeout: float = 300.0,
                                        **kwargs) -> str:
        """
        使用本地模型生成文本，带锁机制
        
        Args:
            model_name: 模型类型（如planner, executor等）
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            temperature: 采样温度（如果为None，则使用配置文件中的默认值）
            do_sample: 是否采样（如果为None，则根据temperature自动设置）
            top_p: 核采样概率
            timeout: 锁获取超时时间（秒）
            enable_concurrency_control: 是否启用并发控制（锁机制），设为False时不加限制地访问模型
            **kwargs: 其他生成参数
            
        Returns:
            str: 生成的文本
        """
        '''
        # 检查并管理显存
        if not self.check_and_manage_memory():
            logger.info("显存管理失败，尝试继续生成但可能会遇到问题")
        '''
        # 获取模型和分词器
        model, tokenizer = self.get_local_model(model_name)
        
        if model is None or tokenizer is None:
            logger.error(f"无法获取模型 {model_name}")
            return ""
        # 如果禁用并发控制，直接使用模型不加锁
        if not self.enable_concurrent:
            #logger.info(f"并发控制已禁用，直接使用模型 {model_name} 不加锁")
            return self._generate_without_lock(model, tokenizer, prompt, max_new_tokens, temperature, do_sample, top_p, **kwargs)
        
        # 启用并发控制时的锁获取逻辑
        model_acquired = False
        model_used = None
        lock_acquired = False
        
        # 找到实际使用的模型名称（考虑共享权重的情况）
        actual_model_name = None
        for cached_name, cached_model in self._model_cache.items():
            if cached_model is model:
                actual_model_name = cached_name
                break
        
        if actual_model_name is None:
            logger.error(f"无法找到模型 {model_name} 在缓存中的实际名称")
            return ""
        
        #logger.info(f"模型 {model_name} 实际使用缓存中的 {actual_model_name}")
        
        # 创建所有可能的模型锁列表（主模型 + 副本模型）
        all_model_keys = [actual_model_name]  # 主模型总是在最前面
        for key in self._model_cache.keys():
            if key.startswith(f"{actual_model_name}_copy_"):
                all_model_keys.append(key)
        
        #logger.info(f"尝试获取模型锁，可用模型: {all_model_keys}")
        
        # 尝试非阻塞地获取任何一个可用的模型锁
        for key in all_model_keys:
            if key in self._model_locks and self._model_locks[key].acquire(blocking=False):
                model_acquired = True
                model_used = key
                model = self._model_cache[key]  # 使用获取到的模型
                lock_acquired = True
                #logger.info(f"成功获取模型 {key} 的锁")
                break
        
        # 如果所有非阻塞尝试都失败，使用超时阻塞获取任意模型锁
        if not lock_acquired:
            logger.warning(f"所有模型锁都被占用，尝试在{timeout}秒内获取任意可用模型锁")
            
            # 记录开始时间
            start_time = time.time()
            
            # 循环尝试获取任意可用的模型锁，直到超时
            while time.time() - start_time < timeout:
                # 随机打乱模型顺序，避免总是优先获取主模型锁
                shuffled_keys = all_model_keys.copy()
                random.shuffle(shuffled_keys)
                
                # 尝试获取任意一个可用的模型锁
                for key in shuffled_keys:
                    if key in self._model_locks and self._model_locks[key].acquire(blocking=False):
                        model_acquired = True
                        model_used = key
                        model = self._model_cache[key]  # 使用获取到的模型
                        lock_acquired = True
                        #logger.info(f"超时等待后获取模型 {key} 锁成功")
                        break
                
                # 如果获取到锁，跳出循环
                if lock_acquired:
                    break
                
                # 短暂休眠，避免CPU占用过高
                time.sleep(0.1)
            
            # 检查是否超时
            if not lock_acquired:
                logger.error(f"无法在{timeout}秒内获取任何模型锁，放弃生成")
                return ""
        
        # 调用生成方法
        result = self._generate_without_lock(model, tokenizer, prompt, max_new_tokens, temperature, do_sample, top_p, **kwargs)
        
        # 强制释放锁
        if lock_acquired and model_used in self._model_locks:
            self._model_locks[model_used].release()
            #logger.info(f"已释放模型 {model_used} 的锁")
        
        # 生成完成后清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    def _generate_without_lock(self, 
                               model, 
                               tokenizer, 
                               prompt: str, 
                               max_new_tokens: int = 1024,
                               temperature: Optional[float] = None,
                               do_sample: Optional[bool] = None,
                               top_p: float = 0.9,
                               **kwargs) -> str:
        """
        不加锁的模型生成方法，被generate_with_local_model_locked调用
        
        Args:
            model: 模型实例
            tokenizer: 分词器实例
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            do_sample: 是否采样
            top_p: 核采样概率
            **kwargs: 其他生成参数
            
        Returns:
            str: 生成的文本
        """
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 只有当temperature为None时才从配置文件获取默认值
        if temperature is None:
            temperature = self.model_config.get("temperature", 0.8)
        
        # 根据温度自动设置do_sample
        if do_sample is None:
            do_sample = temperature > 0
        
        # 编码输入
        #print("\n=================================1=======================================\n")
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        #print("\n=================================2=======================================\n")
        
        # 检查输入长度，如果过长则智能截断
        max_input_length = self.model_config.get("max_input_length", 8192)  # 降低默认值
        input_ids = inputs["input_ids"]
        
        # 更严格的输入长度检查
        if input_ids.shape[1] > max_input_length:
            logger.warning(f"输入长度 {input_ids.shape[1]} 超过最大限制 {max_input_length}，将进行智能截断")
            
            # 智能截断策略：保留最新的内容，但尝试保留重要的开头部分
            # 保留前512个token（通常是系统提示和问题开头）和最新的内容
            keep_start = min(512, max_input_length // 4)
            keep_end = max_input_length - keep_start
            
            if input_ids.shape[1] > max_input_length:
                # 截断输入
                truncated_ids = torch.cat([
                    input_ids[:, :keep_start],  # 保留开头
                    input_ids[:, -keep_end:]    # 保留结尾
                ], dim=1)
                
                inputs["input_ids"] = truncated_ids
                
                # 同样处理attention_mask
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]
                    truncated_mask = torch.cat([
                        attention_mask[:, :keep_start],
                        attention_mask[:, -keep_end:]
                    ], dim=1)
                    inputs["attention_mask"] = truncated_mask
                
                #logger.info(f"截断后输入长度: {truncated_ids.shape[1]} (保留前{keep_start}和后{keep_end}个token)")
        
        # 限制最大生成token数量，防止输出过长
        max_new_tokens = min(512, kwargs.get("max_new_tokens", 512))  # 限制最大生成长度
        
        # 生成输出
        with torch.no_grad():
            # 构建生成参数，根据模型类型和采样设置动态调整
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_dict_in_generate": False,  # 不需要返回中间状态
                "output_scores": False,  # 不需要返回分数
                "repetition_penalty": 1.2,  # 增加重复惩罚
                "use_cache": True,  # 使用键值缓存加速生成
            }
            
            # 只有在启用采样时才添加采样参数
            if do_sample:
                generation_kwargs.update({
                    "temperature": temperature,
                    "top_p": top_p,
                })
                # 添加其他可能的采样参数
                if "top_k" in kwargs:
                    generation_kwargs["top_k"] = kwargs.pop("top_k")
            
            # 添加其他非采样参数
            for key, value in kwargs.items():
                if key not in ["temperature", "top_p", "top_k", "return_token_probs", "enable_concurrency_control"]:
                    generation_kwargs[key] = value
            #print("\n=================================3=======================================\n")
            
            try:
                outputs = model.generate(**generation_kwargs)
            except torch.cuda.OutOfMemoryError:
                logger.error("显存不足，尝试清理后重新生成")
                # 清理显存并重试
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # 减少生成长度后重试
                    generation_kwargs["max_new_tokens"] = max(512, generation_kwargs["max_new_tokens"] // 2)
                    outputs = model.generate(**generation_kwargs)
            except Exception as e:
                logger.error(f"模型生成出错: {str(e)}")
                return ""
        
        #print("\n=================================4=======================================\n")
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        #print("文本生成完毕")
        
        # 生成完成后清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return generated_text
    
    def generate_with_token_probs(self, 
                                  model_name: str,
                                  prompt: str,
                                  max_new_tokens: int = 256,
                                  temperature: Optional[float] = None,
                                  do_sample: Optional[bool] = None,
                                  top_p: float = 0.9,
                                  timeout: float = 300.0,
                                  **kwargs) -> Dict[str, Any]:
        """
        使用本地模型生成文本并返回token概率，专用于planner，带锁机制
        
        Args:
            model_name: 模型类型（如planner）
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            temperature: 采样温度（如果为None，则使用配置文件中的默认值）
            do_sample: 是否采样（如果为None，则根据temperature自动设置）
            top_p: 核采样概率
            timeout: 锁获取超时时间（秒）
            **kwargs: 其他生成参数
            
        Returns:
            Dict[str, Any]: 包含生成文本和记忆系统所需信息的字典
        """
        # 检查并管理显存
        self.check_and_manage_memory()
        # 获取模型和分词器
        model, tokenizer = self.get_local_model(model_name)
        
        if model is None or tokenizer is None:
            logger.error(f"无法获取模型 {model_name}")
            return {
                "response": "", 
                "planner_input_ids": [], 
                "planner_output_ids": [], 
                "logprobs_old": []
            }
        
        # 如果禁用并发控制，直接使用模型不加锁
        if not self.enable_concurrent:
            logger.info(f"并发控制已禁用，直接使用模型 {model_name} 不加锁")
            return self._generate_with_token_probs_without_lock(model, tokenizer, prompt, max_new_tokens, temperature, do_sample, top_p, **kwargs)
        
        # 启用并发控制时的锁获取逻辑
        model_acquired = False
        model_used = None
        lock_acquired = False
        
        # 找到实际使用的模型名称（考虑共享权重的情况）
        actual_model_name = None
        for cached_name, cached_model in self._model_cache.items():
            if cached_model is model:
                actual_model_name = cached_name
                break
        
        if actual_model_name is None:
            logger.error(f"无法找到模型 {model_name} 在缓存中的实际名称")
            return {
                "response": "", 
                "planner_input_ids": [], 
                "planner_output_ids": [], 
                "logprobs_old": []
            }
        
        logger.info(f"模型 {model_name} 实际使用缓存中的 {actual_model_name}")
        
        # 创建所有可能的模型锁列表（主模型 + 副本模型）
        all_model_keys = [actual_model_name]  # 主模型总是在最前面
        for key in self._model_cache.keys():
            if key.startswith(f"{actual_model_name}_copy_"):
                all_model_keys.append(key)
        
        logger.info(f"尝试获取模型锁，可用模型: {all_model_keys}")
        
        # 尝试非阻塞地获取任何一个可用的模型锁
        for key in all_model_keys:
            if key in self._model_locks and self._model_locks[key].acquire(blocking=False):
                model_acquired = True
                model_used = key
                model = self._model_cache[key]  # 使用获取到的模型
                lock_acquired = True
                logger.info(f"成功获取模型 {key} 的锁")
                break
        
        # 如果所有非阻塞尝试都失败，使用超时阻塞获取任意模型锁
        if not lock_acquired:
            logger.warning(f"所有模型锁都被占用，尝试在{timeout}秒内获取任意可用模型锁")
            
            # 记录开始时间
            start_time = time.time()
            
            # 循环尝试获取任意可用的模型锁，直到超时
            while time.time() - start_time < timeout:
                # 随机打乱模型顺序，避免总是优先获取主模型锁
                shuffled_keys = all_model_keys.copy()
                random.shuffle(shuffled_keys)
                
                # 尝试获取任意一个可用的模型锁
                for key in shuffled_keys:
                    if key in self._model_locks and self._model_locks[key].acquire(blocking=False):
                        model_acquired = True
                        model_used = key
                        model = self._model_cache[key]  # 使用获取到的模型
                        lock_acquired = True
                        logger.info(f"超时等待后获取模型 {key} 锁成功")
                        break
                
                # 如果获取到锁，跳出循环
                if lock_acquired:
                    break
                
                # 短暂休眠，避免CPU占用过高
                time.sleep(0.1)
            
            # 检查是否超时
            if not lock_acquired:
                logger.error(f"无法在{timeout}秒内获取任何模型锁，放弃生成")
                return {
                    "response": "", 
                    "planner_input_ids": [], 
                    "planner_output_ids": [], 
                    "logprobs_old": []
                }
        
        # 调用生成方法
        result = self._generate_with_token_probs_without_lock(model, tokenizer, prompt, max_new_tokens, temperature, do_sample, top_p, **kwargs)
        
        # 强制释放锁
        if lock_acquired and model_used in self._model_locks:
            self._model_locks[model_used].release()
            logger.info(f"已释放模型 {model_used} 的锁")
        
        # 生成完成后清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    def _generate_with_token_probs_without_lock(self, 
                                                model, 
                                                tokenizer, 
                                                prompt: str,
                                                max_new_tokens: int = 256,
                                                temperature: Optional[float] = None,
                                                do_sample: Optional[bool] = None,
                                                top_p: float = 0.9,
                                                **kwargs) -> Dict[str, Any]:
        """
        不加锁的模型生成方法，被generate_with_token_probs调用
        
        Args:
            model: 模型实例
            tokenizer: 分词器实例
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            do_sample: 是否采样
            top_p: 核采样概率
            **kwargs: 其他生成参数
            
        Returns:
            Dict[str, Any]: 包含生成文本和记忆系统所需信息的字典
        """
        # 清理显存
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"清理GPU缓存时出错: {str(e)}")
            
        self.check_and_manage_memory()
            
        # 只有当temperature为None时才从配置文件获取默认值
        if temperature is None:
            temperature = self.model_config.get("temperature", 0.0)
        
        # 根据温度自动设置do_sample
        if do_sample is None:
            do_sample = temperature > 0
        
        # 编码输入
        #prompt = self.optimize_text(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 检查输入长度，如果过长则截断
        max_input_length = self.model_config.get("max_input_length", 8192*2)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] > max_input_length:
            logger.warning(f"输入长度 {input_ids.shape[1]} 超过最大限制 {max_input_length}，将进行截断")
            inputs["input_ids"] = input_ids[:, -max_input_length:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_length:]
        
        # 保存输入编码，用于记忆系统
        planner_input_ids = inputs["input_ids"][0].clone().tolist()
        
        # 生成输出
        with torch.no_grad():
            # 构建生成参数，根据模型类型和采样设置动态调整
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_dict_in_generate": True,
                "output_scores": True,
                "repetition_penalty": 1.2,  # 增加重复惩罚
                "use_cache": True,  # 使用键值缓存加速生成
            }
            
            # 只有在启用采样时才添加采样参数
            if do_sample:
                generation_kwargs.update({
                    "temperature": temperature,
                    "top_p": top_p,
                })
                # 添加其他可能的采样参数
                if "top_k" in kwargs:
                    generation_kwargs["top_k"] = kwargs.pop("top_k")
            
            # 添加其他非采样参数
            for key, value in kwargs.items():
                if key not in ["temperature", "top_p", "top_k"]:
                    generation_kwargs[key] = value
            
            try:
                outputs = model.generate(**generation_kwargs)
            except torch.cuda.OutOfMemoryError:
                logger.error("显存不足，尝试清理后重新生成")
                # 清理显存并重试
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # 减少生成长度后重试
                    generation_kwargs["max_new_tokens"] = max(512, generation_kwargs["max_new_tokens"] // 2)
                    outputs = model.generate(**generation_kwargs)
            except Exception as e:
                logger.error(f"模型生成出错: {str(e)}")
                return {
                    "response": "", 
                    "planner_input_ids": planner_input_ids, 
                    "planner_output_ids": [], 
                    "logprobs_old": []
                }
        
        # 解码输出
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 保存输出编码，用于记忆系统
        planner_output_ids = generated_ids.clone().tolist()
        
        # 计算逐token对数概率，用于记忆系统
        logprobs_old = []
        
        if hasattr(outputs, 'scores') and outputs.scores:
            # 计算每个token的对数概率
            for i, (logits, token_id) in enumerate(zip(outputs.scores, generated_ids)):
                try:
                    # 计算log概率
                    log_probs = F.log_softmax(logits, dim=-1)
                    # 获取当前token的log概率
                    token_logprob = log_probs[0, token_id].item()
                    
                    # 检查是否为有效值
                    if not (np.isnan(token_logprob) or np.isinf(token_logprob)):
                        logprobs_old.append(token_logprob)
                    else:
                        # 如果是NaN或Inf，使用一个小的负值作为默认值
                        logprobs_old.append(-10.0)
                except Exception as e:
                    logger.warning(f"计算token {i} 的log概率时出错: {str(e)}")
                    # 使用一个小的负值作为默认值
                    logprobs_old.append(-10.0)
        else:
            logger.warning("模型输出不包含scores信息，无法计算logprobs_old")
            # 为每个生成的token提供一个默认的log概率
            for _ in generated_ids:
                logprobs_old.append(-10.0)
        
        return {
            "response": generated_text,
            "planner_input_ids": planner_input_ids,
            "planner_output_ids": planner_output_ids,
            "logprobs_old": logprobs_old
        }
      
    def copy_model(self, model_type: str, num_copies: int = 1) -> List[Tuple[Any, Any]]:
        """
        复制模型和分词器的副本，用于并发采样
        
        Args:
            model_type: 要复制的模型类型（通常是"planner"）
            num_copies: 要创建的副本数量
            
        Returns:
            List[Tuple]: 包含(model, tokenizer)元组的列表
        """
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 检查原始模型是否存在
        if model_type not in self._model_cache:
            logger.error(f"模型 {model_type} 不存在，无法复制")
            return []
        
        original_model = self._model_cache[model_type]
        original_tokenizer = self._tokenizer_cache[model_type]
        
        copies = []
        
        for i in range(num_copies):
            # 创建副本的唯一标识
            copy_id = f"{model_type}_copy_{i}"
            
            # 如果副本已存在，直接使用
            if copy_id in self._model_cache:
                logger.info(f"使用已存在的模型副本: {copy_id}")
                copies.append((self._model_cache[copy_id], self._tokenizer_cache[copy_id]))
                continue
            
            # 深度复制模型
            try:
                # 创建模型副本
                model_copy = copy.deepcopy(original_model)
                
                # 确保模型副本在正确的设备上
                device = self.model_config.get("device", "cuda")
                if device == "cuda" and torch.cuda.is_available():
                    model_copy = model_copy.cuda()
                
                # 缓存模型副本
                self._model_cache[copy_id] = model_copy
                self._tokenizer_cache[copy_id] = original_tokenizer  # 分词器不需要复制，可以共享
                
                # 为副本模型创建访问锁
                self._model_locks[copy_id] = threading.Lock()
                self._model_usage[copy_id] = False
                
                copies.append((model_copy, original_tokenizer))
                logger.info(f"成功创建模型副本: {copy_id}")
                
            except Exception as e:
                logger.error(f"创建模型副本 {copy_id} 失败: {str(e)}")
                # 如果复制失败，返回原始模型（但可能导致并发问题）
                copies.append((original_model, original_tokenizer))
                
            # 清理显存
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"清理GPU缓存时出错: {str(e)}")
                
        print(f"当前缓存中有：{len(self._model_cache)}个模型")
        return copies
    
    def release_model_copies(self, model_type: str) -> None:
        """
        释放指定类型的所有模型副本
        
        Args:
            model_type: 要释放副本的原始模型类型
        """
        # 找出所有副本模型
        copies_to_remove = []
        for model_id in self._model_cache.keys():
            if model_id.startswith(f"{model_type}_copy_"):
                copies_to_remove.append(model_id)
        
        logger.info(f"准备释放 {len(copies_to_remove)} 个模型副本")
        
        # 释放副本
        for copy_id in copies_to_remove:
            # 显式删除模型以释放GPU内存
            model = self._model_cache.get(copy_id)
            if model is not None and torch.cuda.is_available():
                try:
                    # 先将模型移到CPU以释放GPU内存
                    if hasattr(model, 'cpu'):
                        model.cpu()
                    del model
                    # 同步CUDA操作，确保所有操作完成后再清理缓存
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    logger.info(f"已释放模型副本: {copy_id}")
                except Exception as e:
                    logger.warning(f"释放模型副本 {copy_id} 时出错: {str(e)}")
                    # 即使出错也尝试清理缓存
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    except:
                        pass
            
            # 从缓存中移除
            self._model_cache.pop(copy_id, None)
            self._tokenizer_cache.pop(copy_id, None)
            self._model_locks.pop(copy_id, None)
            self._model_usage.pop(copy_id, None)
        
        # 最终显存清理
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # 获取当前GPU内存使用情况
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"已清理GPU缓存，当前GPU内存使用: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
            except Exception as e:
                logger.warning(f"清理GPU缓存时出错: {str(e)}")
    
    def clear_cache(self, model_type: Optional[str] = None) -> None:
        """
        清理模型缓存
        
        Args:
            model_type: 要清理的模型类型，如果为None则清理所有缓存
        """
        if model_type is None:
            # 先将所有模型移到CPU以释放GPU内存
            for model_name, model in self._model_cache.items():
                if model is not None and torch.cuda.is_available():
                    try:
                        if hasattr(model, 'cpu'):
                            model.cpu()
                        del model
                        # 同步CUDA操作
                        torch.cuda.synchronize()
                    except Exception as e:
                        logger.warning(f"清理模型 {model_name} 时出错: {str(e)}")
            
            # 清理所有缓存
            self._model_cache.clear()
            self._tokenizer_cache.clear()
            self._model_locks.clear()
            self._model_usage.clear()
            
            # 强制清理GPU缓存
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    # 调用垃圾回收
                    import gc
                    gc.collect()
                except Exception as e:
                    logger.warning(f"清理GPU缓存时出错: {str(e)}")
                
            print("所有模型缓存已清理")
        elif model_type in self._model_cache:
            # 清理指定模型及其副本
            models_to_remove = [model_type]
            # 添加所有副本
            for model_id in self._model_cache.keys():
                if model_id.startswith(f"{model_type}_copy_"):
                    models_to_remove.append(model_id)
            
            # 删除模型及其副本
            for model_id in models_to_remove:
                model = self._model_cache.get(model_id)
                if model is not None and torch.cuda.is_available():
                    try:
                        if hasattr(model, 'cpu'):
                            model.cpu()
                        del model
                        # 同步CUDA操作
                        torch.cuda.synchronize()
                    except Exception as e:
                        logger.warning(f"清理模型 {model_id} 时出错: {str(e)}")
                
                if model_id in self._model_cache:
                    del self._model_cache[model_id]
                if model_id in self._tokenizer_cache:
                    del self._tokenizer_cache[model_id]
                if model_id in self._model_locks:
                    del self._model_locks[model_id]
                if model_id in self._model_usage:
                    del self._model_usage[model_id]
            
            #logger.info(f"已清理模型 {model_type} 及其副本的缓存")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    # 调用垃圾回收
                    import gc
                    gc.collect()
                except Exception as e:
                    logger.warning(f"清理GPU缓存时出错: {str(e)}")
                #logger.info("已清理GPU缓存")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        获取当前内存使用情况
        
        Returns:
            Dict: 包含内存使用信息的字典
        """
        memory_info = {}
        
        if torch.cuda.is_available():
            # 获取GPU内存信息
            memory_info["gpu_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            memory_info["gpu_reserved"] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            memory_info["gpu_max_allocated"] = f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
            
            # 获取GPU总内存
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_info["gpu_total"] = f"{total_memory:.2f} GB"
            
            # 计算可用内存
            allocated = torch.cuda.memory_allocated() / 1024**3
            memory_info["gpu_available"] = f"{total_memory - allocated:.2f} GB"
            
            # 计算内存使用率
            memory_info["gpu_usage_percent"] = f"{(allocated / total_memory) * 100:.1f}%"
            
            # 警告如果内存使用率过高
            if (allocated / total_memory) > 0.9:
                memory_info["warning"] = "GPU内存使用率过高，建议清理缓存"
        
        # 获取缓存中的模型信息
        memory_info["cached_models"] = list(self._model_cache.keys())
        memory_info["model_count"] = len(self._model_cache)
        
        return memory_info
    
    def check_and_manage_memory(self, threshold: float = 0.92) -> bool:
        """
        检查显存使用情况，并在超过阈值时自动清理
        
        Args:
            threshold: 内存使用率阈值，默认为92%
            
        Returns:
            bool: 如果内存使用率低于阈值返回True，否则返回False
        """
        if not torch.cuda.is_available():
            return True
            
        # 获取当前内存使用情况
        allocated = torch.cuda.memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        usage_ratio = allocated / total_memory
        
        # 如果内存使用率超过阈值
        if usage_ratio > threshold:
            logger.warning(f"GPU内存使用率过高 ({usage_ratio*100:.1f}% > {threshold*100:.1f}%)，开始清理缓存")
            
            # 首先尝试清理缓存
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"清理GPU缓存时出错: {str(e)}")
            
            # 强制垃圾回收
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"垃圾回收时出错: {str(e)}")
            
            # 检查清理后的内存使用情况
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            usage_ratio_after = allocated_after / total_memory
            
            # 如果清理后仍然超过阈值，逐个释放模型副本
            if usage_ratio_after > threshold:
                logger.warning("清理缓存后内存使用率仍然过高，开始逐个释放模型副本")
                
                # 找出所有模型副本
                copies_to_remove = []
                for key in self._model_cache:
                    if "_copy_" in key:
                        copies_to_remove.append(key)
                
                # 按副本编号排序，先释放高编号的副本
                copies_to_remove.sort(reverse=True)
                
                # 逐个释放模型副本，每次释放后检查内存使用情况
                memory_freed = False
                for copy_key in copies_to_remove:
                    # 获取当前内存使用情况
                    current_allocated = torch.cuda.memory_allocated() / 1024**3
                    current_usage_ratio = current_allocated / total_memory
                    
                    # 如果内存使用率已经低于阈值，停止释放
                    if current_usage_ratio <= threshold:
                        #logger.info(f"内存使用率已降至阈值以下 ({current_usage_ratio*100:.1f}% <= {threshold*100:.1f}%)，停止释放模型副本")
                        memory_freed = True
                        break
                    
                    # 释放当前副本
                    model = self._model_cache.get(copy_key)
                    if model is not None:
                        try:
                            if hasattr(model, 'cpu'):
                                model.cpu()
                            del model
                            # 同步CUDA操作
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                        except Exception as e:
                            logger.warning(f"释放模型副本 {copy_key} 时出错: {str(e)}")
                    
                    # 从缓存中移除
                    self._model_cache.pop(copy_key, None)
                    self._tokenizer_cache.pop(copy_key, None)
                    self._model_locks.pop(copy_key, None)
                    self._model_usage.pop(copy_key, None)
                    
                    #logger.info(f"已释放模型副本: {copy_key}")
                    
                    # 清理缓存并检查内存使用情况
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        # 强制垃圾回收
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    except Exception as e:
                        logger.warning(f"清理GPU缓存时出错: {str(e)}")
                    
                    # 获取释放后的内存使用情况
                    new_allocated = torch.cuda.memory_allocated() / 1024**3
                    new_usage_ratio = new_allocated / total_memory
                    freed_memory = current_allocated - new_allocated
                    
                    #logger.info(f"释放 {copy_key} 后，内存使用率: {new_usage_ratio*100:.1f}%，释放了 {freed_memory:.2f} GB")
                
                # 最终检查
                allocated_final = torch.cuda.memory_allocated() / 1024**3
                usage_ratio_final = allocated_final / total_memory
                
                if usage_ratio_final > threshold:
                    logger.error(f"释放所有模型副本后，内存使用率仍然过高 ({usage_ratio_final*100:.1f}%)")
                    return False
                else:
                    #logger.info(f"内存管理成功，当前使用率: {usage_ratio_final*100:.1f}%")
                    return True
            else:
                #logger.info(f"缓存清理成功，当前使用率: {usage_ratio_after*100:.1f}%")
                return True
        else:
            return True


# 全局LLM引擎实例
_llm_engine = None


def get_llm_engine(config_path: str = "src/configs/config.yaml") -> LLMEngine:
    """
    获取全局LLM引擎实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        LLMEngine: LLM引擎实例
    """
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine(config_path)
    return _llm_engine


def reset_llm_engine():
    """重置全局LLM引擎实例"""
    global _llm_engine
    if _llm_engine is not None:
        _llm_engine.clear_cache()
    _llm_engine = None
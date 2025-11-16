"""
Model loading utilities for AgentFlow
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any


def load_model(model_path: str,
               device: str = "cuda",
               torch_dtype: str = "float16",
               trust_remote_code: bool = True) -> tuple:
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径
        device: 设备
        torch_dtype: 数据类型
        trust_remote_code: 是否信任远程代码
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 模型加载参数
    model_kwargs = {
        "torch_dtype": getattr(torch, torch_dtype),
        "device_map": "auto" if device == "cuda" else None,
        "trust_remote_code": trust_remote_code
    }
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print(f"Model dtype: {model.dtype}")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    return model, tokenizer


def get_model_size(model) -> Dict[str, Any]:
    """获取模型大小信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_gb": total_params * 4 / (1024**3),  # 假设float32
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0
    }


def check_gpu_memory() -> Dict[str, float]:
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "cached_gb": torch.cuda.memory_reserved() / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    else:
        return {"error": "CUDA not available"}
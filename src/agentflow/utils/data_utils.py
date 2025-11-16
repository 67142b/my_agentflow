"""
Data utilities for AgentFlow
"""

import pandas as pd
from datasets import Dataset
from typing import List, Dict, Any, Optional
import numpy as np


def load_dataset(data_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    加载数据集
    
    Args:
        data_path: 数据文件路径
        max_samples: 最大样本数量
        
    Returns:
        List[Dict]: 数据列表
    """
    try:
        # 尝试加载parquet文件
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        # 尝试加载json文件
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # 转换为字典列表
        data = df.to_dict('records')
        
        # 采样
        if max_samples and len(data) > max_samples:
            indices = np.random.choice(len(data), max_samples, replace=False)
            data = [data[i] for i in indices]
        
        print(f"Loaded {len(data)} samples from {data_path}")
        return data
        
    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}")
        return []


def sample_dataset(data: List[Dict[str, Any]], 
                   num_samples: int, 
                   seed: int = 42) -> List[Dict[str, Any]]:
    """
    采样数据集
    
    Args:
        data: 原始数据
        num_samples: 采样数量
        seed: 随机种子
        
    Returns:
        List[Dict]: 采样后的数据
    """
    if len(data) <= num_samples:
        return data
    
    np.random.seed(seed)
    indices = np.random.choice(len(data), num_samples, replace=False)
    return [data[i] for i in indices]


def preprocess_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    预处理数据
    
    Args:
        data: 原始数据
        
    Returns:
        List[Dict]: 预处理后的数据
    """
    processed_data = []
    
    for item in data:
        # 确保必要字段存在
        if "question" not in item or "result" not in item:
            continue
        
        # 清理数据
        processed_item = {
            "question": str(item["question"]).strip(),
            "result": str(item["result"]).strip(),
            "id": item.get("id", len(processed_data)),
            "source": item.get("source", "unknown")
        }
        
        # 跳过空数据
        if not processed_item["question"] or not processed_item["result"]:
            continue
        
        processed_data.append(processed_item)
    
    return processed_data


def split_data(data: List[Dict[str, Any]], 
               train_ratio: float = 0.8,
               seed: int = 42) -> tuple:
    """
    分割训练和验证数据
    
    Args:
        data: 数据列表
        train_ratio: 训练集比例
        seed: 随机种子
        
    Returns:
        tuple: (train_data, val_data)
    """
    np.random.seed(seed)
    np.random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data


def create_data_batches(data: List[Dict[str, Any]], 
                       batch_size: int) -> List[List[Dict[str, Any]]]:
    """
    创建数据批次
    
    Args:
        data: 数据列表
        batch_size: 批次大小
        
    Returns:
        List[List]: 批次数据列表
    """
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        if len(batch) == batch_size:  # 只保留完整批次
            batches.append(batch)
    
    return batches
#!/usr/bin/env python3
"""
训练流程测试脚本
从训练集中采样一个问题，验证一个问题的完整训练流程
除了数据准备阶段不同，其它流程与现有项目中的训练脚本保持一致
"""

import os
import sys
import json
import yaml
import logging
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List, Optional
import torch

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agentflow.simple_agent import SimpleAgent
from src.agentflow.training.enhanced_trainer import EnhancedTrainer


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("test_training.log")
        ]
    )
    return logging.getLogger("TestTraining")


def load_training_samples(config_path: str, num_samples: int = 10) -> List[Dict[str, Any]]:
    """
    从训练集中加载多个样本
    
    Args:
        config_path: 配置文件路径
        num_samples: 要加载的样本数量
        
    Returns:
        多个训练样本的列表
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取训练数据路径
    train_path = config.get('data', {}).get('train_path', 'src/data/train/combined_train.parquet')
    
    # 加载训练数据
    train_df = pd.read_parquet(train_path)
    
    # 检查数据集是否为空
    if len(train_df) == 0:
        raise ValueError("训练数据集为空")
    
    # 随机采样指定数量的样本
    samples_df = train_df.sample(n=min(num_samples, len(train_df)), random_state=42)
    
    samples = []
    for _, row in samples_df.iterrows():
        question = row.get('question', '')
        ground_truth = row.get('result', row.get('answer', ''))
        
        if not question or not ground_truth:
            continue
            
        sample = {
            "question": question,
            "ground_truth": ground_truth
        }
        samples.append(sample)
    
    print(f"加载了 {len(samples)} 个训练样本")
    return samples


def test_multiple_samples_training():
    """测试多个样本的完整训练流程"""
    # 设置日志
    logger = setup_logging()
    #logger.info("开始多个样本训练流程测试")
    
    # 配置路径
    config_path = "src/configs/config.yaml"
    
    # 加载多个训练样本
    #logger.info("加载多个训练样本...")
    train_samples = load_training_samples(config_path, num_samples=1)
    #logger.info(f"加载了 {len(train_samples)} 个训练样本")
    
    # 创建SimpleAgent
    #logger.info("初始化SimpleAgent...")
    agent = SimpleAgent(config_path=config_path)
    
    # 准备训练数据（多个样本）
    train_data = train_samples
    val_data = []  # 测试不需要验证数据
    
    # 创建训练器
    #logger.info("创建训练器...")
    trainer = EnhancedTrainer(
        agent=agent,
        train_data=train_data,
        val_data=val_data,
        config_path=config_path
    )
    
    # 设置测试模式：只训练一个epoch，只处理一个批次
    original_max_epochs = trainer.flow_grpo_trainer.config.max_epochs
    original_batch_size = trainer.flow_grpo_trainer.config.batch_size
    
    # 修改训练参数以适应单样本测试
    trainer.flow_grpo_trainer.config.max_epochs = 1  # 只训练一个epoch
    trainer.flow_grpo_trainer.config.batch_size = 1  # 批次大小为1
    
    #logger.info(f"修改训练参数: max_epochs={trainer.flow_grpo_trainer.config.max_epochs}, batch_size={trainer.flow_grpo_trainer.config.batch_size}")
    
    # 创建临时保存目录
    save_dir = "test_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 执行训练
        #logger.info("开始训练...")
        training_history = trainer.train(save_dir=save_dir)
        
        # 保存训练历史
        history_path = os.path.join(save_dir, "test_training_history.json")
        trainer.save_training_history(history_path)
        
        #logger.info("训练完成!")
        #logger.info(f"训练历史已保存到: {history_path}")
        
        # 打印训练结果摘要
        if "train_loss" in training_history and training_history["train_loss"]:
            #logger.info(f"最终训练损失: {training_history['train_loss'][-1]}")
            pass
        
        if "train_reward" in training_history and training_history["train_reward"]:
            #logger.info(f"最终训练奖励: {training_history['train_reward'][-1]}")
            pass
            
        if "train_success_rate" in training_history and training_history["train_success_rate"]:
            #logger.info(f"最终成功率: {training_history['train_success_rate'][-1]}")
            pass
            
        return True
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
        
    finally:
        # 恢复原始训练参数
        trainer.flow_grpo_trainer.config.max_epochs = original_max_epochs
        trainer.flow_grpo_trainer.config.batch_size = original_batch_size
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            #logger.info(f"GPU内存已清理，当前使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")


def main():
    """主函数"""
    print("=" * 60)
    print("训练流程测试脚本")
    print("从训练集中采样多个问题，验证多个样本的完整训练流程")
    print("=" * 60)
    
    success = test_multiple_samples_training()
    
    if success:
        print("\n✅ 测试成功完成!")
        print("训练流程测试通过，可以安全地进行完整训练。")
    else:
        print("\n❌ 测试失败!")
        print("请检查错误日志并修复问题后再进行完整训练。")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
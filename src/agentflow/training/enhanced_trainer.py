"""
训练器模块，整合Flow-Group训练算法与数据加载
"""

import os
import sys
import json
import yaml
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agentflow.simple_agent import SimpleAgent
from src.agentflow.training.flow_grpo import FlowGRPO, create_flow_grpo_trainer
from src.agentflow.utils.logger import TrainingLogger, create_logger_from_config
from src.agentflow.evaluation import ModelEvaluator, create_evaluator_from_config


class EnhancedTrainer:
    """增强型训练器，整合Flow-Group训练算法与数据加载"""
    
    def __init__(self, agent: SimpleAgent, train_data: List[Dict[str, Any]], 
                 val_data: Optional[List[Dict[str, Any]]] = None, config_path: str = "src/configs/config.yaml"):
        """
        初始化增强训练器
        
        Args:
            agent: SimpleAgent实例
            train_data: 训练数据
            val_data: 验证数据
            config_path: 配置文件路径
        """
        self.agent = agent
        self.train_data = train_data
        self.val_data = val_data
        self.config_path = config_path
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 创建Flow-GRPO训练器
        self.flow_grpo_trainer = create_flow_grpo_trainer(agent, self.config)
        
        # 初始化评估器
        self.evaluator = None
        if self.config.get("evaluation", {}).get("enable_evaluation", False):
            from src.agentflow.evaluation.evaluator import create_evaluator_from_config
            self.evaluator = create_evaluator_from_config(config=self.config,agent=self.agent)
        
        # 训练历史记录
        self.training_history = {
            "train_loss": [],
            "train_reward": [],
            "train_success_rate": [],
            "val_loss": [],
            "val_reward": [],
            "val_success_rate": []
        }
    
    def _create_evaluator(self) -> ModelEvaluator:
        """创建评估器"""
        return create_evaluator_from_config(self.config, self.agent, self.logger)
    
    def evaluate(self, eval_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            eval_data: 评估数据，如果为None则使用验证数据
            
        Returns:
            评估指标字典
        """
        if eval_data is None:
            eval_data = self.val_data
        
        if eval_data is None or len(eval_data) == 0:
            return {}
        
        return self.evaluator.evaluate(eval_data)
    
    def train(self, save_dir: str = "checkpoints") -> Dict[str, List[float]]:
        """
        执行训练
        
        Args:
            save_dir: 模型保存目录
            
        Returns:
            训练历史记录
        """
        # 执行Flow-GRPO训练
        self.flow_grpo_trainer.train(
            train_data=self.train_data,
            val_data=self.val_data,
            save_dir=save_dir
        )
        
        # 收集训练历史
        self.training_history["train_loss"] = self.flow_grpo_trainer.train_stats["loss"]
        self.training_history["train_reward"] = self.flow_grpo_trainer.train_stats["reward"]
        self.training_history["train_success_rate"] = self.flow_grpo_trainer.train_stats["success_rate"]
        
        # 评估模型
        if self.config.get("evaluation", {}).get("enable_evaluation", False):
            eval_results = self.evaluate()
            
            if eval_results:
                # 与基线模型比较
                if (self.config.get("evaluation", {}).get("compare_with_baseline", False) and 
                    self.config.get("evaluation", {}).get("baseline_model_path")):
                    comparison_results = self.evaluator.compare_models(
                        self.config["evaluation"]["baseline_model_path"],
                        os.path.join(save_dir, "final_model.pt"),
                        self.val_data
                    )
        
        return self.training_history
    
    def save_training_history(self, save_path: str) -> None:
        """
        保存训练历史
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_training_history(self, load_path: str) -> None:
        """
        加载训练历史
        
        Args:
            load_path: 加载路径
        """
        with open(load_path, 'r') as f:
            self.training_history = json.load(f)


def prepare_data_from_config(config_path: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    根据配置文件准备训练和验证数据
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        训练数据和验证数据
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取数据路径
    train_path = config.get('data', {}).get('train_path', 'src/data/train/combined_train.parquet')
    val_path = config.get('data', {}).get('val_path', 'src/data/val/aime24.parquet')
    
    # 获取采样参数
    max_train_samples = config.get('data', {}).get('max_train_samples', 100)
    max_val_samples = config.get('data', {}).get('max_val_samples', 10)
    
    # 加载训练数据
    train_df = pd.read_parquet(train_path)
    train_data = []
    
    # 限制样本数量
    train_df = train_df.head(max_train_samples)
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing training data"):
        # 确保有必要的字段
        question = row.get('question', '')
        ground_truth = row.get('result', row.get('answer', ''))
        
        if question and ground_truth:
            train_data.append({
                "question": question,
                "ground_truth": ground_truth
            })
    
    # 加载验证数据
    val_df = pd.read_parquet(val_path)
    val_data = []
    
    # 限制样本数量
    val_df = val_df.head(max_val_samples)
    
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Processing validation data"):
        # 确保有必要的字段
        question = row.get('question', '')
        ground_truth = row.get('result', row.get('answer', ''))
        
        if question and ground_truth:
            val_data.append({
                "question": question,
                "ground_truth": ground_truth
            })
    
    return train_data, val_data


def create_trainer_from_config(config_path: str) -> EnhancedTrainer:
    """
    根据配置文件创建训练器
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        EnhancedTrainer实例
    """
    # 创建SimpleAgent
    agent = SimpleAgent(config_path=config_path)
    
    # 准备数据
    train_data, val_data = prepare_data_from_config(config_path)
    
    # 创建训练器
    trainer = EnhancedTrainer(
        agent=agent,
        train_data=train_data,
        val_data=val_data,
        config_path=config_path
    )
    
    return trainer


def main():
    """主函数，用于直接运行训练"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AgentFlow with Flow-Group algorithm")
    parser.add_argument("--config", type=str, default="src/configs/config.yaml", help="Path to config file")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = create_trainer_from_config(args.config)
    
    # 恢复训练
    if args.resume:
        trainer.flow_grpo_trainer.load_checkpoint(args.resume)
    
    # 开始训练
    training_history = trainer.train(save_dir=args.save_dir)
    
    # 保存训练历史
    history_path = os.path.join(args.save_dir, "training_history.json")
    trainer.save_training_history(history_path)


if __name__ == "__main__":
    main()
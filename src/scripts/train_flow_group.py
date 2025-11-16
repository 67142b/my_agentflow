"""
训练脚本，用于启动Flow-Group训练算法
"""
'''
python src/scripts/train_flow_group.py --config src/configs/config.yaml --save_dir ./checkpoints
'''
import os
import sys
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agentflow.training.enhanced_trainer import create_trainer_from_config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train AgentFlow with Flow-Group algorithm")
    parser.add_argument("--config", type=str, default="src/configs/config.yaml", help="Path to config file")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建训练器
    trainer = create_trainer_from_config(args.config)
    
    try:
        # 恢复训练
        if args.resume:
            trainer.flow_grpo_trainer.load_checkpoint(args.resume)
        
        # 仅评估模式
        if args.eval_only:
            eval_stats = trainer.evaluate()
            return
        
        # 开始训练
        training_history = trainer.train(save_dir=args.save_dir)
        
        # 保存训练历史
        history_path = os.path.join(args.save_dir, "training_history.json")
        trainer.save_training_history(history_path)
        
    except Exception as e:
        # 记录错误
        raise


if __name__ == "__main__":
    main()
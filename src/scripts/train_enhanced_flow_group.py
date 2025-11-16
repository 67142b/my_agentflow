"""
增强版训练脚本，用于启动增强版Flow-Group训练算法
实现多轮/分词级/组归一化的策略优化目标
"""
'''
python src/scripts/train_enhanced_flow_group.py --config src/configs/config.yaml --save_dir ./checkpoints
'''
import os
import sys
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agentflow.simple_agent import SimpleAgent
from src.agentflow.training.enhanced_flow_grpo import create_enhanced_flow_grpo_trainer
from src.agentflow.utils.logger import create_logger_from_config


def prepare_data_from_config(config):
    """
    从配置准备训练和验证数据
    
    Args:
        config: 配置字典
        
    Returns:
        训练数据和验证数据
    """
    data_config = config.get('data', {})
    train_path = data_config.get('train_data_path')
    val_path = data_config.get('val_data_path')
    
    max_train_samples = data_config.get('max_train_samples', 1000)
    max_val_samples = data_config.get('max_val_samples', 100)
    
    train_data = []
    val_data = []
    
    # 加载训练数据
    if train_path and os.path.exists(train_path):
        if train_path.endswith('.parquet'):
            import pandas as pd
            df = pd.read_parquet(train_path)
            # 转换为列表格式
            for i, row in df.iterrows():
                if i >= max_train_samples:
                    break
                item = {
                    "question": row.get("question", ""),
                    "ground_truth": row.get("ground_truth", row.get("answer", ""))
                }
                train_data.append(item)
        else:
            with open(train_path, 'r', encoding='utf-8') as f:
                import json
                for i, line in enumerate(f):
                    if i >= max_train_samples:
                        break
                    if line.strip():
                        item = json.loads(line.strip())
                        train_data.append(item)
    
    # 加载验证数据
    if val_path and os.path.exists(val_path):
        if val_path.endswith('.parquet'):
            import pandas as pd
            df = pd.read_parquet(val_path)
            # 转换为列表格式
            for i, row in df.iterrows():
                if i >= max_val_samples:
                    break
                item = {
                    "question": row.get("question", ""),
                    "ground_truth": row.get("ground_truth", row.get("answer", ""))
                }
                val_data.append(item)
        else:
            with open(val_path, 'r', encoding='utf-8') as f:
                import json
                for i, line in enumerate(f):
                    if i >= max_val_samples:
                        break
                    if line.strip():
                        item = json.loads(line.strip())
                        val_data.append(item)
    
    return train_data, val_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train AgentFlow with Enhanced Flow-Group algorithm")
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
    
    # 创建日志记录器
    logger = create_logger_from_config(config)
    
    # 创建SimpleAgent实例
    agent = SimpleAgent(config)
    
    # 创建增强版Flow-GRPO训练器
    trainer = create_enhanced_flow_grpo_trainer(agent, config)
    
    # 准备数据
    train_data, val_data = prepare_data_from_config(config)
    #logger.info(f"加载训练数据: {len(train_data)} 条")
    #logger.info(f"加载验证数据: {len(val_data)} 条")
    
    try:
        # 恢复训练
        if args.resume:
            trainer.load_checkpoint(args.resume)
            #logger.info(f"从检查点恢复训练: {args.resume}")
        
        # 仅评估模式
        if args.eval_only:
            eval_stats = trainer.evaluate(val_data)
            #logger.info(f"评估结果: {eval_stats}")
            return
        
        # 开始训练
        #logger.info("开始增强版Flow-Group训练...")
        training_history = trainer.train(train_data, val_data, args.save_dir)
        
        # 保存训练历史
        history_path = os.path.join(args.save_dir, "enhanced_training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(training_history, f, indent=2, ensure_ascii=False)
        
        #logger.info(f"训练完成，结果保存在: {args.save_dir}")
        
    except Exception as e:
        # 记录错误
        logger.error(f"训练过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
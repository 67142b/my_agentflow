#!/usr/bin/env python3
"""
模型评估脚本

用于评估训练好的模型性能，支持单个模型评估和模型比较。
"""

import os
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agentflow.simple_agent import SimpleAgent
from src.agentflow.evaluation import ModelEvaluator
from src.agentflow.utils.logger import create_logger_from_config
from src.data.get_train_data import load_data


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估模型性能")
    
    # 必需参数
    parser.add_argument("--config", type=str, required=True,
                        help="配置文件路径")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径")
    
    # 可选参数
    parser.add_argument("--data", type=str, default=None,
                        help="评估数据路径，如果不指定则使用配置中的验证数据")
    parser.add_argument("--output", type=str, default="./eval_results",
                        help="评估结果输出目录")
    parser.add_argument("--compare", type=str, default=None,
                        help="要比较的另一个模型路径")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="评估批次大小")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="评估样本数量，如果不指定则使用全部数据")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载配置
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建日志记录器
    logger = create_logger_from_config(config)
    logger.log_start()
    
    # 创建智能体
    agent = SimpleAgent(config_path=args.config)
    
    # 加载模型
    print(f"Loading model from: {args.model}")
    agent.load_model(args.model)
    
    # 加载评估数据
    if args.data:
        print(f"Loading evaluation data from: {args.data}")
        eval_data = load_data(args.data)
    else:
        print("Using validation data from config")
        eval_data = load_data(config["data"]["val_path"])
    
    # 限制评估样本数量
    if args.sample_size and args.sample_size < len(eval_data):
        print(f"Limiting evaluation to {args.sample_size} samples")
        eval_data = eval_data[:args.sample_size]
    
    # 创建评估器
    evaluator = ModelEvaluator(
        agent=agent,
        logger=logger,
        save_dir=args.output
    )
    
    # 评估单个模型
    print(f"Evaluating model: {args.model}")
    eval_results = evaluator.evaluate(eval_data)
    
    # 如果指定了比较模型，则进行比较
    if args.compare:
        print(f"\nComparing with model: {args.compare}")
        comparison_results = evaluator.compare_models(
            args.compare, 
            args.model, 
            eval_data
        )
    
    # 记录评估结束
    logger.log_end({"evaluation_results": eval_results})
    
    print(f"Evaluation completed! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
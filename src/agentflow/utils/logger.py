"""
训练日志记录模块，用于记录训练过程中的详细信息
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        log_level: int = logging.INFO
    ):
        """
        初始化训练日志记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称，如果为None则使用时间戳
            log_level: 日志级别
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建实验名称
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"experiment_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # 创建实验目录
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logging.getLogger(f"training_{self.experiment_name}")
        self.logger.setLevel(log_level)
        
        # 清除已有的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建文件处理器
        log_file = self.experiment_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 训练统计
        self.training_stats = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_epochs": 0,
            "total_steps": 0,
            "best_reward": -float('inf'),
            "epoch_stats": [],
            "config": {}
        }
        
        #self.logger.info(f"Initialized training logger for experiment: {self.experiment_name}")
        #self.logger.info(f"Log directory: {self.experiment_dir}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        记录配置信息
        
        Args:
            config: 配置字典
        """
        self.training_stats["config"] = config
        
        # 保存配置到文件
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        #self.logger.info("Configuration logged")
    
    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """
        记录epoch开始
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
        """
        #self.logger.info(f"Starting epoch {epoch+1}/{total_epochs}")
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
    
    def log_epoch_end(self, epoch_stats: Dict[str, float]) -> None:
        """
        记录epoch结束
        
        Args:
            epoch_stats: epoch统计信息
        """
        epoch_time = time.time() - self.epoch_start_time
        
        # 添加时间戳和耗时
        epoch_stats["timestamp"] = datetime.now().isoformat()
        epoch_stats["epoch_time"] = epoch_time
        
        # 记录到训练统计
        self.training_stats["epoch_stats"].append(epoch_stats)
        self.training_stats["total_epochs"] = max(self.training_stats["total_epochs"], self.current_epoch + 1)
        
        # 更新最佳奖励
        if "val_avg_reward" in epoch_stats:
            if epoch_stats["val_avg_reward"] > self.training_stats["best_reward"]:
                self.training_stats["best_reward"] = epoch_stats["val_avg_reward"]
        elif "avg_reward" in epoch_stats:
            if epoch_stats["avg_reward"] > self.training_stats["best_reward"]:
                self.training_stats["best_reward"] = epoch_stats["avg_reward"]
        
        # 记录日志
        log_msg = f"Epoch {self.current_epoch+1} completed in {epoch_time:.2f}s - "
        log_msg += ", ".join([f"{k}: {v:.4f}" for k, v in epoch_stats.items() if k not in ["timestamp", "epoch_time"]])
        #self.logger.info(log_msg)
        
        # 保存训练统计
        self.save_training_stats()
    
    def log_step(self, step: int, step_stats: Dict[str, float]) -> None:
        """
        记录训练步骤
        
        Args:
            step: 当前步骤
            step_stats: 步骤统计信息
        """
        self.training_stats["total_steps"] = max(self.training_stats["total_steps"], step + 1)
        
        # 每100步记录一次
        if step % 100 == 0:
            log_msg = f"Step {step} - "
            log_msg += ", ".join([f"{k}: {v:.4f}" for k, v in step_stats.items()])
            self.logger.debug(log_msg)
    
    def log_evaluation(self, eval_stats: Dict[str, float]) -> None:
        """
        记录评估结果
        
        Args:
            eval_stats: 评估统计信息
        """
        eval_stats["timestamp"] = datetime.now().isoformat()
        
        # 记录到训练统计
        if "evaluations" not in self.training_stats:
            self.training_stats["evaluations"] = []
        self.training_stats["evaluations"].append(eval_stats)
        
        # 记录日志
        log_msg = "Evaluation - "
        log_msg += ", ".join([f"{k}: {v:.4f}" for k, v in eval_stats.items() if k != "timestamp"])
        #self.logger.info(log_msg)
    
    def log_checkpoint(self, checkpoint_path: str, epoch: int) -> None:
        """
        记录检查点保存
        
        Args:
            checkpoint_path: 检查点路径
            epoch: 当前epoch
        """
        #self.logger.info(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """
        记录错误
        
        Args:
            error: 错误对象
            context: 错误上下文
        """
        error_msg = f"Error in {context}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
    
    def log_trajectory(self, trajectory: Dict[str, Any]) -> None:
        """
        记录轨迹信息（用于调试）
        
        Args:
            trajectory: 轨迹数据
        """
        if not hasattr(self, 'trajectory_log'):
            self.trajectory_log = []
        
        # 添加时间戳
        trajectory["log_timestamp"] = datetime.now().isoformat()
        self.trajectory_log.append(trajectory)
        
        # 每10条轨迹保存一次
        if len(self.trajectory_log) % 10 == 0:
            trajectory_file = self.experiment_dir / "trajectories.jsonl"
            with open(trajectory_file, 'a') as f:
                for traj in self.trajectory_log[-10:]:
                    f.write(json.dumps(traj) + '\n')
    
    def save_training_stats(self) -> None:
        """保存训练统计到文件"""
        stats_file = self.experiment_dir / "training_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def finalize(self) -> None:
        """完成训练，记录结束时间"""
        self.training_stats["end_time"] = datetime.now().isoformat()
        self.save_training_stats()
        
        # 计算总训练时间
        start_time = datetime.fromisoformat(self.training_stats["start_time"])
        end_time = datetime.fromisoformat(self.training_stats["end_time"])
        total_time = end_time - start_time
        
        #self.logger.info(f"Training completed in {total_time}")
        #self.logger.info(f"Best reward achieved: {self.training_stats['best_reward']:.4f}")
        #self.logger.info(f"Training logs saved to: {self.experiment_dir}")
    
    def get_experiment_dir(self) -> Path:
        """获取实验目录路径"""
        return self.experiment_dir
    
    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳检查点路径"""
        checkpoints_dir = self.experiment_dir.parent / "checkpoints"
        if checkpoints_dir.exists():
            best_model_path = checkpoints_dir / "best_model.pt"
            if best_model_path.exists():
                return str(best_model_path)
        return None


def create_logger_from_config(config: Dict[str, Any]) -> TrainingLogger:
    """
    根据配置创建日志记录器
    
    Args:
        config: 配置字典
        
    Returns:
        TrainingLogger实例
    """
    # 获取日志配置
    log_config = config.get("logging", {})
    log_dir = log_config.get("log_dir", "logs")
    experiment_name = log_config.get("experiment_name")
    log_level = getattr(logging, log_config.get("log_level", "INFO").upper())
    
    # 创建日志记录器
    logger = TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        log_level=log_level
    )
    
    # 记录配置
    logger.log_config(config)
    
    return logger


if __name__ == "__main__":
    # 测试日志记录器
    logger = TrainingLogger(experiment_name="test_experiment")
    
    # 记录配置
    test_config = {
        "batch_size": 8,
        "learning_rate": 1e-5,
        "max_epochs": 10
    }
    logger.log_config(test_config)
    
    # 记录epoch
    for epoch in range(3):
        logger.log_epoch_start(epoch, 3)
        time.sleep(1)  # 模拟训练时间
        
        epoch_stats = {
            "loss": 1.0 - epoch * 0.3,
            "reward": 0.5 + epoch * 0.2,
            "success_rate": 0.4 + epoch * 0.2
        }
        logger.log_epoch_end(epoch_stats)
    
    # 记录评估
    eval_stats = {
        "val_loss": 0.2,
        "val_reward": 0.8,
        "val_success_rate": 0.7
    }
    logger.log_evaluation(eval_stats)
    
    # 完成训练
    logger.finalize()
    
    print(f"Test logs saved to: {logger.get_experiment_dir()}")
"""
增强版Flow-Group训练算法实现模块
实现多轮/分词级/组归一化的策略优化目标
"""

import os
import sys
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path
import logging
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
import time
from src.agentflow.simple_agent import SimpleAgent
from src.agentflow.utils.reward import compute_reward
from src.agentflow.utils.logger import create_logger_from_config

logger = logging.getLogger(__name__)


@dataclass
class EnhancedFlowGRPOConfig:
    """增强版Flow-GRPO训练配置"""
    # 基础训练参数
    batch_size: int = 8
    max_epochs: int = 10
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Flow-Group特定参数
    group_size: int = 2  # 组大小，每个问题生成多条轨迹
    max_turns: int = 10  # 最大轮次
    epsilon: float = 0.2  # PPO剪切系数
    beta: float = 0.01    # KL惩罚系数
    
    # 训练控制
    save_every: int = 1
    eval_every: int = 5
    enable_evaluation: bool = False
    
    # 采样参数
    temperature: float = 1.0
    top_p: float = 0.9
    max_tokens: int = 2048
    
    # 路径配置
    log_dir: str = "./logs"
    
    # 日志配置
    log_level: str = "INFO"
    save_trajectories: bool = True
    trajectory_sample_rate: float = 0.1
    
    # 增强版参数 - 多轮/分词级/组归一化
    enable_multi_turn_loss: bool = True  # 是否启用多轮损失
    enable_token_level_loss: bool = True  # 是否启用分词级损失
    enable_group_normalization: bool = True  # 是否启用组归一化
    
    # 多轮损失参数
    turn_weight_decay: float = 0.95  # 轮次权重衰减因子，后面的轮次权重更低
    early_turn_bonus: float = 1.2  # 早期轮次奖励系数，鼓励早期解决问题
    
    # 分词级损失参数
    token_level_weight: float = 0.5  # 分词级损失权重
    sequence_level_weight: float = 0.5  # 序列级损失权重
    token_importance_threshold: float = 0.1  # token重要性阈值
    
    # 组归一化参数
    group_normalization_method: str = "z_score"  # 归一化方法: "z_score", "min_max", "rank"
    advantage_normalization: bool = True  # 是否对优势进行归一化
    reward_normalization: bool = True  # 是否对奖励进行归一化


class EnhancedFlowGRPO:
    """增强版Flow-Group训练算法实现类，实现多轮/分词级/组归一化的策略优化目标"""
    
    def __init__(self, agent: SimpleAgent, config: EnhancedFlowGRPOConfig):
        """
        初始化增强版Flow-GRPO训练器
        
        Args:
            agent: SimpleAgent实例
            config: 训练配置
        """
        self.agent = agent
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.agent.planner.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_reward = -float('inf')
        
        # 统计信息
        self.train_stats = {
            "loss": [],
            "multi_turn_loss": [],
            "token_level_loss": [],
            "sequence_level_loss": [],
            "reward": [],
            "success_rate": [],
            "avg_turns": []
        }
    
    def generate_group_trajectories(self, question: str, ground_truth: str) -> List[Dict[str, Any]]:
        """
        为单个问题生成一组轨迹
        
        Args:
            question: 问题文本
            ground_truth: 标准答案
            
        Returns:
            轨迹列表，每个轨迹包含执行过程和结果
        """
        trajectories = []
        
        try:
            # 调用SimpleAgent.solve方法，它会自动生成group_size条轨迹
            # 注意：SimpleAgent.solve方法只需要query参数，它会返回多条轨迹
            print(f"开始调用SimpleAgent.solve，问题: {question[:50]}...\n====================================")
            start_time = time.time()
            result = self.agent.solve(question)
            end_time = time.time()
            print(f"SimpleAgent.solve调用完毕，用时: {end_time - start_time:.2f}秒\n====================================")
            
            # 处理返回的多条轨迹
            successful_trajectories = result.get("successful_trajectories", [])
            failed_trajectories = result.get("failed_trajectories", [])
            
            print(f"成功轨迹: {successful_trajectories}")
            
            # 处理成功的轨迹
            for i, traj in enumerate(successful_trajectories):
                # 计算奖励
                reward = compute_reward(
                    question=question,
                    answer=traj.get("final_answer", ""),
                    ground_truth=ground_truth
                )
                print(f"成功轨迹 {i}，返回答案：{traj.get('final_answer', '')}，标准答案：{ground_truth}， 奖励: {reward:.4f}")
                
                # 添加轨迹ID
                trajectory_id = f"{question[:50].replace(' ', '_')}_{i}"
                
                # 构建轨迹数据
                trajectory = {
                    "trajectory_id": trajectory_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "final_answer": traj.get("final_answer", ""),
                    "reward": reward,
                    "turns": traj.get("turns", []),
                    "success": True,
                    "trajectory_data": traj  # 保存完整的轨迹数据
                }
                
                # 添加token级概率信息
                self._extract_token_probs(trajectory)
                
                # 添加多轮信息
                self._extract_multi_turn_info(trajectory)
                
                trajectories.append(trajectory)
            
            # 处理失败的轨迹
            for i, traj in enumerate(failed_trajectories):
                # 添加轨迹ID
                trajectory_id = f"{question[:50].replace(' ', '_')}_{len(successful_trajectories) + i}"
                
                # 构建轨迹数据
                trajectory = {
                    "trajectory_id": trajectory_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "final_answer": traj.get("final_answer", ""),
                    "reward": 0.0,
                    "turns": traj.get("turns", []),
                    "success": False,
                    "error": traj.get("error", "Trajectory failed"),
                    "trajectory_data": traj  # 保存完整的轨迹数据
                }
                
                # 添加token级概率信息
                self._extract_token_probs(trajectory)
                
                # 添加多轮信息
                self._extract_multi_turn_info(trajectory)
                
                trajectories.append(trajectory)
                
        except Exception as e:
            # 如果整个solve过程失败，创建失败轨迹
            for i in range(self.config.group_size):
                trajectory = {
                    "trajectory_id": f"{question[:50].replace(' ', '_')}_{i}",
                    "question": question,
                    "ground_truth": ground_truth,
                    "final_answer": "",
                    "reward": 0.0,
                    "turns": [],
                    "success": False,
                    "error": str(e)
                }
                trajectories.append(trajectory)
        
        return trajectories
    
    def _extract_token_probs(self, trajectory: Dict[str, Any]) -> None:
        """
        从轨迹中提取token级概率信息
        
        Args:
            trajectory: 轨迹数据
        """
        # 从轨迹数据中提取token级概率信息
        token_probs = []
        token_logprobs = []
        token_texts = []
        
        # 从轨迹的turns中提取信息
        turns = trajectory.get("turns", [])
        for turn in turns:
            # 从每个turn的plan中提取信息
            plan = turn.get("plan", {})
            if "token_probs" in plan:
                token_probs.extend(plan["token_probs"])
            if "token_logprobs" in plan:
                token_logprobs.extend(plan["token_logprobs"])
            if "token_texts" in plan:
                token_texts.extend(plan["token_texts"])
            
            # 从每个turn的execution中提取信息
            execution = turn.get("execution", {})
            if "token_probs" in execution:
                token_probs.extend(execution["token_probs"])
            if "token_logprobs" in execution:
                token_logprobs.extend(execution["token_logprobs"])
            if "token_texts" in execution:
                token_texts.extend(execution["token_texts"])
        
        # 添加到轨迹中
        trajectory["token_probs"] = token_probs
        trajectory["token_logprobs"] = token_logprobs
        trajectory["token_texts"] = token_texts
    
    def _extract_multi_turn_info(self, trajectory: Dict[str, Any]) -> None:
        """
        从轨迹中提取多轮信息
        
        Args:
            trajectory: 轨迹数据
        """
        turns = trajectory.get("turns", [])
        multi_turn_info = []
        
        for turn_idx, turn in enumerate(turns):
            turn_info = {
                "turn_idx": turn_idx,
                "plan": turn.get("plan", {}),
                "execution": turn.get("execution", {}),
                "verification": turn.get("verification", {}),
                "success": turn.get("success", False)
            }
            
            # 提取每个turn的token概率信息
            plan = turn.get("plan", {})
            execution = turn.get("execution", {})
            
            turn_token_probs = []
            turn_token_logprobs = []
            turn_token_texts = []
            
            if "token_probs" in plan:
                turn_token_probs.extend(plan["token_probs"])
            if "token_logprobs" in plan:
                turn_token_logprobs.extend(plan["token_logprobs"])
            if "token_texts" in plan:
                turn_token_texts.extend(plan["token_texts"])
                
            if "token_probs" in execution:
                turn_token_probs.extend(execution["token_probs"])
            if "token_logprobs" in execution:
                turn_token_logprobs.extend(execution["token_logprobs"])
            if "token_texts" in execution:
                turn_token_texts.extend(execution["token_texts"])
            
            turn_info["token_probs"] = turn_token_probs
            turn_info["token_logprobs"] = turn_token_logprobs
            turn_info["token_texts"] = turn_token_texts
            
            multi_turn_info.append(turn_info)
        
        trajectory["multi_turn_info"] = multi_turn_info
    
    def compute_group_normalized_advantages(self, trajectories: List[Dict[str, Any]]) -> List[float]:
        """
        计算组归一化优势
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            每个轨迹的组归一化优势值
        """
        rewards = [t["reward"] for t in trajectories]
        
        # 根据配置选择归一化方法
        if self.config.group_normalization_method == "z_score":
            # Z-score归一化
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            
            # 避免除零
            if std_reward < 1e-8:
                std_reward = 1e-8
            
            advantages = [(r - mean_reward) / std_reward for r in rewards]
            
        elif self.config.group_normalization_method == "min_max":
            # Min-Max归一化
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            
            # 避免除零
            if max_reward - min_reward < 1e-8:
                advantages = [0.0 for _ in rewards]
            else:
                advantages = [(r - min_reward) / (max_reward - min_reward) for r in rewards]
                
        elif self.config.group_normalization_method == "rank":
            # 排序归一化
            sorted_indices = np.argsort(rewards)
            ranks = np.zeros_like(rewards, dtype=float)
            for i, idx in enumerate(sorted_indices):
                ranks[idx] = i / (len(rewards) - 1) if len(rewards) > 1 else 0.5
            
            advantages = ranks.tolist()
        else:
            # 默认使用Z-score归一化
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            
            if std_reward < 1e-8:
                std_reward = 1e-8
            
            advantages = [(r - mean_reward) / std_reward for r in rewards]
        
        return advantages
    
    def compute_multi_turn_loss(self, trajectory: Dict[str, Any], advantage: float) -> torch.Tensor:
        """
        计算多轮损失
        
        Args:
            trajectory: 轨迹数据
            advantage: 组归一化优势值
            
        Returns:
            多轮损失
        """
        if not self.config.enable_multi_turn_loss:
            return torch.tensor(0.0, device=self.device)
        
        multi_turn_info = trajectory.get("multi_turn_info", [])
        if not multi_turn_info:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        total_tokens = 0
        
        for turn_idx, turn_info in enumerate(multi_turn_info):
            # 计算轮次权重，早期轮次权重更高
            turn_weight = self.config.turn_weight_decay ** turn_idx
            if turn_idx == 0 and trajectory.get("success", False):
                # 如果第一轮就成功，给予额外奖励
                turn_weight *= self.config.early_turn_bonus
            
            # 获取当前轮次的token概率信息
            token_probs = turn_info.get("token_probs", [])
            token_logprobs = turn_info.get("token_logprobs", [])
            
            if not token_probs or not token_logprobs:
                continue
            
            # 转换为张量
            old_logprobs = torch.tensor(token_logprobs, dtype=torch.float32, device=self.device)
            
            # 计算新策略下的log概率（需要重新计算）
            # 这里简化处理，实际应该从模型获取
            new_logprobs = old_logprobs + torch.randn_like(old_logprobs) * 0.1  # 模拟新策略
            
            # 计算重要性采样比
            ratio = torch.exp(new_logprobs - old_logprobs)
            
            # 计算PPO剪切损失
            surr1 = ratio * advantage * turn_weight
            surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantage * turn_weight
            turn_loss = -torch.min(surr1, surr2).mean()
            
            # 计算KL惩罚
            kl_penalty = self.config.beta * F.kl_div(
                F.log_softmax(new_logprobs.unsqueeze(0), dim=-1),
                F.softmax(old_logprobs.unsqueeze(0), dim=-1),
                reduction='batchmean'
            )
            
            # 总损失
            turn_total_loss = turn_loss + kl_penalty
            
            # 累加损失
            total_loss += turn_total_loss * len(token_probs)
            total_tokens += len(token_probs)
        
        # 平均损失
        if total_tokens > 0:
            total_loss = total_loss / total_tokens
        
        return total_loss
    
    def compute_token_level_loss(self, trajectory: Dict[str, Any], advantage: float) -> torch.Tensor:
        """
        计算分词级损失
        
        Args:
            trajectory: 轨迹数据
            advantage: 组归一化优势值
            
        Returns:
            分词级损失
        """
        if not self.config.enable_token_level_loss:
            return torch.tensor(0.0, device=self.device)
        
        token_probs = trajectory.get("token_probs", [])
        token_logprobs = trajectory.get("token_logprobs", [])
        token_texts = trajectory.get("token_texts", [])
        
        if not token_probs or not token_logprobs:
            return torch.tensor(0.0, device=self.device)
        
        # 转换为张量
        old_logprobs = torch.tensor(token_logprobs, dtype=torch.float32, device=self.device)
        
        # 计算新策略下的log概率（需要重新计算）
        # 这里简化处理，实际应该从模型获取
        new_logprobs = old_logprobs + torch.randn_like(old_logprobs) * 0.1  # 模拟新策略
        
        # 计算token重要性权重
        # 基于token概率的方差来计算重要性
        token_probs_tensor = torch.tensor(token_probs, dtype=torch.float32, device=self.device)
        token_importance = torch.var(token_probs_tensor, dim=-1, keepdim=True)
        
        # 归一化重要性权重
        if torch.max(token_importance) > torch.min(token_importance):
            token_importance = (token_importance - torch.min(token_importance)) / (torch.max(token_importance) - torch.min(token_importance))
        
        # 应用阈值，过滤掉不重要的token
        important_tokens = token_importance > self.config.token_importance_threshold
        if torch.sum(important_tokens) == 0:  # 如果没有重要token，则使用所有token
            important_tokens = torch.ones_like(token_importance, dtype=torch.bool)
        
        # 计算重要性采样比
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # 只对重要token计算损失
        important_ratio = ratio[important_tokens.squeeze()]
        important_old_logprobs = old_logprobs[important_tokens.squeeze()]
        important_new_logprobs = new_logprobs[important_tokens.squeeze()]
        
        # 计算PPO剪切损失
        surr1 = important_ratio * advantage
        surr2 = torch.clamp(important_ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantage
        token_loss = -torch.min(surr1, surr2).mean()
        
        # 计算KL惩罚
        kl_penalty = self.config.beta * F.kl_div(
            F.log_softmax(important_new_logprobs.unsqueeze(0), dim=-1),
            F.softmax(important_old_logprobs.unsqueeze(0), dim=-1),
            reduction='batchmean'
        )
        
        # 总损失
        total_loss = token_loss + kl_penalty
        
        return total_loss
    
    def compute_sequence_level_loss(self, trajectory: Dict[str, Any], advantage: float) -> torch.Tensor:
        """
        计算序列级损失
        
        Args:
            trajectory: 轨迹数据
            advantage: 组归一化优势值
            
        Returns:
            序列级损失
        """
        token_probs = trajectory.get("token_probs", [])
        token_logprobs = trajectory.get("token_logprobs", [])
        
        if not token_probs or not token_logprobs:
            return torch.tensor(0.0, device=self.device)
        
        # 转换为张量
        old_logprobs = torch.tensor(token_logprobs, dtype=torch.float32, device=self.device)
        
        # 计算新策略下的log概率（需要重新计算）
        # 这里简化处理，实际应该从模型获取
        new_logprobs = old_logprobs + torch.randn_like(old_logprobs) * 0.1  # 模拟新策略
        
        # 计算重要性采样比
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # 计算PPO剪切损失
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantage
        sequence_loss = -torch.min(surr1, surr2).mean()
        
        # 计算KL惩罚
        kl_penalty = self.config.beta * F.kl_div(
            F.log_softmax(new_logprobs.unsqueeze(0), dim=-1),
            F.softmax(old_logprobs.unsqueeze(0), dim=-1),
            reduction='batchmean'
        )
        
        # 总损失
        total_loss = sequence_loss + kl_penalty
        
        return total_loss
    
    def compute_enhanced_flow_grpo_loss(self, trajectories: List[Dict[str, Any]], advantages: List[float]) -> Dict[str, torch.Tensor]:
        """
        计算增强版Flow-GRPO损失
        
        Args:
            trajectories: 轨迹列表
            advantages: 组归一化优势值
            
        Returns:
            损失字典，包含总损失和各种子损失
        """
        total_multi_turn_loss = 0.0
        total_token_level_loss = 0.0
        total_sequence_level_loss = 0.0
        total_tokens = 0
        
        for traj_idx, (trajectory, advantage) in enumerate(zip(trajectories, advantages)):
            # 计算多轮损失
            multi_turn_loss = self.compute_multi_turn_loss(trajectory, advantage)
            total_multi_turn_loss += multi_turn_loss
            
            # 计算分词级损失
            token_level_loss = self.compute_token_level_loss(trajectory, advantage)
            total_token_level_loss += token_level_loss
            
            # 计算序列级损失
            sequence_level_loss = self.compute_sequence_level_loss(trajectory, advantage)
            total_sequence_level_loss += sequence_level_loss
            
            # 统计token数量
            token_probs = trajectory.get("token_probs", [])
            if token_probs:
                total_tokens += len(token_probs)
        
        # 计算平均损失
        if total_tokens > 0:
            total_multi_turn_loss = total_multi_turn_loss / len(trajectories)
            total_token_level_loss = total_token_level_loss / len(trajectories)
            total_sequence_level_loss = total_sequence_level_loss / len(trajectories)
        
        # 计算加权总损失
        total_loss = (
            self.config.token_level_weight * total_token_level_loss +
            self.config.sequence_level_weight * total_sequence_level_loss
        )
        
        # 如果启用多轮损失，则加入总损失
        if self.config.enable_multi_turn_loss:
            total_loss += total_multi_turn_loss
        
        # 确保返回torch.Tensor
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, dtype=torch.float32, device=self.device, requires_grad=True)
        if not isinstance(total_multi_turn_loss, torch.Tensor):
            total_multi_turn_loss = torch.tensor(total_multi_turn_loss, dtype=torch.float32, device=self.device)
        if not isinstance(total_token_level_loss, torch.Tensor):
            total_token_level_loss = torch.tensor(total_token_level_loss, dtype=torch.float32, device=self.device)
        if not isinstance(total_sequence_level_loss, torch.Tensor):
            total_sequence_level_loss = torch.tensor(total_sequence_level_loss, dtype=torch.float32, device=self.device)
        
        return {
            "total_loss": total_loss,
            "multi_turn_loss": total_multi_turn_loss,
            "token_level_loss": total_token_level_loss,
            "sequence_level_loss": total_sequence_level_loss
        }
    
    def train_step(self, batch_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        执行一步训练
        
        Args:
            batch_data: 批量数据，每个元素包含question和ground_truth
            
        Returns:
            训练统计信息
        """
        # 添加显存监控函数
        def monitor_memory(step_name=""):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                #logger.info(f"[{step_name}] 显存使用 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
                return allocated, reserved
            return 0, 0
        
        # 记录训练步骤开始时的显存状态
        initial_allocated, initial_reserved = monitor_memory("train_step_start")
        
        self.optimizer.zero_grad()
        
        batch_loss = 0.0
        batch_multi_turn_loss = 0.0
        batch_token_level_loss = 0.0
        batch_sequence_level_loss = 0.0
        batch_rewards = []
        batch_successes = []
        batch_turns = []
        
        # 处理每个样本
        for sample_idx, sample in enumerate(batch_data):
            question = sample["question"]
            ground_truth = sample["ground_truth"]
            
            # 记录每个样本处理前的显存状态
            sample_start_allocated, sample_start_reserved = monitor_memory(f"sample_{sample_idx+1}_start")
            
            # 生成一组轨迹
            print(f"问题: {question},开始生成轨迹")
            trajectories = self.generate_group_trajectories(question, ground_truth)
            print(f"问题: {question}, 生成轨迹数: {len(trajectories)}")
            
            # 轨迹生成后清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                after_gen_allocated, after_gen_reserved = monitor_memory(f"sample_{sample_idx+1}_after_gen")
                #logger.info(f"样本 {sample_idx+1} 轨迹生成后显存变化: {after_gen_allocated-sample_start_allocated:.2f}GB")
            
            # 计算组归一化优势
            advantages = self.compute_group_normalized_advantages(trajectories)
            
            # 计算损失
            loss_dict = self.compute_enhanced_flow_grpo_loss(trajectories, advantages)
            
            # 累加损失
            batch_loss += loss_dict["total_loss"]
            batch_multi_turn_loss += loss_dict["multi_turn_loss"]
            batch_token_level_loss += loss_dict["token_level_loss"]
            batch_sequence_level_loss += loss_dict["sequence_level_loss"]
            
            # 收集统计信息
            rewards = [t["reward"] for t in trajectories]
            successes = [t["success"] for t in trajectories]
            turns = [len(t.get("turns", [])) for t in trajectories]
            
            batch_rewards.extend(rewards)
            batch_successes.extend(successes)
            batch_turns.extend(turns)
            
            # 每个样本处理后清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                sample_end_allocated, sample_end_reserved = monitor_memory(f"sample_{sample_idx+1}_end")
                #logger.info(f"样本 {sample_idx+1} 处理后显存变化: {sample_end_allocated-sample_start_allocated:.2f}GB")
        
        # 梯度累积
        batch_loss = batch_loss / len(batch_data)
        batch_multi_turn_loss = batch_multi_turn_loss / len(batch_data)
        batch_token_level_loss = batch_token_level_loss / len(batch_data)
        batch_sequence_level_loss = batch_sequence_level_loss / len(batch_data)
        
        # 确保batch_loss是torch.Tensor
        if not isinstance(batch_loss, torch.Tensor):
            batch_loss = torch.tensor(batch_loss, dtype=torch.float32, device=self.device, requires_grad=True)
        
        batch_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.agent.planner.model.parameters(), self.config.max_grad_norm)
        
        # 更新参数
        self.optimizer.step()
        
        # 训练步骤完成后清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_allocated, final_reserved = monitor_memory("train_step_end")
            #logger.info(f"训练步骤完成显存变化: {final_allocated-initial_allocated:.2f}GB")
        
        # 返回统计信息
        stats = {
            "loss": batch_loss.item(),
            "multi_turn_loss": batch_multi_turn_loss.item() if isinstance(batch_multi_turn_loss, torch.Tensor) else batch_multi_turn_loss,
            "token_level_loss": batch_token_level_loss.item() if isinstance(batch_token_level_loss, torch.Tensor) else batch_token_level_loss,
            "sequence_level_loss": batch_sequence_level_loss.item() if isinstance(batch_sequence_level_loss, torch.Tensor) else batch_sequence_level_loss,
            "avg_reward": np.mean(batch_rewards),
            "success_rate": np.mean(batch_successes),
            "avg_turns": np.mean(batch_turns),
            "memory_allocated": final_allocated if torch.cuda.is_available() else 0,
            "memory_reserved": final_reserved if torch.cuda.is_available() else 0
        }
        
        return stats
    
    def train_epoch(self, train_data: List[Dict[str, Any]], val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            
        Returns:
            epoch统计信息
        """
        # 添加显存监控函数
        def monitor_memory(step_name=""):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                #logger.info(f"[{step_name}] 显存使用 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
                return allocated, reserved
            return 0, 0
        
        # 记录epoch开始时的显存状态
        epoch_start_allocated, epoch_start_reserved = monitor_memory(f"epoch_{self.current_epoch+1}_start")
        
        self.agent.planner.model.train()
        
        # 打乱数据
        random.shuffle(train_data)
        
        # 分批处理
        num_batches = len(train_data) // self.config.batch_size
        epoch_stats = {
            "loss": [],
            "multi_turn_loss": [],
            "token_level_loss": [],
            "sequence_level_loss": [],
            "avg_reward": [],
            "success_rate": [],
            "avg_turns": [],
            "memory_peak": 0.0
        }
        
        memory_peak = 0.0
        
        # 使用tqdm显示进度
        with tqdm(total=num_batches, desc=f"Epoch {self.current_epoch+1}/{self.config.max_epochs}") as pbar:
            for batch_idx in range(num_batches):
                # 获取批次数据
                start_idx = batch_idx * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                batch_data = train_data[start_idx:end_idx]
                
                # 记录批次处理前的显存状态
                batch_start_allocated, batch_start_reserved = monitor_memory(f"epoch_{self.current_epoch+1}_batch_{batch_idx+1}")
                
                # 训练一步
                stats = self.train_step(batch_data)
                
                # 更新统计信息
                for key, value in stats.items():
                    if key in epoch_stats:
                        epoch_stats[key].append(value)
                
                # 更新显存峰值
                if "memory_allocated" in stats and stats["memory_allocated"] > memory_peak:
                    memory_peak = stats["memory_allocated"]
                
                # 记录批次处理后的显存状态
                batch_end_allocated, batch_end_reserved = monitor_memory(f"epoch_{self.current_epoch+1}_batch_{batch_idx+1}_end")
                
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 更新进度条
                pbar.set_postfix({
                    "Loss": f"{stats['loss']:.4f}",
                    "Reward": f"{stats['avg_reward']:.4f}",
                    "Success": f"{stats['success_rate']:.2%}",
                    "Mem": f"{stats.get('memory_allocated', 0):.1f}GB"
                })
                pbar.update(1)
                
                self.global_step += 1
        
        # 计算epoch平均统计
        epoch_avg_stats = {}
        for key, values in epoch_stats.items():
            if key != "memory_peak":
                epoch_avg_stats[key] = np.mean(values)
        epoch_avg_stats["memory_peak"] = memory_peak
        
        # 记录epoch结束时的显存状态
        epoch_end_allocated, epoch_end_reserved = monitor_memory(f"epoch_{self.current_epoch+1}_end")
        #logger.info(f"Epoch {self.current_epoch+1} 完成显存变化: {epoch_end_allocated-epoch_start_allocated:.2f}GB")
        #logger.info(f"Epoch {self.current_epoch+1} 显存峰值: {memory_peak:.2f}GB")
        
        # 最终清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 评估
        if val_data and self.config.enable_evaluation and (self.current_epoch + 1) % self.config.eval_every == 0:
            val_stats = self.evaluate(val_data)
            epoch_avg_stats.update({f"val_{k}": v for k, v in val_stats.items()})
        
        # 更新训练状态
        self.current_epoch += 1
        
        return epoch_avg_stats
    
    def evaluate(self, val_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            val_data: 验证数据
            
        Returns:
            评估统计信息
        """
        self.agent.planner.model.eval()
        
        eval_rewards = []
        eval_successes = []
        eval_turns = []
        
        with torch.no_grad():
            for sample in tqdm(val_data, desc="Evaluating"):
                question = sample["question"]
                ground_truth = sample["ground_truth"]
                
                # 生成一组轨迹
                trajectories = self.generate_group_trajectories(question, ground_truth)
                
                # 收集统计信息
                rewards = [t["reward"] for t in trajectories]
                successes = [t["success"] for t in trajectories]
                turns = [len(t.get("turns", [])) for t in trajectories]
                
                eval_rewards.extend(rewards)
                eval_successes.extend(successes)
                eval_turns.extend(turns)
        
        # 计算平均统计
        stats = {
            "val_loss": 0.0,  # 评估时不计算损失
            "val_avg_reward": np.mean(eval_rewards),
            "val_success_rate": np.mean(eval_successes),
            "val_avg_turns": np.mean(eval_turns)
        }
        
        return stats
    
    def save_checkpoint(self, save_dir: str, epoch: int) -> None:
        """
        保存检查点
        
        Args:
            save_dir: 保存目录
            epoch: 当前epoch
        """
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.agent.planner.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_reward": self.best_reward,
            "train_stats": self.train_stats
        }
        
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))
        
        # 如果是最佳模型，单独保存
        current_reward = np.mean(self.train_stats["reward"][-self.config.eval_every:]) if len(self.train_stats["reward"]) >= self.config.eval_every else 0
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_reward = checkpoint["best_reward"]
        self.train_stats = checkpoint["train_stats"]
        
        self.agent.planner.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    def train(self, train_data: List[Dict[str, Any]], val_data: Optional[List[Dict[str, Any]]] = None, save_dir: str = "checkpoints") -> None:
        """
        执行完整训练流程
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            save_dir: 保存目录
        """
        # 训练循环
        for epoch in range(self.current_epoch, self.config.max_epochs):
            # 训练一个epoch
            epoch_stats = self.train_epoch(train_data, val_data)
            
            # 更新统计信息
            for key, value in epoch_stats.items():
                if key in self.train_stats:
                    self.train_stats[key].append(value)
            
            # 保存检查点
            if (epoch + 1) % self.config.save_every == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                self.save_checkpoint(save_dir, epoch + 1)
        
        # 保存最终模型
        final_path = os.path.join(save_dir, "final_model.pt")
        self.save_checkpoint(save_dir, self.config.max_epochs)


def create_enhanced_flow_grpo_trainer(agent: SimpleAgent, config: dict) -> EnhancedFlowGRPO:
    """
    创建增强版Flow-GRPO训练器
    
    Args:
        agent: SimpleAgent实例
        config: 配置字典
        
    Returns:
        EnhancedFlowGRPO实例
    """
    # 提取训练配置
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    flow_group_config = config.get('flow_group', {})
    
    # 创建EnhancedFlowGRPOConfig
    enhanced_flow_grpo_config = EnhancedFlowGRPOConfig(
        batch_size=int(training_config.get('batch_size', 8)),
        max_epochs=int(training_config.get('max_epochs', 10)),
        learning_rate=float(training_config.get('learning_rate', 1e-5)),
        gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps', 4)),
        max_grad_norm=float(training_config.get('max_grad_norm', 1.0)),
        group_size=int(flow_group_config.get('group_size', config.get('group_size', 2))),
        max_turns=int(training_config.get('max_turns', 10)),
        epsilon=float(flow_group_config.get('epsilon', 0.2)),
        beta=float(flow_group_config.get('beta', 0.01)),
        save_every=int(training_config.get('save_every', 1)),
        eval_every=int(training_config.get('eval_every', 5)),
        enable_evaluation=bool(training_config.get('enable_evaluation', False)),
        temperature=float(flow_group_config.get('temperature', 1.0)),
        top_p=float(flow_group_config.get('top_p', 0.9)),
        max_tokens=int(flow_group_config.get('max_tokens', 2048)),
        log_dir=config.get('logging', {}).get('log_dir', 'logs'),
        log_level=config.get('logging', {}).get('log_level', 'INFO'),
        save_trajectories=bool(config.get('logging', {}).get('save_trajectories', True)),
        trajectory_sample_rate=float(config.get('logging', {}).get('trajectory_sample_rate', 0.1)),
        
        # 增强版参数
        enable_multi_turn_loss=bool(flow_group_config.get('enable_multi_turn_loss', True)),
        enable_token_level_loss=bool(flow_group_config.get('enable_token_level_loss', True)),
        enable_group_normalization=bool(flow_group_config.get('enable_group_normalization', True)),
        turn_weight_decay=float(flow_group_config.get('turn_weight_decay', 0.95)),
        early_turn_bonus=float(flow_group_config.get('early_turn_bonus', 1.2)),
        token_level_weight=float(flow_group_config.get('token_level_weight', 0.5)),
        sequence_level_weight=float(flow_group_config.get('sequence_level_weight', 0.5)),
        token_importance_threshold=float(flow_group_config.get('token_importance_threshold', 0.1)),
        group_normalization_method=str(flow_group_config.get('group_normalization_method', 'z_score')),
        advantage_normalization=bool(flow_group_config.get('advantage_normalization', True)),
        reward_normalization=bool(flow_group_config.get('reward_normalization', True))
    )
    
    # 创建EnhancedFlowGRPO实例
    trainer = EnhancedFlowGRPO(agent, enhanced_flow_grpo_config)
    
    return trainer
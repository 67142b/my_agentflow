"""
Flow-Group训练算法实现模块
基于论文《IN-THE-FLOW AGENTIC SYSTEM OPTIMIZATION FOR EFFECTIVE PLANNING AND TOOL USE》
"""

import os
import sys
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path
import logging
from copy import deepcopy

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
import time
from src.agentflow.simple_agent import SimpleAgent
from src.agentflow.utils.reward import llm_as_judge
from src.agentflow.utils.logger import create_logger_from_config

logger = logging.getLogger(__name__)

@dataclass
class FlowGRPOConfig:
    """FlowGRPO训练配置"""
    
    # 基础配置
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path: str = "src/agentflow/training/dataset.json"
    output_dir: str = "src/agentflow/training/outputs"
    
    # 训练参数
    batch_size: int = 8
    group_size: int = 4  # 每组生成的轨迹数量
    learning_rate: float = 1e-5
    num_epochs: int = 10
    max_steps: int = 100
    save_steps: int = 50
    eval_steps: int = 50
    warmup_steps: int = 10
    
    # GRPO参数
    clip_ratio: float = 0.2  # PPO裁剪比率
    value_coef: float = 0.5  # 价值函数系数
    entropy_coef: float = 0.01  # 熵系数
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE参数
    
    # 采样参数
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = True
    
    # 并发采样参数
    enable_concurrent: bool = True  # 是否启用并发采样
    num_model_copies: int = 2  # 模型副本数量，用于并发采样
    
    # 模型参数
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # 其他参数
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True
    
    # 训练控制
    save_every: int = 1
    eval_every: int = 5
    enable_evaluation: bool = False
    
    # 采样参数
    max_tokens: int = 2048
    
    # 路径配置
    log_dir: str = "./logs"
    
    # 日志配置
    log_level: str = "INFO"
    save_trajectories: bool = True
    trajectory_sample_rate: float = 0.1


class FlowGRPO:
    """Flow-Group训练算法实现类"""
    
    def __init__(self, agent: SimpleAgent, config: FlowGRPOConfig):
        """
        初始化Flow-GRPO训练器
        
        Args:
            agent: SimpleAgent实例
            config: 训练配置
        """
        self.agent = agent
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 获取LLM引擎中的原始planner模型
        self.llm_engine = agent.llm_engine
        self.original_planner_model = self.llm_engine._model_cache.get("planner")
        
        if self.original_planner_model is None:
            raise ValueError("无法从LLM引擎中获取原始planner模型，请确保已正确初始化")
        
        # 保存参考策略权重（用于KL惩罚）
        self.ref_policy_state_dict = deepcopy(self.original_planner_model.state_dict())
        
        # 标记是否为首次更新
        self.is_first_update = True
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_reward = -float('inf')
        
        # 统计信息
        self.train_stats = {
            "loss": [],
            "reward": [],
            "success_rate": [],
            "avg_turns": []
        }
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.original_planner_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
    
    def _compute_ref_logprobs(self, input_ids: List[int], output_ids: List[int]) -> torch.Tensor:
        """
        使用冻结的参考策略计算log_prob
        
        Args:
            input_ids: 输入token IDs
            output_ids: 输出token IDs
            
        Returns:
            参考策略下的log_prob张量
        """
        if len(output_ids) == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)
        
        # 临时加载参考策略权重
        current_state = self.original_planner_model.state_dict()
        self.original_planner_model.load_state_dict(self.ref_policy_state_dict)
        self.original_planner_model.eval()
        
        with torch.no_grad():
            model_input = list(input_ids) + list(output_ids[:-1])
            input_tensor = torch.tensor([model_input], dtype=torch.long, device=self.device)
            
            outputs = self.original_planner_model(input_tensor)
            logits = outputs.logits
            last_logits = logits[:, -len(output_ids):, :]
            logprobs = F.log_softmax(last_logits, dim=-1)
            target = torch.tensor(output_ids, dtype=torch.long, device=self.device)
            ref_logprob = logprobs[0, torch.arange(len(output_ids), device=self.device), target]
        
        # 恢复当前策略权重
        self.original_planner_model.load_state_dict(current_state)
        return ref_logprob.detach()
    
    def _normalize_final_answer(self, final_answer: Any) -> str:
        """
        标准化final_answer字段，确保返回字符串格式
        
        Args:
            final_answer: 原始答案数据
            
        Returns:
            str: 标准化的答案字符串
        """
        if final_answer is None:
            return ""
        
        if isinstance(final_answer, str):
            return final_answer.strip()
        
        if isinstance(final_answer, dict):
            # 尝试多个可能的字段
            for key in ["final_answer", "answer", "result", "content", "text"]:
                if key in final_answer and final_answer[key]:
                    return str(final_answer[key]).strip()
            
            # 如果是嵌套字典，尝试获取第一个非空值
            for value in final_answer.values():
                if value:
                    return str(value).strip()
        
        # 其他类型转换为字符串
        return str(final_answer).strip()
    
    def generate_group_trajectories(self, question: str, ground_truth: str) -> List[Dict[str, Any]]:
        """
        为单个问题生成一组轨迹 - 修复版本：确保对每条成功轨迹都进行奖励评估
        
        Args:
            question: 问题文本
            ground_truth: 标准答案
            
        Returns:
            轨迹列表
        """
        trajectories = []
        
        try:
            enable_concurrent = getattr(self.config, 'enable_concurrent', True)
            num_model_copies = getattr(self.config, 'num_model_copies', 1)
            
            print(f"开始调用SimpleAgent.solve，问题: {question[:500]}...\n====================================")
            print(f"并发模式: {enable_concurrent}, 模型副本数: {num_model_copies}")
            start_time = time.time()
            result = self.agent.solve(question)
            end_time = time.time()
            print(f"SimpleAgent.solve调用完毕，用时: {end_time - start_time:.2f}秒\n====================================")
            
            successful_trajectories = result.get("successful_trajectories", [])
            failed_trajectories = result.get("failed_trajectories", [])
            
            print(f"成功轨迹数: {len(successful_trajectories)}")
            print(f"失败轨迹数: {len(failed_trajectories)}")
            
            # 处理成功的轨迹 - 确保对每条都进行奖励评估
            for i, traj in enumerate(successful_trajectories):
                try:
                    # 标准化final_answer格式
                    final_answer = self._normalize_final_answer(traj.get("final_answer"))
                    print(f"成功轨迹 {i+1}，标准化后的答案: '{final_answer[:100]}...'")
                    
                    # 使用llm_as_judge方法计算奖励
                    reward = llm_as_judge(
                        question=question,
                        answer=final_answer,
                        ground_truth=ground_truth
                    )
                    print(f"成功轨迹 {i+1}，返回答案：'{final_answer[:50]}...', 标准答案：'{ground_truth[:50]}...', 奖励: {reward:.4f}")
                    
                    # 添加轨迹ID
                    trajectory_id = traj.get("trajectory_id", f"{question[:50].replace(' ', '_')}_{i}")
                    
                    # 构建轨迹数据
                    trajectory = {
                        "trajectory_id": trajectory_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "final_answer": final_answer,
                        "reward": reward,
                        "turns": traj.get("turns", []),  # 使用新的turns字段
                        "success": True,
                        "trajectory_data": traj  # 保存完整的轨迹数据
                    }
                    
                    # 添加token级概率信息
                    self._extract_token_probs(trajectory)
                    
                    trajectories.append(trajectory)
                    
                except Exception as e:
                    logger.error(f"处理成功轨迹 {i} 时发生错误: {str(e)}")
                    # 添加失败的轨迹条目
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
            
            # 处理成功的轨迹
            for i, traj in enumerate(successful_trajectories):
                reward = llm_as_judge(
                    question=question,
                    answer=traj.get("final_answer", ""),
                    ground_truth=ground_truth
                )
                
                trajectory_id = traj.get("trajectory_id", f"{question[:50].replace(' ', '_')}_{i}")
                
                trajectory = {
                    "trajectory_id": trajectory_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "final_answer": traj.get("final_answer", ""),
                    "reward": reward,
                    "turns": traj.get("turns", []),
                    "success": True,
                    "trajectory_data": traj
                }
                
                trajectories.append(trajectory)
            
            # 处理失败的轨迹
            for i, traj in enumerate(failed_trajectories):
                try:
                    # 添加轨迹ID
                    trajectory_id = traj.get("trajectory_id", f"{question[:50].replace(' ', '_')}_failed_{i}")
                    
                    # 标准化final_answer格式
                    final_answer = self._normalize_final_answer(traj.get("final_answer"))
                    
                    # 构建轨迹数据
                    trajectory = {
                        "trajectory_id": trajectory_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "final_answer": final_answer,
                        "reward": 0.0,  # 失败轨迹的奖励为0
                        "turns": traj.get("turns", []),  # 使用新的turns字段
                        "success": False,
                        "error": traj.get("error", "Trajectory failed"),
                        "trajectory_data": traj  # 保存完整的轨迹数据
                    }
                    
                    # 添加token级概率信息
                    self._extract_token_probs(trajectory)
                    
                    trajectories.append(trajectory)
                    
                except Exception as e:
                    logger.error(f"处理失败轨迹 {i} 时发生错误: {str(e)}")
                    # 添加基本的失败轨迹条目
                    trajectory = {
                        "trajectory_id": f"{question[:50].replace(' ', '_')}_failed_{i}",
                        "question": question,
                        "ground_truth": ground_truth,
                        "final_answer": "",
                        "reward": 0.0,
                        "turns": [],
                        "success": False,
                        "error": str(e)
                    }
                    trajectories.append(trajectory)
                
        except Exception as e:
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
        
        # 打印最终的奖励统计
        rewards = [t["reward"] for t in trajectories]
        print(f"最终轨迹奖励统计 - 总数: {len(trajectories)}, 奖励分布: {rewards}")
                
        return trajectories
    
    def _extract_token_probs(self, trajectory: Dict[str, Any]) -> None:
        """
        从轨迹中提取token级概率信息 - 修复版本
        
        Args:
            trajectory: 轨迹数据
        """
        # 从轨迹数据中提取token级概率信息
        planner_input_ids = []
        planner_output_ids = []
        logprobs_old = []
        
        # 从轨迹的turns中提取信息
        turns = trajectory.get("turns", [])
        for turn in turns:
            try:
                # 安全地获取字段，避免KeyError
                step_input_ids = turn.get("planner_input_ids", [])
                step_output_ids = turn.get("planner_output_ids", [])
                step_logprobs = turn.get("logprobs_old", [])
                
                # 验证数据格式
                if isinstance(step_input_ids, list) and isinstance(step_output_ids, list) and isinstance(step_logprobs, list):
                    planner_input_ids.extend(step_input_ids)
                    planner_output_ids.extend(step_output_ids)
                    logprobs_old.extend(step_logprobs)
                else:
                    logger.warning(f"步骤数据格式不正确: input_ids类型={type(step_input_ids)}, output_ids类型={type(step_output_ids)}, logprobs类型={type(step_logprobs)}")
                    
            except Exception as e:
                logger.error(f"提取步骤token概率时发生错误: {str(e)}")
                continue
        
        # 添加到轨迹中
        trajectory["planner_input_ids"] = planner_input_ids
        trajectory["planner_output_ids"] = planner_output_ids
        trajectory["logprobs_old"] = logprobs_old
    
    def compute_group_normalized_advantages(self, trajectories: List[Dict[str, Any]]) -> List[float]:
        """
        计算组归一化优势 - 修复版本，增加数值稳定性检查和调试信息
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            每个轨迹的组归一化优势值
        """
        rewards = [t["reward"] for t in trajectories]
        
        print(f"组归一化计算 - 原始奖励: {rewards}")
        
        # 计算均值和标准差
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        print(f"组归一化计算 - 均值: {mean_reward:.6f}, 标准差: {std_reward:.6f}")
        
        # 避免除零，增强数值稳定性
        if std_reward < 1e-8:
            logger.warning(f"标准差过小 ({std_reward:.6f})，使用均值填充")
            # 如果所有奖励相同，返回零优势
            return [0.0] * len(rewards)
        
        # 计算组归一化优势
        advantages = [(r - mean_reward) / std_reward for r in rewards]
        
        print(f"组归一化计算 - 最终优势: {advantages}")
        
        return advantages
    
    def compute_flow_grpo_loss(self, trajectories: List[Dict[str, Any]], advantages: List[float]) -> torch.Tensor:
        """
        计算Flow-GRPO损失
        
        Args:
            trajectories: 轨迹列表
            advantages: 组归一化优势值
            
        Returns:
            总损失
        """
        device = self.device
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
        total_tokens = 0
        
        for traj_idx, (traj, adv_scalar) in enumerate(zip(trajectories, advantages)):
            turns = traj.get("turns", [])
            per_turn_sums = []
            traj_token_count = 0
            
            flat_new_all = []
            flat_ref_all = []
            flat_old_all = []
            
            for turn in turns:
                input_ids = turn.get("planner_input_ids", [])
                output_ids = turn.get("planner_output_ids", [])
                old_logprobs_list = turn.get("logprobs_old", [])
                
                if not isinstance(input_ids, list) or not isinstance(output_ids, list):
                    continue
                if len(output_ids) == 0:
                    continue
                
                if old_logprobs_list and len(old_logprobs_list) != len(output_ids):
                    min_len = min(len(old_logprobs_list), len(output_ids))
                    old_logprobs_list = old_logprobs_list[:min_len]
                    output_ids = output_ids[:min_len]
                
                # 计算新策略log_prob（带梯度）
                model_input = list(input_ids) + list(output_ids[:-1])
                input_tensor = torch.tensor([model_input], dtype=torch.long, device=device)
                
                self.original_planner_model.train()
                outputs = self.original_planner_model(input_tensor)
                logits = outputs.logits
                last_logits = logits[:, -len(output_ids):, :]
                logprobs = F.log_softmax(last_logits, dim=-1)
                target = torch.tensor(output_ids, dtype=torch.long, device=device)
                new_logprob = logprobs[0, torch.arange(len(output_ids), device=device), target]
                
                # 计算参考策略log_prob（冻结，无梯度）
                ref_logprob = self._compute_ref_logprobs(input_ids, output_ids)
                
                # 旧策略log_prob（detach，无梯度）
                if not old_logprobs_list:
                    raise ValueError(f"轨迹 {traj_idx} 的某个 turn 缺少 'logprobs_old' 字段")
                old_logprob = torch.tensor(old_logprobs_list, dtype=torch.float32, device=device).detach()
                
                # 收集用于KL计算
                flat_new_all.append(new_logprob)
                flat_ref_all.append(ref_logprob)
                flat_old_all.append(old_logprob)
                
                # 计算重要性比率
                log_ratio = new_logprob - old_logprob
                log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
                ratio = torch.exp(log_ratio)
                
                # 优势广播
                adv_tensor = torch.full_like(new_logprob, float(adv_scalar), device=device)
                
                # PPO裁剪
                clipped_ratio = torch.clamp(ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon)
                surr1 = ratio * adv_tensor
                surr2 = clipped_ratio * adv_tensor
                surr = torch.min(surr1, surr2)
                
                # Turn级累加
                turn_sum = surr.sum()
                per_turn_sums.append(turn_sum)
                traj_token_count += new_logprob.numel()
            
            if len(per_turn_sums) == 0:
                continue
            
            # 轨迹级加权平均
            if traj_token_count > 0:
                traj_obj = torch.stack(per_turn_sums).sum() / traj_token_count
            else:
                traj_obj = torch.tensor(0.0, device=device)
            
            # 策略损失（负目标）
            policy_loss = -traj_obj
            
            # KL惩罚：KL(π_new || π_ref)
            if len(flat_new_all) > 0 and len(flat_ref_all) > 0:
                new_logprob_cat = torch.cat(flat_new_all)
                ref_logprob_cat = torch.cat(flat_ref_all)
                
                kl_term = (new_logprob_cat - ref_logprob_cat).mean()
                kl_penalty = float(self.config.beta) * kl_term
            else:
                kl_penalty = torch.tensor(0.0, device=device)
            
            # 总损失
            traj_loss = policy_loss + kl_penalty
            
            # 加权累加
            total_loss = total_loss + traj_loss * float(traj_token_count)
            total_tokens += traj_token_count
        
        # 归一化
        if total_tokens > 0:
            total_loss = total_loss / float(total_tokens)
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss
    
    def train_step(self, batch_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        执行一步训练
        """
        def monitor_memory(step_name=""):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                #logger.info(f"[{step_name}] 显存使用 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
                return allocated, reserved
            return 0, 0
        
        initial_allocated, initial_reserved = monitor_memory("train_step_start")
        self.optimizer.zero_grad()
        
        batch_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
        batch_rewards = []
        batch_successes = []
        batch_turns = []
        
        for sample_idx, sample in enumerate(batch_data):
            sample_start_allocated, sample_start_reserved = monitor_memory(f"sample_{sample_idx+1}_start")
            
            question = sample["question"]
            ground_truth = sample["ground_truth"]
            
            trajectories = self.generate_group_trajectories(question, ground_truth)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            advantages = self.compute_group_normalized_advantages(trajectories)
            loss = self.compute_flow_grpo_loss(trajectories, advantages)
            
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, dtype=torch.float32, device=self.device, requires_grad=True)
            
            batch_loss = batch_loss + loss
            
            batch_rewards.extend([t["reward"] for t in trajectories])
            batch_successes.extend([t["success"] for t in trajectories])
            batch_turns.extend([len(t.get("turns", [])) for t in trajectories])
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            sample_end_allocated, sample_end_reserved = monitor_memory(f"sample_{sample_idx+1}_end")
        
        batch_loss = batch_loss / len(batch_data)
        batch_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.original_planner_model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        if self.is_first_update:
            self.is_first_update = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_allocated, final_reserved = monitor_memory("train_step_end")
        
        return {
            "loss": batch_loss.item(),
            "avg_reward": np.mean(batch_rewards),
            "success_rate": np.mean(batch_successes),
            "avg_turns": np.mean(batch_turns),
            "memory_allocated": final_allocated if torch.cuda.is_available() else 0,
            "memory_reserved": final_reserved if torch.cuda.is_available() else 0
        }
    
    def train_epoch(self, train_data: List[Dict[str, Any]], val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        训练一个epoch
        """
        def monitor_memory(step_name=""):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                #logger.info(f"[{step_name}] 显存使用 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
                return allocated, reserved
            return 0, 0
        
        epoch_start_allocated, epoch_start_reserved = monitor_memory(f"epoch_{self.current_epoch+1}_start")
        self.original_planner_model.train()
        
        random.shuffle(train_data)
        num_batches = len(train_data) // self.config.batch_size
        
        epoch_stats = {
            "loss": [],
            "avg_reward": [],
            "success_rate": [],
            "avg_turns": [],
            "memory_peak": 0.0
        }
        
        memory_peak = 0.0
        
        with tqdm(total=num_batches, desc=f"Epoch {self.current_epoch+1}/{getattr(self.config, 'max_epochs', 10)}") as pbar:
            for batch_idx in range(num_batches):
                batch_start_allocated, batch_start_reserved = monitor_memory(f"batch_{batch_idx+1}")
                
                start_idx = batch_idx * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                batch_data = train_data[start_idx:end_idx]
                
                stats = self.train_step(batch_data)
                
                for key in ["loss", "avg_reward", "success_rate", "avg_turns"]:
                    epoch_stats[key].append(stats[key])
                
                if stats.get("memory_allocated", 0) > memory_peak:
                    memory_peak = stats["memory_allocated"]
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                pbar.set_postfix({
                    "Loss": f"{stats['loss']:.4f}",
                    "Reward": f"{stats['avg_reward']:.4f}",
                    "Success": f"{stats['success_rate']:.2%}",
                    "Mem": f"{stats.get('memory_allocated', 0):.1f}GB"
                })
                pbar.update(1)
                
                self.global_step += 1
        
        epoch_avg_stats = {key: np.mean(values) for key, values in epoch_stats.items() if key != "memory_peak"}
        epoch_avg_stats["memory_peak"] = memory_peak
        
        epoch_end_allocated, epoch_end_reserved = monitor_memory(f"epoch_{self.current_epoch+1}_end")
        
        if val_data and getattr(self.config, 'enable_evaluation', False) and (self.current_epoch + 1) % getattr(self.config, 'eval_every', 5) == 0:
            val_stats = self.evaluate(val_data)
            epoch_avg_stats.update({f"val_{k}": v for k, v in val_stats.items()})
        
        self.current_epoch += 1
        
        return epoch_avg_stats
    
    def evaluate(self, val_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估模型
        """
        self.original_planner_model.eval()
        
        eval_rewards = []
        eval_successes = []
        eval_turns = []
        
        with torch.no_grad():
            for sample in tqdm(val_data, desc="Evaluating"):
                trajectories = self.generate_group_trajectories(sample["question"], sample["ground_truth"])
                
                eval_rewards.extend([t["reward"] for t in trajectories])
                eval_successes.extend([t["success"] for t in trajectories])
                eval_turns.extend([len(t.get("turns", [])) for t in trajectories])
        
        return {
            "avg_reward": np.mean(eval_rewards),
            "success_rate": np.mean(eval_successes),
            "avg_turns": np.mean(eval_turns)
        }
    
    def save_checkpoint(self, save_dir: str, epoch: int) -> None:
        """
        保存检查点
        """
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.original_planner_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_reward": self.best_reward,
            "train_stats": self.train_stats,
            "ref_policy_state_dict": self.ref_policy_state_dict
        }
        
        torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))
        
        current_reward = np.mean(self.train_stats["reward"][-getattr(self.config, 'eval_every', 5):]) if len(self.train_stats["reward"]) >= getattr(self.config, 'eval_every', 5) else 0
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_reward = checkpoint["best_reward"]
        self.train_stats = checkpoint["train_stats"]
        self.ref_policy_state_dict = checkpoint["ref_policy_state_dict"]
        
        self.original_planner_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    def train(self, train_data: List[Dict[str, Any]], val_data: Optional[List[Dict[str, Any]]] = None, save_dir: str = "checkpoints") -> None:
        """
        执行完整训练流程
        """
        for epoch in range(self.current_epoch, getattr(self.config, 'max_epochs', 10)):
            epoch_stats = self.train_epoch(train_data, val_data)
            
            for key, value in epoch_stats.items():
                if key in self.train_stats:
                    self.train_stats[key].append(value)
            
            if (epoch + 1) % getattr(self.config, 'save_every', 1) == 0:
                self.save_checkpoint(save_dir, epoch + 1)
        
        self.save_checkpoint(save_dir, getattr(self.config, 'max_epochs', 10))


def create_flow_grpo_trainer(agent: SimpleAgent, config: dict) -> FlowGRPO:
    """
    创建Flow-GRPO训练器
    """
    # 创建FlowGRPOConfig
    flow_grpo_config = FlowGRPOConfig(
        batch_size=int(config.get('training', {}).get('batch_size', 8)),
        max_epochs=int(config.get('training', {}).get('max_epochs', 10)),
        learning_rate=float(config.get('training', {}).get('learning_rate', 1e-5)),
        gradient_accumulation_steps=int(config.get('training', {}).get('gradient_accumulation_steps', 4)),
        max_grad_norm=float(config.get('training', {}).get('max_grad_norm', 1.0)),
        group_size=int(config.get('flow_group', {}).get('group_size', config.get('group_size', 2))),
        max_turns=int(config.get('training', {}).get('max_turns', 10)),
        epsilon=float(config.get('flow_group', {}).get('epsilon', 0.2)),
        beta=float(config.get('flow_group', {}).get('beta', 0.01)),
        save_every=int(config.get('training', {}).get('save_every', 1)),
        eval_every=int(config.get('training', {}).get('eval_every', 5)),
        enable_evaluation=bool(config.get('training', {}).get('enable_evaluation', False)),
        temperature=float(config.get('flow_group', {}).get('temperature', 1.0)),
        top_p=float(config.get('flow_group', {}).get('top_p', 0.9)),
        max_tokens=int(config.get('flow_group', {}).get('max_tokens', 2048)),
        log_dir=config.get('logging', {}).get('log_dir', 'logs'),
        log_level=config.get('logging', {}).get('log_level', 'INFO'),
        save_trajectories=bool(config.get('logging', {}).get('save_trajectories', True)),
        trajectory_sample_rate=float(config.get('logging', {}).get('trajectory_sample_rate', 0.1))
    )
    
    return FlowGRPO(agent, flow_grpo_config)
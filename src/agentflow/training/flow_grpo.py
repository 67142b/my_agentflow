from copy import deepcopy
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
    learning_rate: float = 1e-4
    num_epochs: int = 10
    max_steps: int = 100
    save_steps: int = 50
    eval_steps: int = 50
    warmup_steps: int = 10
    
    # GRPO参数
    clip_ratio: float = 0.2  # PPO裁剪比率
    value_coef: float = 0.5  # 价值函数系数
    entropy_coef: float = 0.1  # 熵系数
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
    
    # 其他训练参数
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
    entropy_coef: float = 0.1  # 熵系数，控制探索程度，防止损失为负
    use_kl_penalty: bool = False  # 是否使用KL散度惩罚（默认不计算）
    
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


class FlowGRPO:
    """Flow-Group训练算法实现类"""
    
    def __init__(self, agent: SimpleAgent, config: FlowGRPOConfig):
        """
        初始化Flow-GRPO训练器
        
        Args:
            agent: SimpleAgent实例
            config: 训练配置
            logger: 日志记录器
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
        #self.ref_policy_state_dict = deepcopy(self.original_planner_model.state_dict())
        
        # 初始化优化器，优化原始模型参数
        self.optimizer = torch.optim.AdamW(
            self.original_planner_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
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
    
    def _compute_ref_logprobs_optimized(self, input_ids: List[int], output_ids: List[int]) -> torch.Tensor:
        """
        使用冻结的参考策略计算log_prob（高度优化版本，大幅减少显存使用）
        
        Args:
            input_ids: 输入token IDs
            output_ids: 输出token IDs
            
        Returns:
            参考策略下的log_prob张量
        """
        if len(output_ids) == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)
        
        # 使用混合精度减少显存
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True), torch.no_grad():
            # 临时加载参考策略权重
            current_state = self.original_planner_model.state_dict()
            self.original_planner_model.load_state_dict(self.ref_policy_state_dict)
            self.original_planner_model.eval()
            
            try:
                model_input = list(input_ids) + list(output_ids[:-1])
                input_tensor = torch.tensor([model_input], dtype=torch.long, device=self.device)
                
                # 只计算需要的输出部分，而不是整个序列
                outputs = self.original_planner_model(input_tensor)
                logits = outputs.logits
                last_logits = logits[:, -len(output_ids):, :]
                logprobs = F.log_softmax(last_logits, dim=-1)
                target = torch.tensor(output_ids, dtype=torch.long, device=self.device)
                ref_logprob = logprobs[0, torch.arange(len(output_ids), device=self.device), target]
                
                # 立即清理中间变量
                del logits, last_logits, logprobs, outputs, input_tensor, target
                
            finally:
                # 确保无论如何都恢复模型权重
                self.original_planner_model.load_state_dict(current_state)
                del current_state
        
        # 返回detach后的结果
        return ref_logprob.detach()
    
    def generate_group_trajectories(self, question: str, ground_truth: str) -> List[Dict[str, Any]]:
        """
        为单个问题生成一组轨迹（高度优化版本，大幅减少显存使用）
        
        Args:
            question: 问题文本
            ground_truth: 标准答案
            
        Returns:
            轨迹列表，每个轨迹包含执行过程和结果
        """
        # 预先清理显存，使用更安全的方式
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()  # 先同步所有CUDA操作
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"清理显存时出错: {e}")
            
        trajectories = []
        
        try:
            # 确定是否使用并发和模型副本
            enable_concurrent = self.config.enable_concurrent if hasattr(self.config, 'enable_concurrent') else True
            num_model_copies = self.config.num_model_copies if hasattr(self.config, 'num_model_copies') else 1
            
            # 减少并发数量以降低显存压力
            if enable_concurrent:
                num_model_copies = min(num_model_copies, 2)  # 限制最大并发数
            
            # 调用SimpleAgent.solve方法，使用新的参数
            #print(f"开始调用SimpleAgent.solve，问题: {question[:500]}...\n====================================")
            #print(f"并发模式: {enable_concurrent}, 模型副本数: {num_model_copies}")
            start_time = time.time()
            result = self.agent.singal_solve(question)
            end_time = time.time()
            #print(f"SimpleAgent.solve调用完毕，用时: {end_time - start_time:.2f}秒\n====================================")
            
            # 处理返回的多条轨迹
            successful_trajectories = result.get("successful_trajectories", [])
            failed_trajectories = result.get("failed_trajectories", [])
            
            #print(f"成功轨迹数: {len(successful_trajectories)}")
            
            # 分批处理轨迹以减少内存峰值
            batch_size = 2  # 每批处理2个轨迹
            
            # 分批处理成功的轨迹
            for batch_start in range(0, len(successful_trajectories), batch_size):
                batch_end = min(batch_start + batch_size, len(successful_trajectories))
                batch_trajectories = successful_trajectories[batch_start:batch_end]
                
                for traj in batch_trajectories:
                    # 使用llm_as_judge方法计算奖励
                    final_answer = traj.get("final_answer", {}).get("final_answer", "")
                    #print(f"评估奖励：问题：\n{question}\n标准答案：{ground_truth}\n待评估答案：{final_answer}\n")
                    
                    try:
                        reward = llm_as_judge(
                            question=question,
                            answer=final_answer,
                            ground_truth=ground_truth
                        )
                    except Exception as e:
                        print(f"计算奖励时出错: {e}")
                        reward = 0.0  # 出错时使用默认奖励
                    
                    #print(f"轨迹奖励: {reward:.4f}")
                    # 添加轨迹ID
                    trajectory_id = traj.get("trajectory_id","")
                    
                    # 构建轨迹数据（只保留必要信息）
                    trajectory = {
                        "trajectory_id": trajectory_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "final_answer": final_answer,
                        "reward": reward,
                        "steps": traj.get("turns", []),  # 使用新的steps字段
                        "success": True
                        # 不再保存完整的轨迹数据以节省内存
                    }
                    
                    trajectories.append(trajectory)
                    
                    # 立即清理临时变量
                    del final_answer, reward, trajectory_id
                
                # 每批次处理后安全清理显存
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()  # 先同步所有CUDA操作
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"清理显存时出错: {e}")
                
                # 清理当前批次
                del batch_trajectories
            
            # 分批处理失败的轨迹
            for batch_start in range(0, len(failed_trajectories), batch_size):
                batch_end = min(batch_start + batch_size, len(failed_trajectories))
                batch_trajectories = failed_trajectories[batch_start:batch_end]
                
                for traj in batch_trajectories:
                    # 添加轨迹ID
                    trajectory_id = traj.get("trajectory_id","")
                    
                    # 构建轨迹数据（只保留必要信息）
                    trajectory = {
                        "trajectory_id": trajectory_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "final_answer": traj.get("final_answer", ""),
                        "reward": 0.0,
                        "steps": traj.get("turns", []),  # 使用新的steps字段
                        "success": False,
                        "error": traj.get("error", "Trajectory failed")
                        # 不再保存完整的轨迹数据以节省内存
                    }
                    
                    trajectories.append(trajectory)
                    
                    # 立即清理临时变量
                    del trajectory_id
                
                # 每批次处理后安全清理显存
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()  # 先同步所有CUDA操作
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"清理显存时出错: {e}")
                
                # 清理当前批次
                del batch_trajectories
            
            # 清理原始轨迹数据
            del successful_trajectories, failed_trajectories, result
                
        except Exception as e:
            # 如果整个solve过程失败，创建失败轨迹
            print(f"生成轨迹时出错: {e}")
            for i in range(self.config.group_size):
                trajectory = {
                    "trajectory_id": f"{question[:50].replace(' ', '_')}_{i}",
                    "question": question,
                    "ground_truth": ground_truth,
                    "final_answer": "",
                    "reward": 0.0,
                    "steps": [],
                    "success": False,
                    "error": str(e)
                }
                trajectories.append(trajectory)
        
        # 最终安全清理内存
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()  # 先同步所有CUDA操作
                torch.cuda.empty_cache()
                # 强制垃圾回收
                import gc
                gc.collect()
            except Exception as e:
                print(f"最终清理显存时出错: {e}")
        
        return trajectories

    def compute_group_normalized_advantages(self, trajectories: List[Dict[str, Any]]) -> List[float]:
        """
        计算组归一化优势
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            每个轨迹的组归一化优势值
        """
        rewards = [t["reward"] for t in trajectories]
        
        # 添加调试信息
        #print(f"\n=== 计算组归一化优势 ===")
        #print(f"轨迹数量: {len(trajectories)}")
        #print(f"原始奖励: {rewards}")
        
        # 检查奖励多样性
        unique_rewards = set(rewards)
        #print(f"唯一奖励值数量: {len(unique_rewards)}")
        #print(f"唯一奖励值: {unique_rewards}")
        
        # 如果所有奖励都是0，返回所有零优势值
        if len(unique_rewards) == 1 and 0.0 in unique_rewards:
            #print("警告: 所有轨迹奖励均为0，所有轨迹答案都不正确，返回零优势值避免向错误方向更新")
            advantages = [0.0 for _ in rewards]
            #print(f"归一化优势: {advantages}")
            #print("=== 优势计算完成 ===\n")
            return advantages
        
        # 如果所有奖励相同但不是0（例如都是1），添加随机扰动
        if len(unique_rewards) <= 1:
            #print("警告: 所有轨迹奖励相同，添加随机扰动以确保训练可以继续")
            # 为每个奖励添加小的随机扰动
            perturbed_rewards = []
            for i, r in enumerate(rewards):
                # 基于轨迹索引添加确定性扰动，确保可重现
                perturbation = (i - len(rewards) / 2) * 0.1
                perturbed_rewards.append(r + perturbation)
            rewards = perturbed_rewards
            #print(f"扰动后奖励: {rewards}")
        
        # 转换为torch张量进行计算
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # 计算均值和标准差
        mean_reward = torch.mean(reward_tensor).item()
        std_reward = torch.std(reward_tensor).item()
        
        #print(f"奖励均值: {mean_reward:.6f}")
        #print(f"奖励标准差: {std_reward:.6f}")
        
        # 如果标准差仍然太小，添加噪声
        if std_reward < 1e-6:
            #print("警告: 奖励标准差过小，添加随机噪声")
            noise = torch.randn_like(reward_tensor) * 0.1
            reward_tensor = reward_tensor + noise
            mean_reward = torch.mean(reward_tensor).item()
            std_reward = torch.std(reward_tensor).item()
            #print(f"添加噪声后奖励均值: {mean_reward:.6f}")
            #print(f"添加噪声后奖励标准差: {std_reward:.6f}")
        
        # 计算组归一化优势
        advantages = [(r - mean_reward) / std_reward for r in reward_tensor.tolist()]
        
        #print(f"归一化优势: {advantages}")
        #print(f"优势范围: [{min(advantages):.6f}, {max(advantages):.6f}]")
        #print("=== 优势计算完成 ===\n")
        
        return advantages
    
    def stage1_build_replay(self,
                            trajectories: List[Dict[str, Any]],
                            advantages: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Build a minimal replay list from sampled trajectories.
        Each item corresponds to one planner turn (one set of planner_output tokens).
        Item fields:
        - 'input_ids': list[int]  (prompt tokens)
        - 'output_ids': list[int] (generated planner tokens)
        - 'old_logprobs': list[float] (saved from sampling)
        - 'advantage': float (group normalized advantage for the whole traj)
        - optional debug fields: 'traj_id', 'turn_id'
        This function runs with NO grad and does not store any tensors that require_grad.
        """
        replay = []

        for traj_idx, (traj, adv) in enumerate(zip(trajectories, advantages)):
            turns = traj.get("steps", [])
            for turn_idx, turn in enumerate(turns):
                # support different naming conventions
                input_ids = turn.get("plan", {}).get("planner_input_ids")
                output_ids = turn.get("plan", {}).get("planner_output_ids")
                old_lp = turn.get("plan", {}).get("logprobs_old")
                if not isinstance(input_ids, list) or not isinstance(output_ids, list):
                    print(f"警告: 轨迹{traj_idx+1}第{turn_idx+1}轮输入或输出ID不是列表类型")
                    continue
                if len(output_ids) == 0:
                    print(f"警告: 轨迹{traj_idx+1}第{turn_idx+1}轮输出ID为空")
                    continue
                # align lengths if necessary
                print(f"轨迹{traj_idx+1}第{turn_idx+1}轮原始旧日志概率长度: {len(old_lp)},输入ID长度: {len(input_ids)},输出ID长度: {len(output_ids)}")
                if old_lp and len(old_lp) != len(output_ids):
                    min_len = min(len(old_lp), len(output_ids))
                    old_lp = old_lp[:min_len]
                    output_ids = output_ids[:min_len]
                # minimal item
                item = {
                    "input_ids": list(input_ids),
                    "output_ids": list(output_ids),
                    "old_logprobs": list(old_lp),
                    "advantage": float(adv),
                    "traj_id": traj_idx,
                    "turn_id": turn_idx
                }
                replay.append(item)
        return replay

    # --- Stage 2: training from replay in memory-friendly micro-batches ---
    def stage2_update_from_replay(self,
                                replay_items: List[Dict[str, Any]],
                                *,
                                micro_batch_size: int = 8,
                                use_amp: Optional[bool] = None,
                                scaler: Optional[torch.cuda.amp.GradScaler] = None,
                                return_loss: bool = False):
        """
        Replay-based training: re-forward on small micro-batches of replay items, compute PPO surrogate + entropy,
        accumulate gradients and optimizer.step() per micro-batch.

        Args:
        replay_items: list produced by stage1_build_replay
        micro_batch_size: number of replay items per backward/step (controls memory)
        use_amp: whether to use autocast (if None, determined from config & device)
        scaler: optional GradScaler (if using amp on CUDA)
        return_loss: if True, returns the final average loss value (float) for logging

        Returns:
        avg_loss_value (float) if return_loss True, else None
        """
        cfg = self.config
        device = self.device
        model = self.original_planner_model
        model.train()

        if use_amp is None:
            use_amp = bool(getattr(cfg, "use_fp16", False) and device.type == "cuda")
        print(f"使用自动混合精度训练: {use_amp}")

        # Shuffle to decorrelate updates (optional)
        # random.shuffle(replay_items)

        n = len(replay_items)
        if n == 0:
            return 0.0 if return_loss else None

        total_loss_accum = 0.0
        total_tokens = 0

        # zero grads
        self.optimizer.zero_grad()

        # micro-batching loop
        for start in range(0, n, micro_batch_size):
            batch = replay_items[start: start + micro_batch_size]
            # compute loss for this micro-batch (sum of turn losses)
            batch_loss = None
            batch_token_count = 0

            if use_amp and device.type == "cuda":
                autocast_ctx = torch.cuda.amp.autocast
            else:
                # fallback context manager that does nothing
                class _noop_ctx:
                    def __enter__(self): return None
                    def __exit__(self, exc_type, exc, tb): return False
                autocast_ctx = _noop_ctx

            with torch.amp.autocast(device_type="cuda"):
                # accumulate loss tensor for micro-batch
                for item in batch:
                    input_ids = item["input_ids"]
                    output_ids = item["output_ids"]
                    old_logprobs = item["old_logprobs"]
                    adv = float(item["advantage"])

                    L = len(output_ids)
                    if L == 0:
                        continue
                    batch_token_count += L

                    # build input tensor: prompt + output[:-1]
                    model_input = list(input_ids) + list(output_ids[:-1])
                    input_tensor = torch.tensor([model_input], dtype=torch.long, device=device)
                    target_tensor = torch.tensor(output_ids, dtype=torch.long, device=device)
                    old_lp_tensor = torch.tensor(old_logprobs, dtype=torch.float32, device=device).detach()

                    outputs = model(input_tensor)
                    logits = outputs.logits  # shape [1, seq_len, V]
                    last_logits = logits[:, -L:, :]  # [1, L, V]
                    logprobs = F.log_softmax(last_logits, dim=-1)  # [1, L, V]

                    idx = torch.arange(L, device=device)
                    new_logprob = logprobs[0, idx, target_tensor]  # [L]

                    # PPO surrogate (token-level)
                    log_r = (new_logprob - old_lp_tensor).clamp(min=-20.0, max=20.0)
                    ratio = torch.exp(log_r)
                    adv_tensor = torch.full_like(new_logprob, float(adv), device=device)

                    clipped_ratio = torch.clamp(ratio, 1.0 - cfg.epsilon, 1.0 + cfg.epsilon)
                    surr1 = ratio * adv_tensor
                    surr2 = clipped_ratio * adv_tensor
                    per_token_surr = torch.min(surr1, surr2)  # [L]
                    turn_mean = per_token_surr.mean()  # scalar

                    policy_loss = -turn_mean  # scalar

                    # entropy: accurate using logits
                    probs = torch.exp(logprobs)  # [1, L, V]
                    entropy_per_token = -(probs * logprobs).sum(dim=-1).squeeze(0)  # [L]
                    entropy_mean = entropy_per_token.mean()  # scalar

                    entropy_term = -cfg.entropy_coef * entropy_mean

                    turn_loss = policy_loss + entropy_term  # scalar tensor
                    print(f"轨迹{start//micro_batch_size+1}第{start%micro_batch_size+1}轮损失: {turn_loss.item():.4f},策略损失: {policy_loss.item():.4f},熵损失: {entropy_term.item():.4f}")

                    # accumulate into batch loss
                    weighted_turn_loss = turn_loss * float(L)  # weight by token count to average later
                    if batch_loss is None:
                        batch_loss = weighted_turn_loss
                    else:
                        batch_loss = batch_loss + weighted_turn_loss

                # end loop over items in micro-batch

            # if no valid tokens in micro-batch
            if batch_loss is None:
                continue

            # normalize batch_loss by total tokens in micro-batch to get average
            batch_loss = batch_loss / float(batch_token_count)

            # backward & step with optional amp/scaler
            if use_amp and device.type == "cuda" and scaler is not None:
                scaler.scale(batch_loss).backward()
                # unscale, clip, step
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

            # zero grads for next micro-batch
            self.optimizer.zero_grad()

            # accumulate numeric stats
            total_loss_accum += float(batch_loss.detach().cpu().item()) * float(batch_token_count)
            total_tokens += batch_token_count

            # free ephemeral tensors (GC hint)
            del batch_loss
            torch.cuda.empty_cache()

        # end micro-batching
        avg_loss = (total_loss_accum / total_tokens) if total_tokens > 0 else 0.0
        if return_loss:
            return avg_loss
        return None
    
    def compute_flow_grpo_loss(
        self,
        trajectories: List[Dict[str, Any]],
        advantages: List[float],
        *,
        accum_steps: int = 1,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> torch.Tensor:
        """
        Memory-optimized Flow-GRPO loss computation.

        Args:
        trajectories: list of trajectory dicts; each traj contains turns, and each turn must have
                        'planner_input_ids' (List[int]), 'planner_output_ids' (List[int]),
                        'logprobs_old' (List[float]).
        advantages: list of scalar advantages (group-normalized) for each trajectory.
        do_backward: if True, function will perform backward() and optimizer.step() internally
                    every `accum_steps` micro-batches (recommended for low memory).
                    If False, it returns a loss tensor (connected graph) and caller should do backward.
        accum_steps: number of turns (micro-batches) to accumulate before stepping (only used if do_backward True).
        scaler: optional GradScaler for AMP (if not None and device is CUDA, used when doing backward).

        Returns:
        If do_backward True: returns a scalar torch.Tensor (detached, for logging) with the average loss value.
        If do_backward False: returns a scalar torch.Tensor (requires_grad=True) representing average loss per token.
        """

        device = self.device
        cfg = self.config
        model = self.original_planner_model
        model.train()

        use_amp = (hasattr(torch.cuda.amp, "autocast") and cfg.use_fp16 and device.type == "cuda")

        # Accumulators (scalars / numbers only)
        total_loss_value = 0.0        # python float sum of (loss.detach().item() * token_count)
        total_token_count = 0         # int

        # If returning graph (do_backward False), we need a tensor to aggregate loss (but avoid huge graph)
        # 初始化统计变量
        total_token_count = 0
        total_loss_value = 0.0
        loss_accum_for_return = None  # 将是一个表示 sum(loss * token_count) 的张量

        # Iterate trajectories -> turns -> per-turn micro-loss
        for traj_idx, (traj, adv_scalar) in enumerate(zip(trajectories, advantages)):
            turns = traj.get("steps", [])
            if not turns:
                continue

            # per-trajectory: collect per-turn means, but we will compute per-turn and combine immediately
            traj_token_count = 0
            traj_loss_value = 0.0  # python float accumulator for this trajectory (loss.detach().item() * tokens)

            for turn_idx, turn in enumerate(turns):
                # robust extraction (support multiple schema)
                input_ids = turn.get("plan", {}).get("planner_input_ids", [])
                output_ids = turn.get("plan", {}).get("planner_output_ids", [])
                old_logprobs_list = turn.get("plan", {}).get("logprobs_old", [])
                print(f"input_ids长度{len(input_ids)},output_ids长度{len(output_ids)},old_logprobs_list长度{len(old_logprobs_list)}")

                # basic validation
                if not isinstance(input_ids, list) or not isinstance(output_ids, list):
                    continue
                if len(output_ids) == 0:
                    continue
                # align lengths
                if old_logprobs_list and (len(old_logprobs_list) != len(output_ids)):
                    min_len = min(len(old_logprobs_list), len(output_ids))
                    old_logprobs_list = old_logprobs_list[:min_len]
                    output_ids = output_ids[:min_len]

                L = len(output_ids)
                traj_token_count += L

                # Make tensors (short-lived)
                input_tensor = torch.tensor([list(input_ids) + list(output_ids[:-1])], dtype=torch.long, device=device)
                target_tensor = torch.tensor(output_ids, dtype=torch.long, device=device)
                old_logprob_tensor = torch.tensor(old_logprobs_list, dtype=torch.float32, device=device).detach()

                # Forward (with grad!). Use AMP autocast if available
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(input_tensor)
                        logits = outputs.logits  # [1, seq_len, vocab]
                        last_logits = logits[:, -L:, :]  # [1, L, V]
                        logprobs = F.log_softmax(last_logits, dim=-1)  # [1, L, V]
                else:
                    outputs = model(input_tensor)
                    logits = outputs.logits
                    last_logits = logits[:, -L:, :]
                    logprobs = F.log_softmax(last_logits, dim=-1)

                idx = torch.arange(L, device=device)
                new_logprob = logprobs[0, idx, target_tensor]  # [L], connected to model

                # --- PPO surrogate (token-level) ---
                # clamp log ratio for numeric stability
                log_r = (new_logprob - old_logprob_tensor).clamp(min=-20.0, max=20.0)
                ratio = torch.exp(log_r)
                adv_tensor = torch.full_like(new_logprob, float(adv_scalar), device=device)

                clipped_ratio = torch.clamp(ratio, 1.0 - cfg.epsilon, 1.0 + cfg.epsilon)
                surr1 = ratio * adv_tensor
                surr2 = clipped_ratio * adv_tensor
                per_token_surr = torch.min(surr1, surr2)  # [L]
                turn_mean = per_token_surr.mean()  # scalar (connected)

                # policy loss for this turn (we minimize -objective)
                turn_policy_loss = -turn_mean  # scalar

                # --- entropy regularizer (correct formula) ---
                # entropy per token: -sum(p * log p) over vocab, then mean over tokens
                probs = torch.exp(logprobs)  # [1, L, V]
                entropy_per_token = -(probs * logprobs).sum(dim=-1).squeeze(0)  # [L]
                entropy_mean = entropy_per_token.mean()  # scalar

                # entropy loss term: -entropy_coef * entropy_mean (we subtract because we maximize entropy)
                entropy_term = -cfg.entropy_coef * entropy_mean

                # total loss for this turn (scalar tensor, graph attached)
                turn_loss = turn_policy_loss + entropy_term
                print(f"轨迹: {traj_idx+1}, 回合: {turn_idx+1}, 损失: {turn_loss.item():.4f}, 策略损失: {turn_policy_loss.item():.4f}, 熵损失: {entropy_term.item():.4f}")

                # --- handle loss accumulation ---
                # 不进行反向传播，累积损失图以便后续统一处理
                # We accumulate weighted sum so returned loss is averaged per token later.
                if loss_accum_for_return is None:
                    loss_accum_for_return = turn_loss * float(L)
                else:
                    loss_accum_for_return = loss_accum_for_return + turn_loss * float(L)

                # store scalar numeric for logging (optional)
                traj_loss_value += turn_loss.detach().cpu().item() * L

                # minimal cleanup: delete big intermediates and let GC free them
                del outputs, logits, last_logits, logprobs, probs, entropy_per_token, new_logprob, old_logprob_tensor
                torch.cuda.empty_cache()

            # end turns loop for this trajectory
            total_loss_value += traj_loss_value
            total_token_count += traj_token_count

        # finalization
        # 返回连接图的平均每token损失
        if total_token_count == 0:
            # nothing, return zero scalar with grad (so caller won't crash)
            return torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
        # loss_accum_for_return is sum(turn_loss * token_count)
        loss_out = loss_accum_for_return / float(total_token_count)
        return loss_out



    
    def train_step(self, batch_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        执行一步训练（高度优化版本，大幅减少显存使用）
        使用两步损失计算方法：
        1. 每个turn分别forward+计算loss，记录loss.detach()*token_count
        2. 缓存"每个turn loss的值+输入输出token"为lightweight replay item
        3. 第二轮重新forward（微批）进行真正的反向传播
        
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
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 初始化统计信息
        batch_losses = []
        batch_rewards = []
        batch_successes = []
        batch_turns = []
        
        # 获取设备和混合精度缩放器
        device = self.device
        scaler = getattr(self, 'scaler', None)
        
        # 清零梯度，因为我们将在外部进行反向传播
        self.optimizer.zero_grad()
        
        # 存储所有样本的replay items，用于统一更新
        all_replay_items = []
        
        # 处理每个样本
        for sample_idx, sample in enumerate(batch_data):
            question = sample["question"]
            ground_truth = sample["ground_truth"]
            
            # 记录每个样本处理前的显存状态
            sample_start_allocated, sample_start_reserved = monitor_memory(f"sample_{sample_idx+1}_start")
            
            # 生成一组轨迹
            #print(f"问题: {question},开始生成轨迹")
            trajectories = self.generate_group_trajectories(question, ground_truth)
            
            # 轨迹生成后清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                after_gen_allocated, after_gen_reserved = monitor_memory(f"sample_{sample_idx+1}_after_gen")
                #logger.info(f"样本 {sample_idx+1} 轨迹生成后显存变化: {after_gen_allocated-sample_start_allocated:.2f}GB")
            
            # 计算组归一化优势
            #print(f"问题: {question}, 开始计算组归一化优势")
            advantages = self.compute_group_normalized_advantages(trajectories)
            #print(f"问题: {question}, 组归一化优势: {advantages}")
            
            # 计算损失前清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Stage 1: 构建replay items（不保存计算图，只保存必要数据）
            replay_items = self.stage1_build_replay(trajectories, advantages)
            all_replay_items.extend(replay_items)
            
            # 收集统计信息
            rewards = [t["reward"] for t in trajectories]
            successes = [t["success"] for t in trajectories]
            turns = [len(t.get("turns", [])) for t in trajectories]
            
            batch_rewards.extend(rewards)
            batch_successes.extend(successes)
            batch_turns.extend(turns)
            
            # 清理轨迹数据以释放显存
            del trajectories, advantages, rewards, successes, turns, replay_items
            
            # 每个样本处理后清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                sample_end_allocated, sample_end_reserved = monitor_memory(f"sample_{sample_idx+1}_end")
                #logger.info(f"样本 {sample_idx+1} 处理后显存变化: {sample_end_allocated-sample_start_allocated:.2f}GB")
        
        # Stage 2: 从replay items进行模型更新（微批次处理）
        if all_replay_items:
            avg_loss = self.stage2_update_from_replay(
                all_replay_items, 
                micro_batch_size=2,  # 自适应微批次大小
                use_amp=(hasattr(torch.cuda.amp, "autocast") and self.config.use_fp16 and device.type == "cuda"),
                scaler=scaler,
                return_loss=True
            )
            batch_losses.append(avg_loss)
        
        # 计算平均损失（用于统计）
        if batch_losses:
            avg_loss = np.mean(batch_losses)
        else:
            avg_loss = 0.0
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_allocated, final_reserved = monitor_memory("train_step_end")
            #logger.info(f"训练步骤完成显存变化: {final_allocated-initial_allocated:.2f}GB")
        
        # 返回统计信息
        stats = {
            "loss": avg_loss,
            "avg_reward": np.mean(batch_rewards) if batch_rewards else 0.0,
            "success_rate": np.mean(batch_successes) if batch_successes else 0.0,
            "avg_turns": np.mean(batch_turns) if batch_turns else 0.0,
            "memory_allocated": final_allocated if torch.cuda.is_available() else 0,
            "memory_reserved": final_reserved if torch.cuda.is_available() else 0
        }
        
        # 清理中间变量
        del all_replay_items, batch_losses, batch_rewards, batch_successes, batch_turns
        
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
        
        self.original_planner_model.train()
        
        # 打乱数据
        random.shuffle(train_data)
        
        # 分批处理
        num_batches = len(train_data) // self.config.batch_size
        epoch_stats = {
            "loss": [],
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
        评估模型，严格按照训练流程的逻辑采样并计算评估指标
        
        Args:
            val_data: 验证数据
            
        Returns:
            评估统计信息
        """
        self.original_planner_model.eval()
        
        eval_rewards = []
        eval_successes = []
        eval_turns = []
        eval_correctness = []  # 新增：正确率指标
        
        with torch.no_grad():
            for sample in tqdm(val_data, desc="Evaluating"):
                question = sample["question"]
                ground_truth = sample["ground_truth"]
                
                # 生成一组轨迹（与训练流程完全一致）
                trajectories = self.generate_group_trajectories(question, ground_truth)
                
                # 计算组归一化优势（与训练流程完全一致）
                advantages = self.compute_group_normalized_advantages(trajectories)
                
                # 收集统计信息（与训练流程完全一致）
                rewards = [t["reward"] for t in trajectories]
                successes = [t["success"] for t in trajectories]
                #turns = [len(t.get("steps", [])) for t in trajectories]
                
                # 计算正确率（基于奖励阈值判断）
                correctness = [1 if reward == 1 else 0 for reward in rewards]  # 假设奖励=1为正确
                
                eval_rewards.extend(rewards)
                eval_successes.extend(successes)
                eval_correctness.extend(correctness)
                
                # 清理显存
                del trajectories, advantages, rewards, successes, turns, correctness
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 计算平均统计（与训练流程指标一致）
        stats = {
            "val_avg_reward": np.mean(eval_rewards),
            "val_success_rate": np.mean(eval_successes),  # 成功率
            "val_accuracy": np.mean(eval_correctness)    # 正确率
        }
        
        # 清理评估数据
        del eval_rewards, eval_successes, eval_turns, eval_correctness
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
            "model_state_dict": self.original_planner_model.state_dict(),
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
        
        self.original_planner_model.load_state_dict(checkpoint["model_state_dict"])
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
    
    Args:
        agent: SimpleAgent实例
        config: 配置字典
        
    Returns:
        FlowGRPO实例
    """
    # 提取训练配置
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    flow_group_config = config.get('flow_group', {})
     
    # 创建FlowGRPOConfig
    flow_grpo_config = FlowGRPOConfig(
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
        trajectory_sample_rate=float(config.get('logging', {}).get('trajectory_sample_rate', 0.1))
    )
    
    # 创建FlowGRPO实例
    trainer = FlowGRPO(agent, flow_grpo_config)
    
    return trainer
import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from datetime import datetime

from src.agentflow.simple_agent import SimpleAgent
from src.agentflow.utils.reward import compute_reward, compute_batch_rewards
from src.agentflow.utils.logger import TrainingLogger


class ModelEvaluator:
    """模型评估器，用于评估训练过程中的模型性能"""
    
    def __init__(self, 
                 agent: SimpleAgent,
                 logger: Optional[TrainingLogger] = None,
                 save_dir: str = "./eval_results"):
        """
        初始化评估器
        
        Args:
            agent: 要评估的智能体
            logger: 日志记录器
            save_dir: 评估结果保存目录
        """
        self.agent = agent
        self.logger = logger
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 评估指标历史
        self.evaluation_history = {
            "timestamp": [],
            "epoch": [],
            "accuracy": [],
            "reward": [],
            "success_rate": [],
            "avg_tokens": [],
            "kl_divergence": []
        }
    
    def evaluate(self, 
                eval_data: List[Dict[str, Any]], 
                epoch: Optional[int] = None,
                save_results: bool = True) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            eval_data: 评估数据
            epoch: 当前训练轮次
            save_results: 是否保存评估结果
            
        Returns:
            评估指标字典
        """
        print(f"Evaluating model on {len(eval_data)} samples...")
        
        # 初始化统计变量
        total_reward = 0.0
        correct_predictions = 0
        total_tokens = 0
        successful_runs = 0
        kl_divergences = []
        
        # 记录评估开始
        if self.logger:
            self.logger.log_info(f"Starting evaluation on {len(eval_data)} samples")
        
        # 批量评估
        batch_size = 8
        for i in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating"):
            batch = eval_data[i:i+batch_size]
            
            # 生成响应
            try:
                # 记录输入问题
                questions = [item.get("question", "") for item in batch]
                
                # 执行推理
                results = []
                for j, item in enumerate(batch):
                    try:
                        # 使用智能体解决问题
                        result = self.agent.solve(item.get("question", ""))
                        results.append({
                            "question": item.get("question", ""),
                            "expected_answer": item.get("answer", ""),
                            "generated_answer": result,
                            "success": True
                        })
                        successful_runs += 1
                    except Exception as e:
                        results.append({
                            "question": item.get("question", ""),
                            "expected_answer": item.get("answer", ""),
                            "generated_answer": f"Error: {str(e)}",
                            "success": False,
                            "error": str(e)
                        })
                        if self.logger:
                            self.logger.log_error(f"Evaluation error for sample {i+j}: {str(e)}")
                
                # 计算奖励
                rewards = compute_batch_rewards(results)
                
                # 更新统计
                for j, (result, reward) in enumerate(zip(results, rewards)):
                    total_reward += reward
                    total_tokens += len(result["generated_answer"].split())
                    
                    # 检查答案是否正确
                    if self._is_correct_answer(
                        result["expected_answer"], 
                        result["generated_answer"]
                    ):
                        correct_predictions += 1
                    
                    # 计算KL散度（如果有参考概率）
                    if "reference_probs" in batch[j] and "generated_probs" in result:
                        kl_div = self._compute_kl_divergence(
                            batch[j]["reference_probs"], 
                            result["generated_probs"]
                        )
                        kl_divergences.append(kl_div)
                
            except Exception as e:
                print(f"Batch evaluation error: {str(e)}")
                if self.logger:
                    self.logger.log_error(f"Batch evaluation error: {str(e)}")
                continue
        
        # 计算平均指标
        num_samples = len(eval_data)
        avg_reward = total_reward / num_samples
        accuracy = correct_predictions / num_samples
        success_rate = successful_runs / num_samples
        avg_tokens = total_tokens / num_samples
        avg_kl_div = np.mean(kl_divergences) if kl_divergences else 0.0
        
        # 构建评估结果
        eval_results = {
            "accuracy": accuracy,
            "reward": avg_reward,
            "success_rate": success_rate,
            "avg_tokens": avg_tokens,
            "kl_divergence": avg_kl_div,
            "num_samples": num_samples,
            "timestamp": datetime.now().isoformat()
        }
        
        # 更新历史记录
        self.evaluation_history["timestamp"].append(datetime.now().isoformat())
        self.evaluation_history["epoch"].append(epoch if epoch is not None else -1)
        self.evaluation_history["accuracy"].append(accuracy)
        self.evaluation_history["reward"].append(avg_reward)
        self.evaluation_history["success_rate"].append(success_rate)
        self.evaluation_history["avg_tokens"].append(avg_tokens)
        self.evaluation_history["kl_divergence"].append(avg_kl_div)
        
        # 打印评估结果
        print(f"Evaluation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Success Rate: {success_rate:.4f}")
        print(f"  Average Tokens: {avg_tokens:.2f}")
        print(f"  KL Divergence: {avg_kl_div:.6f}")
        
        # 记录评估结果
        if self.logger:
            self.logger.log_evaluation(epoch if epoch is not None else 0, eval_results)
        
        # 保存评估结果
        if save_results:
            self._save_evaluation_results(eval_results, epoch)
        
        return eval_results
    
    def _is_correct_answer(self, expected: str, generated: str) -> bool:
        """
        检查生成的答案是否正确
        
        Args:
            expected: 期望答案
            generated: 生成的答案
            
        Returns:
            是否正确
        """
        # 简单的字符串匹配，实际应用中可能需要更复杂的逻辑
        # 提取数字答案
        import re
        
        # 尝试从期望答案中提取数字
        expected_numbers = re.findall(r'-?\d+\.?\d*', expected)
        # 尝试从生成答案中提取数字
        generated_numbers = re.findall(r'-?\d+\.?\d*', generated)
        
        if expected_numbers and generated_numbers:
            # 比较数字
            try:
                expected_val = float(expected_numbers[0])
                generated_val = float(generated_numbers[0])
                return abs(expected_val - generated_val) < 1e-6
            except ValueError:
                pass
        
        # 如果没有数字或数字转换失败，进行字符串比较
        return expected.strip().lower() in generated.strip().lower()
    
    def _compute_kl_divergence(self, p_dist: List[float], q_dist: List[float]) -> float:
        """
        计算两个概率分布之间的KL散度
        
        Args:
            p_dist: 参考概率分布
            q_dist: 生成概率分布
            
        Returns:
            KL散度值
        """
        # 确保概率分布长度一致
        min_len = min(len(p_dist), len(q_dist))
        p = p_dist[:min_len]
        q = q_dist[:min_len]
        
        # 添加小常数避免log(0)
        eps = 1e-10
        p = np.array(p) + eps
        q = np.array(q) + eps
        
        # 归一化
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # 计算KL散度
        kl_div = np.sum(p * np.log(p / q))
        return kl_div
    
    def _save_evaluation_results(self, 
                               eval_results: Dict[str, Any], 
                               epoch: Optional[int] = None):
        """
        保存评估结果
        
        Args:
            eval_results: 评估结果
            epoch: 当前训练轮次
        """
        # 创建文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if epoch is not None:
            filename = f"eval_epoch_{epoch}_{timestamp}.json"
        else:
            filename = f"eval_{timestamp}.json"
        
        # 保存评估结果
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # 保存评估历史
        history_filepath = os.path.join(self.save_dir, "evaluation_history.json")
        with open(history_filepath, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        
        print(f"Evaluation results saved to: {filepath}")
    
    def compare_models(self, 
                      model1_path: str, 
                      model2_path: str, 
                      eval_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        比较两个模型的性能
        
        Args:
            model1_path: 第一个模型路径
            model2_path: 第二个模型路径
            eval_data: 评估数据
            
        Returns:
            比较结果
        """
        print(f"Comparing models: {model1_path} vs {model2_path}")
        
        # 加载第一个模型
        print("Loading first model...")
        self.agent.load_model(model1_path)
        results1 = self.evaluate(eval_data, save_results=False)
        
        # 加载第二个模型
        print("Loading second model...")
        self.agent.load_model(model2_path)
        results2 = self.evaluate(eval_data, save_results=False)
        
        # 计算改进
        improvements = {}
        for key in ["accuracy", "reward", "success_rate"]:
            if key in results1 and key in results2:
                improvement = results2[key] - results1[key]
                improvements[key] = {
                    "model1": results1[key],
                    "model2": results2[key],
                    "improvement": improvement,
                    "improvement_pct": (improvement / results1[key]) * 100 if results1[key] != 0 else float('inf')
                }
        
        # 构建比较结果
        comparison_results = {
            "model1_path": model1_path,
            "model2_path": model2_path,
            "model1_results": results1,
            "model2_results": results2,
            "improvements": improvements,
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存比较结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.json"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"Model comparison results saved to: {filepath}")
        
        # 打印比较结果
        print("\nModel Comparison Results:")
        for key, values in improvements.items():
            print(f"  {key}:")
            print(f"    Model 1: {values['model1']:.4f}")
            print(f"    Model 2: {values['model2']:.4f}")
            print(f"    Improvement: {values['improvement']:.4f} ({values['improvement_pct']:.2f}%)")
        
        return comparison_results


def create_evaluator_from_config(config: Dict[str, Any], 
                                agent: SimpleAgent,
                                logger: Optional[TrainingLogger] = None) -> ModelEvaluator:
    """
    从配置创建评估器
    
    Args:
        config: 配置字典
        agent: 智能体实例
        logger: 日志记录器
        
    Returns:
        评估器实例
    """
    # 获取评估配置
    eval_config = config.get("evaluation", {})
    
    # 创建评估器
    evaluator = ModelEvaluator(
        agent=agent,
        logger=logger,
        save_dir=eval_config.get("save_dir", "./eval_results")
    )
    
    return evaluator
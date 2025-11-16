"""
奖励计算模块，用于评估智能体回答的质量
"""

import re
import json
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def extract_answer(text: str) -> str:
    """
    从文本中提取答案
    
    Args:
        text: 输入文本
        
    Returns:
        提取的答案
    """
    # 尝试从不同模式中提取答案
    patterns = [
        r'答案[：:]\s*(.+?)(?:\n|$)',  # 答案：xxx
        r'答案[是]?\s*[：:]\s*(.+?)(?:\n|$)',  # 答案是：xxx
        r'最终答案[：:]\s*(.+?)(?:\n|$)',  # 最终答案：xxx
        r'结果[：:]\s*(.+?)(?:\n|$)',  # 结果：xxx
        r'结论[：:]\s*(.+?)(?:\n|$)',  # 结论：xxx
        r'<answer>(.+?)</answer>',  # <answer>xxx</answer>
        r'Answer[：:]\s*(.+?)(?:\n|$)',  # Answer: xxx
        r'The answer[ is]*[：:]\s*(.+?)(?:\n|$)',  # The answer is: xxx
        r'(.+)',  # 默认匹配整个文本
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # 如果答案太长，尝试进一步提取
            if len(answer) > 100:
                # 尝试提取数字或简短答案
                number_match = re.search(r'[-+]?\d*\.?\d+', answer)
                if number_match:
                    return number_match.group(0)
            return answer
    
    # 如果没有匹配到任何模式，返回原始文本
    return text.strip()


def normalize_answer(answer: str) -> str:
    """
    标准化答案格式
    
    Args:
        answer: 原始答案
        
    Returns:
        标准化后的答案
    """
    # 移除多余空格
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # 移除标点符号（保留数字、字母和中文）
    answer = re.sub(r'[^\w\s\u4e00-\u9fff]', '', answer)
    
    # 转换为小写
    answer = answer.lower()
    
    return answer


def compute_numerical_similarity(answer1: str, answer2: str) -> float:
    """
    计算数值型答案的相似度
    
    Args:
        answer1: 答案1
        answer2: 答案2
        
    Returns:
        相似度分数 (0-1)
    """
    # 尝试提取数字
    numbers1 = re.findall(r'[-+]?\d*\.?\d+', answer1)
    numbers2 = re.findall(r'[-+]?\d*\.?\d+', answer2)
    
    if not numbers1 or not numbers2:
        return 0.0
    
    # 转换为浮点数
    try:
        num1 = float(numbers1[0])
        num2 = float(numbers2[0])
        
        # 计算相对误差
        if num1 == 0 and num2 == 0:
            return 1.0
        elif num1 == 0 or num2 == 0:
            return 0.0
        else:
            relative_error = abs(num1 - num2) / max(abs(num1), abs(num2))
            similarity = max(0.0, 1.0 - relative_error)
            return similarity
    except:
        return 0.0


def compute_text_similarity(answer1: str, answer2: str) -> float:
    """
    计算文本型答案的相似度
    
    Args:
        answer1: 答案1
        answer2: 答案2
        
    Returns:
        相似度分数 (0-1)
    """
    # 标准化答案
    norm_answer1 = normalize_answer(answer1)
    norm_answer2 = normalize_answer(answer2)
    
    # 完全匹配
    if norm_answer1 == norm_answer2:
        return 1.0
    
    # 包含关系
    if norm_answer1 in norm_answer2 or norm_answer2 in norm_answer1:
        return 0.8
    
    # 计算词汇重叠度
    words1 = set(norm_answer1.split())
    words2 = set(norm_answer2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    jaccard_similarity = len(intersection) / len(union)
    
    return jaccard_similarity


def compute_reward(question: str, answer: str, ground_truth: str) -> float:
    """
    计算智能体回答的奖励分数
    
    Args:
        question: 问题
        answer: 智能体回答
        ground_truth: 标准答案
        
    Returns:
        奖励分数 (0-1)
    """
    # 提取答案
    extracted_answer = extract_answer(answer)
    
    # 如果答案为空，返回0分
    if not extracted_answer:
        return 0.0
    
    # 计算数值相似度
    numerical_similarity = compute_numerical_similarity(extracted_answer, ground_truth)
    
    # 计算文本相似度
    text_similarity = compute_text_similarity(extracted_answer, ground_truth)
    
    # 综合评分
    # 如果数值相似度高，优先使用数值相似度
    if numerical_similarity > 0.8:
        reward = numerical_similarity
    else:
        # 否则使用文本相似度
        reward = text_similarity
    
    # 确保奖励在0-1范围内
    reward = max(0.0, min(1.0, reward))
    
    return reward


def llm_as_judge(question: str, answer: str, ground_truth: str, model_name: str = None) -> float:
    """
    使用大模型作为裁判评估答案质量
    
    Args:
        question: 问题
        answer: 生成的答案
        ground_truth: 标准答案
        model_name: 模型名称，默认使用配置中的verifier模型
        
    Returns:
        float: 评估分数 (0-1)
    """
    try:
        # 导入必要的库
        import yaml
        import os
        from dashscope import Generation
        
        # 加载配置文件
        config_path = "src/configs/config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 从配置中获取API密钥和模型信息
        api_key = config["model"]["dashscope_api_key"]
        base_url = config["model"]["dashscope_base_url"]
        
        # 使用指定的模型名称或默认使用verifier模型
        if model_name is None:
            model_name = config["model"]["verifier_model_name"]
        
        # 设置API密钥
        Generation.api_key = api_key
        # 构建评估提示词
        prompt = f"""请评估以下答案的质量，给出0或1的分数：

问题: {question}
标准答案: {ground_truth}
待评估答案: {answer}

请仅返回一个数字分数（0或1），不要包含任何解释或文字。0表示答案错误，1表示答案正确。
现在给出你的评估分数：
"""
        
        # 调用API
        response = Generation.call(
            model=model_name,
            prompt=prompt,
            max_tokens=10,
            temperature=0.0
        )
        
        if response.status_code == 200:
            # 提取分数
            score_text = response.output.text.strip()
            print(f"奖励模型返回的分数文本: \n{score_text}\n")
            try:
                score = float(score_text)
                # 确保分数在0-1范围内
                score = max(0.0, min(1.0, score))
                return score
            except ValueError:
                # 如果无法解析为数字，返回默认分数
                return 0.5
        else:
            # API调用失败，使用默认分数
            return 0.5
            
    except Exception as e:
        print(f"LLM评估失败: {e}")
        # 出错时回退到传统方法
        return compute_reward(question, answer, ground_truth)


def compute_batch_rewards(batch_data: list) -> list:
    """
    批量计算奖励分数
    
    Args:
        batch_data: 包含问题和答案的批量数据，每个元素应包含question和generated_answer字段
        
    Returns:
        奖励分数列表
    """
    rewards = []
    
    for item in batch_data:
        question = item.get("question", "")
        generated_answer = item.get("generated_answer", "")
        expected_answer = item.get("expected_answer", "")
        
        reward = compute_reward(question, generated_answer, expected_answer)
        rewards.append(reward)
    
    return rewards


def batch_compute_reward(questions: list, answers: list, ground_truths: list) -> list:
    """
    批量计算奖励分数
    
    Args:
        questions: 问题列表
        answers: 回答列表
        ground_truths: 标准答案列表
        
    Returns:
        奖励分数列表
    """
    rewards = []
    
    for q, a, gt in zip(questions, answers, ground_truths):
        reward = compute_reward(q, a, gt)
        rewards.append(reward)
    
    return rewards


if __name__ == "__main__":
    # 测试奖励计算
    question = "用 [1,1,6,9] 四则运算得到 24"
    ground_truth = "24"
    
    test_answers = [
        "24",
        "答案是：24",
        "最终答案是24",
        "(1+1)*9+6 = 24",
        "25",
        "答案是：25",
        "我不知道答案"
    ]
    
    for answer in test_answers:
        reward = compute_reward(question, answer, ground_truth)
        print(f"Answer: {answer}, Reward: {reward:.4f}")
# AGENTFLOW — 训练与计算细则（研究者版）
> 目标：精确、可复现地说明 AGENTFLOW 中 **Planner** 的 in-the-flow on-policy 优化（Flow-GRPO）的训练流程与计算细节：轨迹需要保存哪些信息、以什么格式、如何从中计算 advantage / importance ratios / clipped objective / KL penalty / group-normalization / 广播机制等。

---

## 目录
1. 高层训练流程（算法级）  
2. 系统模块与接口（Planner / Executor / Verifier / Generator / Memory）  
3. 轨迹数据结构（rollout schema）  
4. 奖励计算（LLM judge）  
5. Flow-GRPO 目标与公式分解  
6. 组归一化优势计算  
7. 广播机制  
8. 必须记录的中间量  
9. 训练超参数  
10. 从轨迹到梯度的完整计算步骤  
11. Memory 格式与解析  
12. 实验性结论与复现提示  
13. 可选的度量与诊断  
14. 最小可运行示例  
15. 实现检查清单

---

## 1. 高层训练流程（Algorithm 1 概要）
1. 初始化：t ← 1, M₁ ← q。
2. 使用行为策略 π₍θₒₗd₎ 做 on-policy rollout：  
   - 采样 aₜ ∼ π₍θₒₗd₎(·|q,K,Mₜ)  
   - 工具执行 eₜ ∼ E(eₜ|aₜ,K)  
   - 验证 vₜ ∼ V(vₜ|q,eₜ,Mₜ)  
   - 更新记忆 Mₜ₊₁ = fₘₑₘ(Mₜ,aₜ,eₜ,vₜ)
3. 当 verifier 终止或到 Tmax 停止，Generator 生成最终答案 o。  
4. 通过 LLM judge 得到最终 reward R(τ)。  
5. 对 group 采样（G 条轨迹）做 group normalization 并优化 Flow-GRPO。

---

## 2. 模块与接口
**Planner (π_θ)**：输入 `(q,K,Mₜ)`，输出 token 序列（并返回 logprobs）。  
**Executor (E)**：输入 action，返回结构化工具执行结果。  
**Verifier (V)**：输入 `(q,eₜ,Mₜ)`，输出 vt∈{0,1} 与说明。  
**Generator (G)**：输入 `(q,M_T)`，输出最终回答 o。  
**Memory (M)**：由多个 turn 条目构成的结构化记录。

---

## 3. 轨迹数据结构（JSON Schema）
```json
{
  "query": "<string>",
  "tools_metadata": {...},
  "turns": [
    {
      "t": 1,
      "memory_before": "<structured>",
      "action": {
        "text": "<action>",
        "tokens": [int],
        "token_logprobs": [float]
      },
      "executor_result": {...},
      "verifier": {"vt": 0_or_1, "analysis": "..."},
      "memory_after": "<structured>"
    }
  ],
  "final_solution": "<string>",
  "judge_result": {"verdict": true_or_false, "analysis": "..."},
  "R_tau": 0_or_1
}
```

---

## 4. 奖励计算
使用 LLM judge模板输出：  
```
<analysis>: ...
<true false>: "True" or "False"
```
True→1, False→0，得到 R(τ)，并广播到每个 turn。

---

## 5. Flow-GRPO 目标函数
\(
JFlow−GRPO​(θ)=E(q,y∗)∼D,{τi​}i=1G​∼πθold​​​[G1​i=1∑G​Ti​1​t=1∑Ti​​∣ati​∣1​j=1∑∣ati​∣​min{ρj(i,t)​Ati​, clip(ρj(i,t)​,1−ϵ,1+ϵ)Ati​}−βDKL​(πθ​∥πref​)]
\)

其中  
ρ_j = exp(log π_θ - log π_{θold})  
A_i 为组归一化优势（见 §6）。

---

## 6. 组归一化优势计算
对 group 中 G 条轨迹：  
\(
A^i = (R_i - mean(R)) / std(R)
\)

例：R=[1,0,1,0] → mean=0.5,std=0.5 → A=[1,-1,1,-1]

---

## 7. 广播机制
对轨迹 i 的每一 turn t：`A_t^i = A^i`  
保证每个 token 都能使用相同 advantage。

---

## 8. 必须记录的中间量
- state (q,K,Mₜ)  
- action.token_logprobs (π_{θold})  
- executor_result, verifier, memory_after  
- final_solution, judge_result  
- rollout_id, seed, timestamps

---

## 9. 训练超参数
| 参数 | 值 |
|------|----|
| 学习率 | 1e-6 |
| β (KL) | 0.001 |
| Tmax | 3 |
| batch size | 32 |
| group size G | 8 |
| ε (clip) | 0.2 |
| planner temp | 0.5 (train), 0.7 (eval) |

---

## 10. 从轨迹到梯度的完整计算步骤
1. 提取每条轨迹的 R_i。  
2. 计算 group mean/std 得到 A_i。  
3. 对每个 token：计算  
   `ρ_j = exp(logprob_new - logprob_old)`  
   `L_clip_j = min(ρ_j*A_i, clip(ρ_j,1-ε,1+ε)*A_i)`  
4. 对 tokens 取均值 → turn 均值 → group 均值。  
5. 计算 KL penalty，合成总 loss = - (J_clip - β*KL)。  
6. 反向传播更新 θ。

---

## 11. Memory 格式
```json
"memory": {
  "initial_query": "...",
  "entries": [
    {
      "turn": 1,
      "sub_goal": "...",
      "tool_name": "...",
      "command": "...",
      "result": "...",
      "verifier": {"vt": 1, "analysis": "..."},
      "timestamp": "..."
    }
  ]
}
```

---

## 12. 实验性结论与复现提示
- Flow-GRPO 提升工具使用正确率与规划稳定性。  
- 工具 LLM 温度 0.0 保持确定性输出。  
- Planner 输出更简洁（token 减少）。

---

## 13. 可选诊断指标
- 工具错误率 = 错误调用数 / 总调用数  
- 平均 turn 数 T  
- 响应 token 长度分布

---

## 14. 最小运行示例
1. 对 D_small 采样轨迹（含 token_logprobs_old）。  
2. judge 得 R, 组归一化 A。  
3. 按上式计算 loss 并更新。

---

## 15. 实现检查清单
- [x] 保存行为策略 logprobs  
- [x] 完整 memory 序列化  
- [x] LLM judge 二值输出  
- [x] 组归一化防除零（+ε）  
- [x] β, lr, ε 设置正确

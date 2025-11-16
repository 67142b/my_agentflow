import signal
import sys
import logging
import traceback
import os
from src.agentflow.simple_agent import SimpleAgent

# 输出进程ID
print(f"当前进程ID: {os.getpid()}")
print(f"父进程ID: {os.getppid() if hasattr(os, 'getppid') else 'N/A'}")

config_path = "src/configs/config.yaml"
agent = SimpleAgent(config_path=config_path)
question_list = [
    "编写一个Python函数，实现斐波那契数列的前20项，并计算它们的和",
    "who played guitar on the top gun anthem?",
    "who was the lead singer of ram jam",
    "Evaluate the limit: $$\\lim_{(x,y)\\to(0,0)} (x^2+y^2)^{x^2y^2}.$$ Determine the correct value of the limit."
]

# 选择要测试的问题索引
a = 3
try:
    result = agent.solve(question_list[a])
    print(f"成功轨迹数：{len(result.get('successful_trajectories', []))}")
    for trajectory in result.get("successful_trajectories", []):
        print(f"\n成功轨迹 第{a+1}个问题: {trajectory.get('final_answer', {}).get('final_answer', '')}\n")
    print("执行成功！")
except Exception as e:
    print(f"执行过程中出现错误: {str(e)}")
    print(f"错误详情: {traceback.format_exc()}")

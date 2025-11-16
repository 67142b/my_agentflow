"""
Python Executor Tool
Python代码执行工具 - 重构版本，修复功能缺陷并增强安全性
现在支持通过DashScope API调用Qwen-Plus模型生成代码
"""

import subprocess
import sys
import tempfile
import os
import json
import re
import time
import signal
import yaml
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from .base import BaseTool, ToolResult
from ...LLM_engine import get_llm_engine


class PythonExecutorTool(BaseTool):
    """Python代码执行工具 - 重构版本，支持通过DashScope API生成代码"""
    
    def __init__(self, 
                 timeout: int = 10, 
                 max_output: int = 1000, 
                 allow_network: bool = False,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model_name: str = "qwen-plus",
                 config_path: Optional[str] = None,
                 llm_engine=None):
        super().__init__(
            name="python_executor",
            description="Execute Python code for calculations and computations with enhanced security"
        )
        self.timeout = timeout
        self.max_output = max_output
        self.allow_network = allow_network
        self.model_name = model_name
        
        # 从配置文件加载API参数
        if not api_key or not base_url:
            config = self._load_config(config_path)
            if not api_key:
                api_key = config.get("model", {}).get("dashscope_api_key", "")
            if not base_url:
                base_url = config.get("model", {}).get("dashscope_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            
            # 从配置文件中更新model_name
            if config.get("model", {}).get("python_executor"):
                self.model_name = config.get("model", {}).get("python_executor").get("model_name", self.model_name)
        
        # 初始化LLM引擎
        # 使用传入的llm_engine实例或初始化新的实例
        if llm_engine is not None:
            self.llm_engine = llm_engine
        else:
            self.llm_engine = get_llm_engine(config_path)

    def get_parameters(self) -> Dict[str, Any]:
        """
        获取工具参数定义
        
        Returns:
            Dict: 参数定义，包括必需参数和可选参数
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Python代码或需要执行的数学表达式"
                },
                "context": {
                    "type": "string",
                    "description": "执行上下文或额外信息"
                }
            },
            "required": ["query"]
        }
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        从配置文件加载配置
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
            
        Returns:
            Dict: 配置字典
        """
        # 如果没有提供配置路径，使用默认路径
        if not config_path:
            # 尝试多个可能的配置文件路径
            project_root = Path(__file__).parent.parent.parent.parent
            possible_paths = [
                os.path.join(project_root, "src", "configs", "config.yaml"),
                os.path.join(project_root, "config.yaml"),
                "src/configs/config.yaml",
                "config.yaml"
            ]
        else:
            possible_paths = [config_path]
        
        # 尝试加载配置文件
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    print(f"✅ 成功从 {path} 加载配置")
                    return config
        
        print("⚠️  未找到配置文件，使用空配置")
        return {}
    
    def _call_llm_engine(self, prompt: str) -> Dict[str, Any]:
        """
        使用LLM引擎生成代码，共享planner的权重和分词器
        
        Args:
            prompt: 提示词
            
        Returns:
            Dict: API响应结果
        """
        try:
            # 使用LLM引擎生成代码，共享planner的权重和分词器
            response = self.llm_engine.generate(
                model_source="local",
                model_name="python_executor",  # 使用python_executor作为模型类型，会自动共享planner的权重
                prompt=prompt,
                max_new_tokens=512,  # 使用max_new_tokens而不是max_tokens
                temperature=0.5  # 使用确定性模式生成代码
            )
            
            result = response
            
            return {
                "success": True,
                "result": result,
                "usage": {}
            }
        except Exception as e:
            return {
                "success": False,
                "result": "",
                "error": f"LLM engine error: {str(e)}"
            }
    
    def _generate_code(self, query: str) -> Optional[str]:
        """
        使用Qwen-Plus模型生成Python代码
        
        Args:
            query: 用户查询
            
        Returns:
            str: 生成的Python代码，如果生成失败则返回None
        """
        # 构建代码生成提示词
        prompt = f"""You are a Python code generator. Generate Python code to solve the given problem.

Requirements:
1. Generate only Python code and its execution result, no explanations
2. Use print() statements to output the result
3. Don't use external libraries unless absolutely necessary
4. Keep the code simple and focused on the problem


Problem: {query}

Generate only the Python code, no explanations or reasoning:"""

        # 调用LLM引擎生成代码
        api_result = self._call_llm_engine(prompt)
        
        if not api_result["success"]:
            print(f"Code generation failed: {api_result.get('error', 'Unknown error')}")
            return None
        
        # 提取代码块
        return api_result["result"]
        code_match = re.search(r'```python\n(.*?)\n```', generated_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 如果没有找到代码块，尝试提取整个文本作为代码
        code_match = re.search(r'```\n(.*?)\n```', generated_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # 如果仍然没有找到代码块，返回整个文本
        return generated_text.strip() if generated_text.strip() else None
    
    def execute(self, query: str, context: Optional[str] = None) -> ToolResult:
        """执行Python代码 - 修改版本，支持代码生成"""
        start_time = time.time()
        generated_code = self._generate_code(query)
        end_time = time.time()
        execution_time = end_time - start_time
        return ToolResult(
            success=True,
            result=generated_code,
            metadata={
                "execution_time": execution_time,
                "code_length": len(generated_code),
            }
        )
        
        # 检查查询是否是无效的占位符文本
        if query == "No specific context provided" or not query or query.strip() == "":
            return ToolResult(
                success=False,
                result="",
                error="无效的查询内容，无法执行Python代码。请提供具体的Python代码或数学表达式。"
            )
        
        # 如果查询中没有代码，尝试生成代码
        code = self._extract_code(query)
        
        if not code:
            # 尝试生成代码
            generated_code = self._generate_code(query)
            
            if not generated_code:
                return ToolResult(
                    success=False,
                    result="",
                    error="Failed to generate Python code for the query"
                )
            
            code = generated_code
            
            # 安全检查
            security_check = self._security_check(code)
            if not security_check[0]:
                return ToolResult(
                    success=False,
                    result="",
                    error=f"Security check failed: {security_check[1]}"
                )
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                # 添加安全限制和导入限制
                safe_code = self._prepare_safe_code(code)
                f.write(safe_code)
                temp_file = f.name
            
            # 执行代码 - 修复Windows兼容性问题
            if os.name == 'nt':  # Windows系统
                # Windows不支持preexec_fn参数
                process = subprocess.Popen(
                    [sys.executable, temp_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            else:  # Unix/Linux系统
                process = subprocess.Popen(
                    [sys.executable, temp_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=os.setsid
                )
            
            # 等待执行完成
            stdout, stderr = process.communicate(timeout=self.timeout)
            execution_time = time.time() - start_time
            
            if process.returncode == 0:
                output = stdout.strip()
                
                # 限制输出长度
                if len(output) > self.max_output:
                    output = output[:self.max_output] + "... (truncated)"
                
                return ToolResult(
                    success=True,
                    result=output,
                    metadata={
                        "execution_time": execution_time,
                        "code_length": len(code),
                        "return_code": process.returncode,
                        "generated_code": code != self._extract_code(query),  # 标记是否为生成的代码
                        "code": code  # 添加实际执行的代码
                    }
                )
            else:
                # 执行失败时，提供更详细的错误信息
                stdout_content = stdout.strip() if stdout.strip() else ""
                stderr_content = stderr.strip() if stderr.strip() else ""
                
                # 构建详细的错误信息
                error_details = f"Python execution failed (code {process.returncode})"
                if stderr_content:
                    error_details += f"\nStderr: {stderr_content}"
                if stdout_content:
                    error_details += f"\nStdout: {stdout_content}"
                
                return ToolResult(
                    success=False,
                    result=stdout_content,  # 即使失败也保留stdout输出
                    error=error_details,
                    metadata={
                        "execution_time": execution_time,
                        "code_length": len(code),
                        "return_code": process.returncode,
                        "has_stderr": bool(stderr_content),
                        "has_stdout": bool(stdout_content),
                        "generated_code": code != self._extract_code(query),  # 标记是否为生成的代码
                        "code": code  # 添加实际执行的代码
                    }
                )
                        
            # 终止进程组 - 修复Windows兼容性问题
            if os.name == 'nt':  # Windows系统
                process.terminate()
            else:  # Unix/Linux系统
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            return ToolResult(
                success=False,
                result="",
                error=f"Python execution timeout after {self.timeout} seconds",
                metadata={
                    "execution_time": self.timeout,
                    "code_length": len(code),
                    "timeout": True,
                    "generated_code": code != self._extract_code(query),  # 标记是否为生成的代码
                    "code": code  # 添加实际执行的代码
                }
            )
                
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _extract_code(self, text: str) -> Optional[str]:
        """从文本中提取Python代码 - 改进版本"""
        # 首先检查是否是中文查询，如果是，直接返回None以便生成代码
        if re.search(r'[\u4e00-\u9fff]', text):
            return None
            
        # 首先尝试提取代码块
        code_block_patterns = [
            r'```python\n(.*?)\n```',
            r'```Python\n(.*?)\n```',
            r'```\n(.*?)\n```'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0]
        
        # 如果没有代码块，尝试提取代码行
        lines = text.strip().split('\n')
        code_lines = []
        
        for line in lines:
            line = line.strip()
            
            # 跳过明显的自然语言行
            if not line or line.lower().startswith(('write', 'create', 'generate', 'please', 'can you', 'help me')):
                continue
            
            # 检查是否是代码行
            if self._is_code_line(line):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # 如果都没有，检查整个文本是否可能是代码
        if self._is_code_line(text):
            return text
        
        # 尝试识别数学表达式或简单计算
        # 如果文本包含数学运算符，可能是需要计算的数学表达式
        if any(op in text for op in ['+', '-', '*', '/', '**', '%', '//']):
            # 尝试提取数学表达式部分
            # 使用正则表达式提取数学表达式，忽略中文描述
            # 匹配包含数字和运算符的部分
            math_pattern = r'([\d\s\+\-\*\/\(\)\.\*\*%//,]+)'
            matches = re.findall(math_pattern, text)
            if matches:
                # 取最长的匹配作为数学表达式
                math_expr = max(matches, key=len).strip()
                if math_expr and any(op in math_expr for op in ['+', '-', '*', '/', '**', '%', '//']):
                    # 确保表达式以数字开头
                    if re.match(r'^\d', math_expr.strip()):
                        return math_expr
        
        # 如果上述方法都没有提取到代码，尝试更智能的文本处理
        # 对于包含中文描述的数学表达式，尝试分离中文和数学部分
        # 移除中文标点符号
        text = text.replace('：', ':').replace('，', ',').replace('（', '(').replace('）', ')')
        
        # 尝试提取数学表达式，忽略中文字符
        # 使用更精确的正则表达式匹配数学表达式
        math_expr_pattern = r'((?:\d+(?:\.\d+)?\s*[\+\-\*\/\%\(\)]\s*)+\d+(?:\.\d+)?)'
        matches = re.findall(math_expr_pattern, text)
        if matches:
            # 取最长的匹配作为数学表达式
            math_expr = max(matches, key=len).strip()
            if math_expr and self._is_valid_math_expression(math_expr):
                return math_expr
        
        return None
    
    def _is_valid_math_expression(self, expr: str) -> bool:
        """检查是否是有效的数学表达式"""
        # 尝试编译表达式，但不执行
        compile(expr, '<string>', 'eval')
        return True
    
    def _is_code_line(self, line: str) -> bool:
        """判断一行文本是否可能是代码"""
        # 包含编程关键字的行可能是代码
        code_keywords = [
            'import', 'from', 'def', 'class', 'if', 'elif', 'else', 'for', 'while',
            'try', 'except', 'finally', 'with', 'return', 'yield', 'lambda', 'print',
            'assert', 'del', 'global', 'nonlocal', 'pass', 'break', 'continue'
        ]
        
        # 检查是否包含代码关键字
        if any(keyword in line for keyword in code_keywords):
            return True
        
        # 检查是否包含赋值操作
        if re.search(r'\w+\s*=\s*', line):
            return True
        
        # 检查是否包含函数调用
        if re.search(r'\w+\([^)]*\)', line):
            return True
        
        # 检查是否包含常见的数据结构
        if any(char in line for char in '[]{}()'):
            return True
        
        # 检查是否是数学表达式
        if any(op in line for op in ['+', '-', '*', '/', '**', '%', '//']):
            # 确保不是自然语言描述
            if not any(word in line.lower() for word in ['calculate', 'compute', 'find', 'what is', 'plus', 'minus', 'times']):
                return True
        
        return False
    
    def _security_check(self, code: str) -> Tuple[bool, str]:
        """安全检查代码"""
        # 危险操作列表 - 移除print等基本函数
        dangerous_patterns = [
            r'os\.system',
            r'subprocess\.call',
            r'subprocess\.run',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'compile\s*\(',
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\(',
            r'hasattr\s*\(',
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'delattr\s*\(',
            r'property\s*\(',
            r'super\s*\(',
            r'isinstance\s*\(',
            r'issubclass\s*\(',
            r'callable\s*\(',
        ]
        
        # 网络相关操作（如果不允许）
        if not self.allow_network:
            dangerous_patterns.extend([
                r'urllib',
                r'requests',
                r'http',
                r'socket',
                r'ftplib',
                r'smtplib',
                r'telnetlib',
            ])
        
        # 检查危险模式
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Potentially dangerous operation detected: {pattern}"
        
        return True, "Code passed security check"
    
    def _prepare_safe_code(self, code: str) -> str:
        """准备安全的代码执行环境"""
        # 添加导入限制和输出捕获
        safe_code = """import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr

# 捕获标准输出
output_capture = io.StringIO()
error_capture = io.StringIO()

with redirect_stdout(output_capture), redirect_stderr(error_capture):
"""
        
        # 缩进原始代码
        indented_code = '\n'.join('        ' + line for line in code.split('\n'))
        safe_code += indented_code
        
        safe_code += """
    
    # 打印捕获的输出
    if output_capture.getvalue():
        print(output_capture.getvalue(), end='')
    
    # 如果有错误，也打印错误
    if error_capture.getvalue():
        print("STDERR:", error_capture.getvalue(), end='')
"""
        
        return safe_code
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取工具元数据"""
        return {
            "name": "python_executor",
            "description": "Execute Python code for calculations and computations with enhanced security",
            "parameters": ["query"],
            "timeout": self.timeout,
            "max_output": self.max_output,
            "allow_network": self.allow_network,
            "version": "2.0"
        }
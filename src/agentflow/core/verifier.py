import time
import uuid
"""
Verifier Module for AgentFlow
执行验证器：验证执行结果的有效性和记忆状态的充分性
"""

import re
import os
from typing import Dict, Any, Tuple
from dashscope import Generation
import dashscope
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import yaml
import json

# 导入LLM引擎
from src.LLM_engine import get_llm_engine


class Verifier:
    """
    验证器类
    
    职责：
    1. 验证工具执行结果的有效性
    2. 评估记忆状态的充分性
    3. 决定是继续推理还是终止
    4. 提供详细的验证报告和改进建议
    """
    
    def __init__(self, config_path: str = "src/configs/config.yaml", llm_engine=None):
        """
        初始化验证器
        
        Args:
            config_path: 配置文件路径
            llm_engine: 外部传入的LLM引擎实例（可选）
        """
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.verification_threshold = self.config.get("agent", {}).get("verification_threshold", 0.8)
        
        # 使用传入的llm_engine实例或初始化新的实例
        if llm_engine is not None:
            self.llm_engine = llm_engine
        else:
            self.llm_engine = get_llm_engine(config_path)
        
        # 从LLM引擎获取模型配置
        model_config = self.config.get("model", {})
        self.api_key = model_config.get("dashscope_api_key", "")
        self.api_url = model_config.get("dashscope_base_url", "")
        self.model_name = model_config.get("verifier_model_name", "qwen2.5-7b-instruct-1m")
        self.timeout = self.config.get("agent", {}).get("tool_timeout", 60)
        
        # 设置dashscope API密钥
        if self.api_key:
            dashscope.api_key = self.api_key
        
        # 直接使用模型和API
        
        # 验证状态跟踪
        self.verification_history = []
        self.consecutive_failures = 0
        self.max_consecutive_failures = self.config.get("max_consecutive_failures", 3)
        self.critical_errors = []
        
        # 验证规则配置
        self.verification_rules = self.config.get("verification_rules", {})
        self.is_shared_planner = self.config.get("model", {}).get("verifier_is_shared_planner", False)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file {config_path} not found, using default configuration")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Configuration file parsing error: {e}")
            return {}
    
    def format_input(self, 
                    query: str,
                    sub_goal: str,
                    tool_info: Dict[str, Any],
                    execution_result: str,
                    memory_context: str) -> str:
        """
        Format verification input
        
        Args:
            query: Original query
            sub_goal: Current sub-goal
            tool_info: Tool information
            execution_result: Execution result
            memory_context: Memory context
            
        Returns:
            Formatted input
        """
        tool_name = tool_info.get("name", "Unknown tool")
        tool_description = tool_info.get("description", "No description")
        
        prompt = f"""
You are a verifier in an intelligent agent system, responsible for evaluating the usefulness of tool execution results for solving the original problem and completing the current sub-goal.

[Original Query]
{query}

[Current Sub-goal]
{sub_goal}

[Tool Information]
Tool Name: {tool_name}
Tool Description: {tool_description}

[Execution Result]
{execution_result}

Please evaluate the usefulness of the tool execution result for solving the original problem.

Evaluation Criteria:
1. Does the tool execution result directly help solve the original query?
2. Is the result accurate, reliable, and relevant?
3. Does the result provide new valuable information?
4. Does the result advance the problem-solving process?

Please output ONLY a single number between 0.0 and 1.0 representing your confidence score.
- 0.0 means completely useless
- 1.0 means extremely useful
- Use decimal values like 0.7, 0.85, etc.

Confidence Score:
"""
        return prompt
    
    def parse_output(self, response: str, current_turn: int, max_turn: int) -> Dict[str, Any]:
        """
        Parse verification output
        
        Args:
            response: Verifier output
            current_turn: Current turn
            max_turn: Maximum turn
            
        Returns:
            Parsed verification result, containing verifier_status
        """
        # Check if response is None
        if response is None:
            logger.error("Verifier response is None")
            return {
                "success": False,
                "error": "Verifier response is None",
                "verifier_status": 0
            }
        
        # Ensure response is string type
        if not isinstance(response, str):
            response = str(response)
        
        # Check if response is empty
        if not response.strip():
            logger.error("Verifier response is empty")
            return {
                "success": False,
                "error": "Verifier response is empty",
                "verifier_status": 0
            }
        
        # Log the response for debugging
        logger.debug(f"Verifier response: {response}")
        
        # Initialize confidence with a default value
        confidence = 0.5  # Default middle value
        confidence_extracted = False
        
        # Method 1: Try to extract a single number from the response
        # Look for patterns like "0.85", "0.7", "1.0", "0.0"
        #print("验证response:\n", response)
        number_match = re.search(r'(\d+\.?\d*)', response.strip())
        if number_match:
            try:
                confidence = float(number_match.group(1))
                # Ensure confidence is between 0 and 1
                if confidence > 1:
                    confidence = confidence / 10  # If it's like 8.5, make it 0.85
                confidence_extracted = True
                #logger.info(f"Confidence extracted as number: {confidence}")
            except ValueError:
                pass
        
        # Method 2: If direct number extraction fails, try to extract from common patterns
        if not confidence_extracted:
            # Look for patterns like "confidence: 0.8" or "score is 0.8"
            confidence_patterns = [
                r'confidence[:\s]+(\d+\.?\d*)',
                r'confidence score[:\s]+is[:\s]+(\d+\.?\d*)',
                r'score[:\s]+(\d+\.?\d*)',
                r'rating[:\s]+(\d+\.?\d*)',
                r'评估[:\s]+(\d+\.?\d*)',
                r'置信度[:\s]+(\d+\.?\d*)'
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    confidence = float(match.group(1))
                    # Ensure confidence is between 0 and 1
                    if confidence > 1:
                        confidence = confidence / 10  # If it's like 8.5, make it 0.85
                    confidence_extracted = True
                    #logger.info(f"Confidence extracted using pattern: {confidence}")
                    break
        
        # Method 3: Try to infer confidence from qualitative descriptions
        if not confidence_extracted:
            # Look for qualitative descriptions
            positive_indicators = ['high', 'very useful', 'excellent', 'perfect', 'great', 'good', 'helpful', 'effective', 'valuable', 'relevant', 'accurate', 'reliable', '高', '很好', '优秀', '完美', '很好', '有帮助', '有效', '有价值', '相关', '准确', '可靠']
            negative_indicators = ['low', 'not useful', 'poor', 'bad', 'unhelpful', 'ineffective', 'irrelevant', 'inaccurate', 'unreliable', '低', '没用', '差', '不好', '无帮助', '无效', '不相关', '不准确', '不可靠']
            
            response_lower = response.lower()
            
            # Count occurrences
            positive_count = sum(1 for indicator in positive_indicators if indicator in response_lower)
            negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)
            
            if positive_count > 0 or negative_count > 0:
                # Calculate confidence based on ratio
                total_indicators = positive_count + negative_count
                if total_indicators > 0:
                    confidence = positive_count / total_indicators
                    confidence_extracted = True
                    #logger.info(f"Confidence inferred from qualitative descriptions: {confidence} (positive: {positive_count}, negative: {negative_count})")
        
        # If still not extracted, log error and use default
        if not confidence_extracted:
            logger.warning("Unable to extract confidence assessment, using default value of 0.5")
            confidence = 0.5
        
        # Determine verifier_status based on confidence and turns
        # If confidence > 0.8 and current turn <= max_turn, then verifier_status is 1 (tool result is useful)
        # Otherwise it's 0 (tool result is useless, need to re-plan)
        verifier_status = 1 if (confidence >= self.verification_threshold and current_turn <= max_turn) else 0
        
        #logger.info(f"Final confidence: {confidence:.2f}, verifier_status: {verifier_status}")
        
        return {
            "success": True,
            "confidence": confidence,
            "analysis": "",  # No analysis expected with simplified format
            "verifier_status": verifier_status,
            "should_continue": verifier_status == 1,
            "reason": f"Confidence: {confidence:.2f}, Turn: {current_turn}/{max_turn}"
        }
            
    
    def _parse_verification_status(self, status_text: str) -> Dict[str, Any]:
        """Parse verification status"""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', status_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON, try to extract information from text
                continue_reasoning = "continue" in status_text.lower() or "继续" in status_text
                reason = status_text.strip()
                
                return {
                    "continue": continue_reasoning,
                    "reason": reason,
                    "next_focus": ""
                }
        except Exception as e:
            logger.warning(f"Failed to parse verification status: {e}")
            return {
                "continue": True,  # Default to continue
                "reason": "Parsing failed, default to continue",
                "next_focus": ""
            }
    
    def _parse_errors(self, errors_text: str) -> list[Dict[str, Any]]:
        """Parse error information"""
        errors = []
        
        if "No errors" in errors_text or "无错误" in errors_text or not errors_text.strip():
            return errors
        
        try:
            # Try to extract error entries
            error_sections = re.split(r'- Error Type:|- 错误类型：', errors_text)
            
            for section in error_sections[1:]:  # Skip the first empty part
                error = {}
                
                # Extract error type
                type_match = re.search(r'(.*?)\n', section)
                if type_match:
                    error["type"] = type_match.group(1).strip()
                
                # Extract error description
                desc_match = re.search(r'Error Description:|错误描述：(.*?)\n', section)
                if desc_match:
                    error["description"] = desc_match.group(1).strip()
                
                # Extract severity
                severity_match = re.search(r'Severity:|严重程度：(.*?)\n', section)
                if severity_match:
                    error["severity"] = severity_match.group(1).strip()
                
                # Extract fix suggestion
                fix_match = re.search(r'Fix Suggestion:|修复建议：(.*?)(?=\n- Error Type:|(?=\n- 错误类型：)|$)', section, re.DOTALL)
                if fix_match:
                    error["fix_suggestion"] = fix_match.group(1).strip()
                
                if error:  # Only add non-empty errors
                    errors.append(error)
                    
        except Exception as e:
            logger.warning(f"Failed to parse error information: {e}")
            # If parsing fails, create a generic error
            errors.append({
                "type": "Parsing Error",
                "description": f"Unable to parse error information: {e}",
                "severity": "Medium",
                "fix_suggestion": "Check error format"
            })
        
        return errors
    
    def _apply_verification_rules(self, status: Dict[str, Any], errors: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply verification rules to modify status"""
        # If there are high severity errors, force continue
        high_severity_errors = [e for e in errors if e.get("severity") in ["高", "High"]]
        if high_severity_errors:
            status["continue"] = True
            status["reason"] = f"High severity error found, needs fixing: {high_severity_errors[0].get('description', '')}"
            if not status.get("next_focus"):
                status["next_focus"] = f"Fix error: {high_severity_errors[0].get('type', '')}"
        
        # Apply custom rules
        if self.verification_rules:
            try:
                # Rule 1: If error count exceeds threshold, force continue
                max_errors = self.verification_rules.get("max_errors", 5)
                if len(errors) > max_errors:
                    status["continue"] = True
                    status["reason"] = f"Error count ({len(errors)}) exceeds threshold ({max_errors}), needs continued processing"
                
                # Rule 2: If there are specific types of errors, force continue
                force_continue_types = self.verification_rules.get("force_continue_error_types", [])
                for error in errors:
                    if error.get("type") in force_continue_types:
                        status["continue"] = True
                        status["reason"] = f"Error type found that needs handling: {error.get('type')}"
                        break
                        
            except Exception as e:
                logger.warning(f"Failed to apply verification rules: {e}")
        
        return status
    
    def get_txt(self, prompt: str) -> str:
        """调用LLM引擎进行验证 - 共享planner的权重和分词器"""

        #logger.info(f"调用LLM引擎进行验证，模型: {self.model_name}")
        #logger.info(f"提示词长度: {len(prompt)} 字符")
        response = self.llm_engine.generate(
            model_source="local" if self.is_shared_planner else "dashscope",
            model_name=self.model_name,
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.5
        )
        
        result = response if response else ""
        #logger.info(f"LLM引擎响应接收，长度: {len(result) if result else 0} 字符\n内容：\n{result}\n====================================================\n")
        return result
                

    
    def verify(self, 
               query: str,
               sub_goal: str,
               tool_info: Dict[str, Any],
               execution_result: str,
               memory_context: str,
               current_turn: int,
               max_turn: int) -> Dict[str, Any]:
        """
        Execute verification
        
        Args:
            query: Original query
            sub_goal: Current sub-goal
            tool_info: Tool information
            execution_result: Execution result
            memory_context: Memory context
            current_turn: Current turn
            max_turn: Maximum turn
            
        Returns:
            Verification result, containing verifier_status
        """
        start_time = time.time()
        verification_id = str(uuid.uuid4())
        
        # Record verification start
        verification_record = {
            "id": verification_id,
            "query": query,
            "sub_goal": sub_goal,
            "tool_info": tool_info,
            "start_time": start_time,
            "status": "running"
        }
        
        # 格式化输入
        formatted_input = self.format_input(query, sub_goal, tool_info, execution_result, memory_context)
        #logger.info(f"Formatted input length: {len(formatted_input)} characters")
        
        #logger.info("Executing verification...")
        response = self.get_txt(formatted_input)
        
        #print(f"验证原始 response: \n{response}\n,验证current_turn: {current_turn}\n,验证max_turn: {max_turn}")
        
        # 检查响应是否为None或空
        if response is None:
            logger.error("API returned None response")
            raise Exception("API returned None response")
        
        if not response.strip():
            logger.error("API returned empty response")
            raise Exception("API returned empty response")
        
        # 解析输出
        parsed_output = self.parse_output(response, current_turn, max_turn)
        
        # 记录验证成功
        verification_time = time.time() - start_time
        verification_record.update({
            "status": "success",
            "result": parsed_output,
            "verification_time": verification_time,
            "end_time": time.time()
        })
        
        # 重置连续失败计数
        self.consecutive_failures = 0
        
        #logger.info(f"Verification completed, time taken: {verification_time:.2f} seconds")
        
        return {
            "success": True,
            "result": parsed_output,
            "raw_output": response,
            "verification_time": verification_time,
            "verification_id": verification_id
        }
    
    def get_verification_history(self, limit: int = 10) -> list[Dict[str, Any]]:
        """Get verification history"""
        return self.verification_history[-limit:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.verification_history:
            return {
                "total_verifications": 0,
                "successful_verifications": 0,
                "failed_verifications": 0,
                "success_rate": 0,
                "average_verification_time": 0,
                "consecutive_failures": 0,
                "critical_errors": len(self.critical_errors)
            }
        
        total_verifications = len(self.verification_history)
        successful_verifications = sum(1 for v in self.verification_history if v["status"] == "success")
        failed_verifications = total_verifications - successful_verifications
        success_rate = successful_verifications / total_verifications
        
        verification_times = [v["verification_time"] for v in self.verification_history if "verification_time" in v]
        average_verification_time = sum(verification_times) / len(verification_times) if verification_times else 0
        
        return {
            "total_verifications": total_verifications,
            "successful_verifications": successful_verifications,
            "failed_verifications": failed_verifications,
            "success_rate": success_rate,
            "average_verification_time": average_verification_time,
            "consecutive_failures": self.consecutive_failures,
            "critical_errors": len(self.critical_errors),
            "last_errors": self.critical_errors[-5:]  # Last 5 critical errors
        }
    
    def reset_error_counters(self):
        """Reset error counters"""
        self.consecutive_failures = 0
        self.critical_errors = []
        #logger.info("Error counters have been reset")
    
    def is_healthy(self) -> bool:
        """Check verifier health status"""
        return (
            self.consecutive_failures < self.max_consecutive_failures and
            len(self.critical_errors) == 0
        )
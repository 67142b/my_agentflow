"""
Prompt Templates Module for AgentFlow
提示词模板管理模块：统一各角色间的提示词格式和内容
"""

from typing import Dict, Any, List


class PromptTemplates:
    """
    提示词模板管理类
    
    职责：
    1. 统一各角色间的提示词格式
    2. 提供标准化的提示词模板
    3. 支持提示词的动态配置和定制
    """
    
    # 规划器提示词模板
    PLANNER_TEMPLATES = {
        "query_analysis": """You are an action planner in an agentic reasoning system. Your task is to analyze the query and determine the required skills and tools.

[Query]
{query}

[Memory Context]
{memory_context}

Your task is to:
1. Analyze the query to understand what the user is asking
2. Identify the required skills to answer the query
3. Recommend appropriate tools from the available list

Available tools: {available_tools}

Tool descriptions:
{tool_descriptions}

Provide your analysis in the following format:

[Query Analysis]
{Provide a concise summary of what the user is asking for}

[Required Skills]
{List the specific skills needed to answer the query}

[Recommended Tools]
{Choose the most appropriate tools from the available list}

[Additional Considerations]
{Any additional context or considerations that might be relevant}

Now, provide your analysis:""",

        "next_step": """You are an action planner in an agentic reasoning system. Based on the query and current memory context, plan the next action.

[Query]
{query}

[Memory Context]
{memory_context}

[Required Skills]
{required_skills}

Your task is to:
1. Analyze the current situation
2. Determine if the query has been answered
3. If not answered, plan the next step:
   - Identify the current sub-goal
   - Select the appropriate tool
   - Provide context for the tool

Available tools: {available_tools}

Tool descriptions:
{tool_descriptions}

Provide your response in this format:

[Justification]
Provide your reasoning for selecting this tool and sub-goal. Explain why this approach is appropriate for the current query and memory context.

[Context]
Provide specific context and detailed instructions for the selected tool to work with. Include all necessary information, parameters, and constraints the tool needs to execute successfully.

[Sub-Goal]
Describe the specific, achievable sub-goal for this turn. This should be a clear, focused objective that contributes to answering the overall query.

[Tool Name]
Select one tool from the available tools: base_generator, python_executor, web_search, wikipedia_search

Example:
[Justification]
The query requires calculating a mathematical result that needs precise computation.

[Context]
Calculate the sum of the first 100 prime numbers. The calculation should be accurate and efficient.

[Sub-Goal]
Compute the sum of prime numbers

[Tool Name]
python_executor

Now, provide your plan:

Note for testing: In test mode, you can include a line "TEST_MODE_TOOL: [tool_name]" at the beginning of your response to force the selection of a specific tool."""
    }
    
    # 执行器提示词模板
    EXECUTOR_TEMPLATES = {
        "tool_execution": """You are an action executor in an agentic reasoning system. Your task is to execute the selected tool and provide a result.

[Tool to Execute]
{tool_name}

[Tool Description]
{tool_description}

[Original Query]
{query}

[Memory Context]
{memory_context}

[Tool Context]
{tool_context}

Your task is to execute the tool based on the context and provide a result. Follow these guidelines:

1. For base_generator: Provide reasoning and analysis to address the query
2. For python_executor: Write and execute Python code to solve the problem
3. For web_search: Search for relevant information and summarize findings
4. For wikipedia_search: Search Wikipedia for encyclopedic information

Provide your response in this format:

[Execution Result]
{Your execution result}

[Confidence]
{High/Medium/Low}

[Additional Notes]
{Any additional relevant information}

Now, execute the tool:""",

        "verification": """You are an action executor in an agentic reasoning system. Your task is to verify if the tool execution was successful and the result is valid.

[Tool Used]
{tool_name}

[Tool Context]
{tool_context}

[Execution Result]
{execution_result}

[Original Query]
{query}

Your task is to verify the execution result and determine if it's valid and useful. Consider:
1. Did the tool execute successfully?
2. Is the result relevant to the query?
3. Are there any errors or issues?
4. Is additional verification needed?

Provide your verification in the following format:

[Verification Status]
{Success/Partial Success/Failed}

[Reasoning]
{Explain your verification decision}

[Error Signal]
{If there's an error, describe it; otherwise, write "None"}

Now, provide your verification:"""
    }
    
    # 评估器提示词模板
    EVALUATOR_TEMPLATES = {
        "result_evaluation": """You are an evaluator in an agentic reasoning system. Your task is to evaluate the tool execution result and decide whether to continue reasoning or terminate.

[Query]
{query}

[Current Sub-Goal]
{sub_goal}

[Tool Used]
{tool_name}

[Tool Execution Result]
{tool_result}

[Memory Context]
{memory_context}

Your task is to evaluate the result and determine if the current sub-goal has been adequately addressed. Consider:
1. Has the sub-goal been achieved?
2. Is the information sufficient to answer the original query?
3. Are there any errors or issues that need to be addressed?
4. Is additional reasoning or tool use needed?

Provide your evaluation in the following format:

[Verification Status]
{0 for continue, 1 for terminate}

[Reasoning]
{Explain your decision}

[Error Signal]
{If there's an error, describe it; otherwise, write "None"}

[Confidence]
{High/Medium/Low}

Example:
[Verification Status]
1

[Reasoning]
The tool successfully calculated the sum of the first 100 prime numbers, which directly answers the user's query. No further computation is needed.

[Error Signal]
None

[Confidence]
High

Now, provide your evaluation:""",

        "context_verification": """You are an evaluator in an agentic reasoning system. Your task is to verify if the current context is complete and consistent for answering the query.

[Query]
{query}

[Current Context]
{context}

Your task is to evaluate the context and determine if it's sufficient and consistent. Consider:
1. Does the context contain all necessary information to answer the query?
2. Is the information consistent and coherent?
3. Are there any contradictions or gaps?
4. Is additional information needed?

Provide your evaluation in the following format:

[Completeness]
{Complete/Partial/Incomplete}

[Consistency]
{Consistent/Mostly Consistent/Inconsistent}

[Additional Information Needed]
{Yes/No}

[Reasoning]
{Explain your evaluation}

[Confidence]
{High/Medium/Low}

Now, provide your evaluation:"""
    }
    
    # 基础生成器提示词模板
    BASE_GENERATOR_TEMPLATES = {
        "single_reasoning": """You are a helpful AI assistant. Provide a clear and accurate answer to the following question.

[Question]
{input_text}

{context_section}

Please provide a comprehensive answer:

[Answer]
{detailed_answer}

[Confidence]
{confidence_level}

[Additional Notes]
{additional_notes}""",

        "continuous_reasoning": """You are a reasoning engine in an agentic system. Your task is to provide reasoning and analysis to address the query.

[Original Query]
{input_text}

[Memory Context]
{memory_context}

[Planned Sub-Goal]
{sub_goal}

[Selected Tool]
{selected_tool}

[Tool Context]
{tool_context}

{context_section}

Your task is to provide reasoning and analysis to address the query based on the sub-goal and context. Follow these guidelines:

1. Analyze the problem thoroughly
2. Provide logical reasoning
3. Consider different perspectives
4. Draw conclusions based on evidence
5. Indicate if the query has been fully answered or if more work is needed

Provide your response in this format:

[Reasoning]
{detailed_reasoning}

[Conclusion]
{conclusion}

[Confidence]
{confidence_level}

[Additional Notes]
{additional_notes}

Now, provide your reasoning:"""
    }
    
    # 搜索增强提示词模板
    SEARCH_ENHANCEMENT_TEMPLATES = {
        "result_summarization": """You are a search result analyzer in an agentic reasoning system. Your task is to analyze the search results and extract information that is most relevant to solving the original query.

[Original Query]
{original_query}

[Search Tool Used]
{search_tool}

[Search Results]
{search_results}

Your task is to:
1. Analyze the search results to identify information that directly addresses the original query
2. Extract key facts, data, and insights that are most relevant
3. Organize the information in a clear and concise manner
4. Highlight any contradictions or uncertainties in the information
5. Provide a summary that helps solve the original problem

Provide your analysis in the following format:

[Relevant Information]
{Extract the most relevant information from the search results that directly addresses the original query}

[Key Insights]
{Provide key insights, patterns, or conclusions that can be drawn from the search results}

[Information Quality]
{Assess the quality and reliability of the information found}

[Summary]
{Provide a concise summary that helps solve the original problem}

Now, analyze the search results:"""
    }
    
    # 工具元数据模板
    TOOL_METADATA = {
        "base_generator": {
            "description": "Default reasoning engine for general problem solving",
            "parameters": ["input_text", "context"],
            "usage": "Use for general reasoning, analysis, and problem-solving tasks"
        },
        "python_executor": {
            "description": "Generate and execute Python code for calculations and computations using Qwen-Plus model",
            "parameters": ["query", "context"],
            "usage": "Use for mathematical calculations, data processing, and algorithmic tasks. The tool will automatically generate Python code based on the query and execute it."
        },
        "web_search": {
            "description": "Search the web for information",
            "parameters": ["search_query", "context"],
            "usage": "Use for finding current information, news, and general web content"
        },
        "wikipedia_search": {
            "description": "Search Wikipedia for encyclopedic information",
            "parameters": ["search_query", "context"],
            "usage": "Use for factual information, definitions, and encyclopedic knowledge"
        }
    }
    
    @classmethod
    def get_planner_prompt(cls, template_name: str, **kwargs) -> str:
        """
        获取规划器提示词
        
        Args:
            template_name: 模板名称
            **kwargs: 模板参数
            
        Returns:
            str: 格式化后的提示词
        """
        if template_name not in cls.PLANNER_TEMPLATES:
            raise ValueError(f"Unknown planner template: {template_name}")
        
        template = cls.PLANNER_TEMPLATES[template_name]
        
        # 添加默认值
        if "available_tools" not in kwargs:
            kwargs["available_tools"] = ", ".join(cls.TOOL_METADATA.keys())
        
        if "tool_descriptions" not in kwargs:
            kwargs["tool_descriptions"] = cls._format_tool_descriptions()
        
        return template.format(**kwargs)
    
    @classmethod
    def get_executor_prompt(cls, template_name: str, **kwargs) -> str:
        """
        获取执行器提示词
        
        Args:
            template_name: 模板名称
            **kwargs: 模板参数
            
        Returns:
            str: 格式化后的提示词
        """
        if template_name not in cls.EXECUTOR_TEMPLATES:
            raise ValueError(f"Unknown executor template: {template_name}")
        
        template = cls.EXECUTOR_TEMPLATES[template_name]
        
        # 添加工具描述
        if "tool_name" in kwargs and "tool_description" not in kwargs:
            tool_name = kwargs["tool_name"]
            kwargs["tool_description"] = cls.TOOL_METADATA.get(tool_name, {}).get("description", "")
        
        return template.format(**kwargs)
    
    @classmethod
    def get_evaluator_prompt(cls, template_name: str, **kwargs) -> str:
        """
        获取评估器提示词
        
        Args:
            template_name: 模板名称
            **kwargs: 模板参数
            
        Returns:
            str: 格式化后的提示词
        """
        if template_name not in cls.EVALUATOR_TEMPLATES:
            raise ValueError(f"Unknown evaluator template: {template_name}")
        
        template = cls.EVALUATOR_TEMPLATES[template_name]
        return template.format(**kwargs)
    
    @classmethod
    def get_base_generator_prompt(cls, template_name: str, **kwargs) -> str:
        """
        获取基础生成器提示词
        
        Args:
            template_name: 模板名称
            **kwargs: 模板参数
            
        Returns:
            str: 格式化后的提示词
        """
        if template_name not in cls.BASE_GENERATOR_TEMPLATES:
            raise ValueError(f"Unknown base generator template: {template_name}")
        
        template = cls.BASE_GENERATOR_TEMPLATES[template_name]
        
        # 添加上下文部分
        if "context" in kwargs and kwargs["context"]:
            kwargs["context_section"] = f"\n[Context]\n{kwargs['context']}\n"
        else:
            kwargs["context_section"] = ""
        
        # 为single_reasoning模板添加默认值
        if template_name == "single_reasoning":
            if "detailed_answer" not in kwargs:
                kwargs["detailed_answer"] = "{Your detailed answer}"
            if "confidence_level" not in kwargs:
                kwargs["confidence_level"] = "High/Medium/Low"
            if "additional_notes" not in kwargs:
                kwargs["additional_notes"] = "Any additional relevant information or considerations"
        
        # 为continuous_reasoning模板添加默认值
        elif template_name == "continuous_reasoning":
            if "detailed_reasoning" not in kwargs:
                kwargs["detailed_reasoning"] = "{Your detailed reasoning and analysis}"
            if "conclusion" not in kwargs:
                kwargs["conclusion"] = "STOP if the query is fully answered, or CONTINUE if more work is needed"
            if "confidence_level" not in kwargs:
                kwargs["confidence_level"] = "High/Medium/Low"
            if "additional_notes" not in kwargs:
                kwargs["additional_notes"] = "Any additional relevant information or considerations"
        
        return template.format(**kwargs)
    
    @classmethod
    def get_tool_metadata(cls, tool_name: str = None) -> Dict[str, Any]:
        """
        获取工具元数据
        
        Args:
            tool_name: 工具名称，如果为None则返回所有工具元数据
            
        Returns:
            Dict[str, Any]: 工具元数据
        """
        if tool_name is None:
            return cls.TOOL_METADATA
        
        return cls.TOOL_METADATA.get(tool_name, {})
    
    @classmethod
    def _format_tool_descriptions(cls) -> str:
        """
        格式化工具描述
        
        Returns:
            str: 格式化后的工具描述
        """
        descriptions = []
        for tool_name, metadata in cls.TOOL_METADATA.items():
            descriptions.append(f"- {tool_name}: {metadata['description']}")
        
        return "\n".join(descriptions)
    
    @classmethod
    def get_search_enhancement_prompt(cls, template_name: str, **kwargs) -> str:
        """
        获取搜索增强提示词
        
        Args:
            template_name: 模板名称
            **kwargs: 模板参数
            
        Returns:
            str: 格式化后的提示词
        """
        if template_name not in cls.SEARCH_ENHANCEMENT_TEMPLATES:
            raise ValueError(f"Unknown search enhancement template: {template_name}")
        
        template = cls.SEARCH_ENHANCEMENT_TEMPLATES[template_name]
        return template.format(**kwargs)
    
    @classmethod
    def get_all_templates(cls) -> Dict[str, Dict[str, str]]:
        """
        获取所有模板
        
        Returns:
            Dict[str, Dict[str, str]]: 所有模板
        """
        return {
            "planner": cls.PLANNER_TEMPLATES,
            "executor": cls.EXECUTOR_TEMPLATES,
            "evaluator": cls.EVALUATOR_TEMPLATES,
            "base_generator": cls.BASE_GENERATOR_TEMPLATES,
            "search_enhancement": cls.SEARCH_ENHANCEMENT_TEMPLATES
        }
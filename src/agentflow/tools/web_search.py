"""
Web Search Tool
Web搜索工具 - 仅保留DuckDuckGo和Google Custom Search实现
"""

import requests
import json
import os
import re
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode, quote_plus
from bs4 import BeautifulSoup
from .base import BaseTool, ToolResult


class WebContentExtractor:
    """网页内容提取器 - 用于获取URL的详细内容"""
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = min(max_retries, 3)  # 确保最多重试3次
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_content(self, url: str, max_length: int = 2000) -> Dict[str, Any]:
        """
        从URL提取内容，支持重试机制
        
        Args:
            url: 要提取内容的URL
            max_length: 最大内容长度
            
        Returns:
            Dict: 包含提取的内容和元数据
        """
        last_error = None
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                # 发送请求
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # 解析HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 移除脚本和样式元素
                for script in soup(["script", "style"]):
                    script.extract()
                
                # 提取标题
                title = soup.find('title')
                title_text = title.get_text(strip=True) if title else "无标题"
                
                # 提取正文内容
                # 尝试常见的内容区域选择器
                content_selectors = [
                    'article', 'main', '.content', '.post-content', 
                    '.entry-content', '.article-body', '#content'
                ]
                
                content_element = None
                for selector in content_selectors:
                    content_element = soup.select_one(selector)
                    if content_element:
                        break
                
                # 如果没有找到内容区域，使用body
                if not content_element:
                    content_element = soup.find('body')
                
                # 提取文本内容
                if content_element:
                    content_text = content_element.get_text(separator=' ', strip=True)
                    # 清理多余的空白字符
                    content_text = re.sub(r'\s+', ' ', content_text)
                    
                    # 限制内容长度
                    if len(content_text) > max_length:
                        content_text = content_text[:max_length] + "..."
                else:
                    content_text = "无法提取内容"
                
                return {
                    "success": True,
                    "title": title_text,
                    "content": content_text,
                    "url": url,
                    "content_length": len(content_text)
                }
                
            except (requests.RequestException, Exception) as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    # 等待一段时间再重试
                    import time
                    time.sleep(1 * retry_count)  # 递增等待时间
        
        # 所有重试都失败，返回错误信息
        return {
            "success": False,
            "error": f"无法获取URL内容，已重试{self.max_retries}次。最后错误: {str(last_error)}",
            "url": url
        }


class DuckDuckGoSearchTool(BaseTool):
    """DuckDuckGo搜索工具 - 免费，无需API密钥"""
    
    def __init__(self, timeout: int = 10, max_results: int = 5, fetch_content: bool = True, max_retries: int = 3):
        super().__init__(
            name="duckduckgo_search",
            description="Search the web using DuckDuckGo (free, no API key required)"
        )
        self.timeout = timeout
        self.max_results = max_results
        self.fetch_content = fetch_content  # 是否获取URL的详细内容
        self.max_retries = min(max_retries, 3)  # 确保最多重试3次
        self.content_extractor = WebContentExtractor(timeout=timeout, max_retries=self.max_retries)
    
    def execute(self, query: str, context: Optional[str] = None) -> ToolResult:
        """执行DuckDuckGo搜索，支持网络错误处理和重试机制"""
        # 导入ddgs库
        print("\n使用DuckDuckGo搜索工具\n")
        from ddgs import DDGS
        import time
        
        results = []
        last_error = None
        retry_count = 0
        
        # 尝试使用ddgs库进行搜索
        while retry_count < self.max_retries and not results:
            try:
                with DDGS() as ddgs:
                    # 使用region和timelimit参数获取更相关的结果
                    # 避免使用可能触发Wikipedia API的参数
                    search_results = list(ddgs.text(
                        query, 
                        max_results=self.max_results,
                        region='us-en',  # 改为美国英语，避免可能的区域问题
                        safesearch='moderate',  # 适度安全搜索
                        timelimit=None  # 移除时间限制，避免可能的API调用
                    ))
                    
                    for result in search_results:
                        result_item = {
                            'title': result['title'],
                            'url': result['href'],
                            'snippet': result['body']
                        }
                        
                        # 如果启用了内容获取，获取URL的详细内容
                        if self.fetch_content and result_item['url'] and result_item['url'].startswith('http'):
                            content_result = self.content_extractor.extract_content(result_item['url'])
                            if content_result['success']:
                                result_item['content'] = content_result['content']
                                result_item['content_title'] = content_result['title']
                            else:
                                result_item['content'] = f"无法获取内容: {content_result.get('error', '未知错误')}"
                                result_item['content_title'] = ""
                        
                        results.append(result_item)
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    # 等待一段时间再重试
                    time.sleep(1 * retry_count)  # 递增等待时间
        
        # 如果ddgs搜索失败，尝试HTML解析方法
        if not results:
            try:
                return self._execute_with_html_parsing(query)
            except Exception as e:
                last_error = e
                # 如果HTML解析也失败，尝试即时答案API
                try:
                    return self._get_instant_answer(query)
                except Exception as e:
                    last_error = e
                    # 所有方法都失败，返回空结果
                    return ToolResult(
                        success=False,
                        result="",
                        error=f"搜索失败，已尝试所有方法。最后错误: {str(last_error)}"
                    )
        
        # 格式化结果
        result_text = f"DuckDuckGo search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            result_text += f"{i}. {result['title']}\n"
            result_text += f"   {result['snippet']}\n"
            result_text += f"   URL: {result['url']}\n"
            
            # 如果有详细内容，添加到结果中
            if 'content' in result and result['content']:
                result_text += f"   Content: {result['content'][:5000]}"
                if len(result['content']) > 5000:
                    result_text += "..."
                result_text += "\n"
            
            result_text += "\n"
        
        print(f"搜索完成，共找到{len(results)}个结果")
        return ToolResult(
            success=True,
            result=result_text.strip(),
            metadata={
                "query": query,
                "results_count": len(results),
                "tool": "duckduckgo_search",
                "source": "duckduckgo",
                "fetch_content": self.fetch_content
            }
        )
    
    def _execute_with_html_parsing(self, query: str) -> ToolResult:
        """使用HTML解析的回退方法，支持网络错误处理和重试机制"""
        # 导入必要的库
        from urllib.parse import quote_plus
        import requests
        from bs4 import BeautifulSoup
        import time
        
        # 创建会话
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 构建搜索URL
        encoded_query = quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={encoded_query}"
        
        results = []
        last_error = None
        retry_count = 0
        
        while retry_count < self.max_retries and not results:
            try:
                # 发送请求
                response = session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # 解析HTML结果
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 提取搜索结果
                result_divs = soup.find_all('div', class_='result')
                
                for div in result_divs[:self.max_results]:
                    # 提取标题和链接
                    title_tag = div.find('a', class_='result__a')
                    if title_tag:
                        title = title_tag.get_text(strip=True)
                        result_url = title_tag.get('href', '')
                        
                        # 提取摘要
                        snippet_tag = div.find('a', class_='result__snippet')
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                        
                        if title and snippet:
                            result_item = {
                                'title': title,
                                'url': result_url,
                                'snippet': snippet
                            }
                            
                            # 如果启用了内容获取，获取URL的详细内容
                            if self.fetch_content and result_url and result_url.startswith('http'):
                                content_result = self.content_extractor.extract_content(result_url)
                                if content_result['success']:
                                    result_item['content'] = content_result['content']
                                    result_item['content_title'] = content_result['title']
                                else:
                                    result_item['content'] = f"无法获取内容: {content_result.get('error', '未知错误')}"
                                    result_item['content_title'] = ""
                            
                            results.append(result_item)
                            
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    # 等待一段时间再重试
                    time.sleep(1 * retry_count)  # 递增等待时间
        
        if not results:
            # 如果HTML解析也失败，尝试即时答案API
            try:
                return self._get_instant_answer(query)
            except Exception as e:
                last_error = e
                # 所有方法都失败，返回空结果
                return ToolResult(
                    success=False,
                    result="",
                    error=f"HTML解析搜索失败，已尝试所有方法。最后错误: {str(last_error)}"
                )
        
        # 格式化结果
        result_text = f"DuckDuckGo search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            result_text += f"{i}. {result['title']}\n"
            result_text += f"   {result['snippet']}\n"
            result_text += f"   URL: {result['url']}\n"
            
            # 如果有详细内容，添加到结果中
            if 'content' in result and result['content']:
                result_text += f"   Content: {result['content'][:5000]}"
                if len(result['content']) > 5000:
                    result_text += "..."
                result_text += "\n"
            
            result_text += "\n"
        
        return ToolResult(
            success=True,
            result=result_text.strip(),
            metadata={
                "query": query,
                "results_count": len(results),
                "tool": "duckduckgo_search",
                "source": "duckduckgo",
                "fetch_content": self.fetch_content,
                "method": "html_parsing_fallback"
            }
        )
    
    def _get_instant_answer(self, query: str) -> ToolResult:
        """获取DuckDuckGo即时答案，支持网络错误处理和重试机制"""
        import time
        
        # 使用DuckDuckGo即时答案API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        
        last_error = None
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                
                # 提取即时答案
                if data.get('AbstractText'):
                    result_text = f"Instant Answer for '{query}':\n\n"
                    result_text += f"{data['AbstractText']}\n\n"
                    if data.get('AbstractURL'):
                        result_text += f"Source: {data['AbstractURL']}"
                    
                    return ToolResult(
                        success=True,
                        result=result_text.strip(),
                        metadata={
                            "query": query,
                            "tool": "duckduckgo_search",
                            "source": "instant_answer",
                            "abstract_source": data.get('AbstractSource')
                        }
                    )
                
                # 如果没有即时答案，返回相关主题
                if data.get('RelatedTopics'):
                    topics = data['RelatedTopics'][:3]
                    result_text = f"Related topics for '{query}':\n\n"
                    
                    for topic in topics:
                        if topic.get('Text') and topic.get('FirstURL'):
                            result_text += f"• {topic['Text']}\n"
                            result_text += f"  {topic['FirstURL']}\n\n"
                    
                    return ToolResult(
                        success=True,
                        result=result_text.strip(),
                        metadata={
                            "query": query,
                            "tool": "duckduckgo_search",
                            "source": "related_topics"
                        }
                    )
                
                # 没有找到结果，不需要重试
                return ToolResult(
                    success=False,
                    result="",
                    error=f"No results found for '{query}'"
                )
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    # 等待一段时间再重试
                    time.sleep(1 * retry_count)  # 递增等待时间
        
        # 所有重试都失败，返回错误信息
        return ToolResult(
            success=False,
            result="",
            error=f"即时答案API调用失败，已尝试{self.max_retries}次。最后错误: {str(last_error)}"
        )
    
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
                    "description": "搜索查询关键词"
                }
            },
            "required": ["query"]
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取工具元数据"""
        return {
            "name": "duckduckgo_search",
            "description": "Search the web using DuckDuckGo (free, no API key required)",
            "parameters": ["query"],
            "timeout": self.timeout,
            "max_results": self.max_results,
            "fetch_content": self.fetch_content,
            "version": "1.0"
        }


class GoogleCustomSearchTool(BaseTool):
    """Google Custom Search API工具 - 需要API密钥，支持网络错误处理和重试机制"""
    
    def __init__(self, api_key: str, search_engine_id: str, timeout: int = 10, max_results: int = 5, fetch_content: bool = True, max_retries: int = 3):
        super().__init__(
            name="google_custom_search",
            description="Search using Google Custom Search API (requires API key)"
        )
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.timeout = timeout
        self.max_results = min(max_results, 10)  # Google API限制
        self.fetch_content = fetch_content  # 是否获取URL的详细内容
        self.max_retries = min(max_retries, 3)  # 确保最多重试3次
        self.session = requests.Session()
        self.content_extractor = WebContentExtractor(timeout=timeout, max_retries=self.max_retries)
    
    def execute(self, query: str, context: Optional[str] = None) -> ToolResult:
        """执行Google Custom Search，支持网络错误处理和重试机制"""
        import time
        print("\n使用Google Custom Search搜索工具\n")
        # 构建API URL
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': self.max_results
        }
        
        items = []
        last_error = None
        retry_count = 0
        
        while retry_count < self.max_retries and not items:
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                
                # 检查错误
                if 'error' in data:
                    raise Exception(f"Google API error: {data['error'].get('message', 'Unknown error')}")
                
                # 提取结果
                items = data.get('items', [])
                
                if not items:
                    return ToolResult(
                        success=False,
                        result="",
                        error=f"No results found for '{query}'"
                    )
                    
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    # 等待一段时间再重试
                    time.sleep(1 * retry_count)  # 递增等待时间
        
        if not items:
            # 所有重试都失败，返回空结果
            return ToolResult(
                success=False,
                result="",
                error=f"Google自定义搜索失败，已尝试{self.max_retries}次。最后错误: {str(last_error)}"
            )
        
        # 格式化结果
        result_text = f"Google search results for '{query}':\n\n"
        for i, item in enumerate(items, 1):
            title = item.get('title', '')
            link = item.get('link', '')
            snippet = item.get('snippet', '')
            
            result_text += f"{i}. {title}\n"
            result_text += f"   {snippet}\n"
            result_text += f"   URL: {link}\n"
            
            # 如果启用了内容获取，获取URL的详细内容
            if self.fetch_content and link and link.startswith('http'):
                content_result = self.content_extractor.extract_content(link)
                if content_result['success']:
                    result_text += f"   Content: {content_result['content'][:5000]}"
                    if len(content_result['content']) > 5000:
                        result_text += "..."
                    result_text += "\n"
                else:
                    result_text += f"   Content: 无法获取内容: {content_result.get('error', '未知错误')}\n"
            
            result_text += "\n"
        print(f"搜索完成，共找到{len(items)}个结果")
        return ToolResult(
            success=True,
            result=result_text.strip(),
            metadata={
                "query": query,
                "results_count": len(items),
                "tool": "google_custom_search",
                "search_info": data.get('searchInformation', {})
            }
        )
    
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
                    "description": "搜索查询关键词"
                }
            },
            "required": ["query"]
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取工具元数据"""
        return {
            "name": "google_custom_search",
            "description": "Search using Google Custom Search API (requires API key)",
            "parameters": ["query"],
            "timeout": self.timeout,
            "max_results": self.max_results,
            "fetch_content": self.fetch_content,
            "version": "1.0"
        }
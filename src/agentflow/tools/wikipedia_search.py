"""
Wikipedia Search Tool
维基百科搜索工具 - 与real_search.py中的实现逻辑匹配
"""

import requests
import json
import re
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode, quote_plus
from bs4 import BeautifulSoup
from .base import BaseTool, ToolResult


class WikipediaSearchTool(BaseTool):
    """维基百科搜索工具 - 基于维基百科API"""
    
    def __init__(self, language: str = "en", timeout: int = 10, max_results: int = 5, max_retries: int = 3):
        super().__init__(
            name="wikipedia_search",
            description="Search Wikipedia articles using the official Wikipedia API"
        )
        self.max_results = max_results
        self.language = language  # 默认使用英文
        self.max_retries = min(max_retries, 3)  # 确保最多重试3次
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AgentFlow/1.0 (Wikipedia Search Tool)'
        })
    
    def execute(self, query: str, context: Optional[str] = None) -> ToolResult:
        """执行维基百科搜索，支持网络错误处理和重试机制"""
        import time
        print(f"执行维基百科搜索：{query}")
        # 构建搜索URL
        base_url = f"https://{self.language}.wikipedia.org/w/api.php"
        
        # 尝试多种搜索策略
        search_results = []
        last_error = None
        retry_count = 0
        
        # 策略1: 直接搜索原始查询
        while retry_count < self.max_retries and not search_results:
            try:
                search_results.extend(self._search_with_params(base_url, query, {
                    'srqiprofile': 'classic_noboostlinks',
                    'srwhat': 'text',
                    'srlimit': 3
                }))
                
                # 如果成功获取到结果，跳出重试循环
                if search_results:
                    break
                    
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    # 等待一段时间再重试
                    time.sleep(1 * retry_count)  # 递增等待时间
        
        # 如果结果不够好，尝试策略2: 搜索关键词组合
        if len(search_results) < 2:
            retry_count = 0
            # 提取关键词并重新组合
            keywords = query.split()
            if len(keywords) > 1:
                # 尝试前两个关键词的组合
                combined_query = " ".join(keywords[:2])
                
                while retry_count < self.max_retries and len(search_results) < 2:
                    try:
                        combined_results = self._search_with_params(base_url, combined_query, {
                            'srqiprofile': 'classic',
                            'srwhat': 'nearmatch',
                            'srlimit': 3
                        })
                        
                        # 添加不重复的结果
                        existing_ids = {r['pageid'] for r in search_results}
                        for result in combined_results:
                            if result['pageid'] not in existing_ids:
                                search_results.append(result)
                                existing_ids.add(result['pageid'])
                        
                        # 如果成功获取到结果，跳出重试循环
                        if len(search_results) >= 2:
                            break
                            
                    except Exception as e:
                        last_error = e
                        retry_count += 1
                        if retry_count < self.max_retries:
                            # 等待一段时间再重试
                            time.sleep(1 * retry_count)  # 递增等待时间
        
        # 如果结果仍然不够，尝试策略3: 最广泛搜索
        if len(search_results) < 2:
            retry_count = 0
            broad_query = keywords[0] if keywords else query
            
            while retry_count < self.max_retries and len(search_results) < 2:
                try:
                    broad_results = self._search_with_params(base_url, broad_query, {
                        'srqiprofile': 'popular_inclinks_pv',
                        'srwhat': 'text',
                        'srlimit': 5
                    })
                    
                    # 添加不重复的结果
                    existing_ids = {r['pageid'] for r in search_results}
                    for result in broad_results:
                        if result['pageid'] not in existing_ids:
                            search_results.append(result)
                            existing_ids.add(result['pageid'])
                    
                    # 如果成功获取到结果，跳出重试循环
                    if len(search_results) >= 2:
                        break
                        
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    if retry_count < self.max_retries:
                        # 等待一段时间再重试
                        time.sleep(1 * retry_count)  # 递增等待时间
        
        # 限制结果数量
        search_results = search_results[:self.max_results]
        
        if not search_results:
            return ToolResult(
                success=False,
                result="",
                error=f"维基百科搜索失败，已尝试所有策略。最后错误: {str(last_error)}"
            )
        
        # 获取页面完整内容（不仅仅是摘要）
        page_ids = [str(result['pageid']) for result in search_results]
        extract_params = {
            'action': 'query',
            'pageids': '|'.join(page_ids),
            'prop': 'extracts|info',
            'explaintext': 1,
            # 移除exintro=1参数以获取完整内容，而不仅仅是介绍
            'exlimit': self.max_results,
            'inprop': 'url|displaytitle',
            'format': 'json',
            'utf8': 1,
            'exsectionformat': 'plain'  # 使用纯文本格式
        }
        
        # 获取页面内容，添加重试机制
        retry_count = 0
        extract_data = None
        last_error = None
        
        while retry_count < self.max_retries and extract_data is None:
            try:
                extract_response = self.session.get(
                    base_url, 
                    params=extract_params, 
                    timeout=self.timeout
                )
                extract_response.raise_for_status()
                extract_data = extract_response.json()
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    # 等待一段时间再重试
                    time.sleep(1 * retry_count)  # 递增等待时间
        
        if extract_data is None:
            return ToolResult(
                success=False,
                result="",
                error=f"获取维基百科页面内容失败，已尝试{self.max_retries}次。最后错误: {str(last_error)}"
            )
        
        # 提取页面信息
        pages = extract_data.get('query', {}).get('pages', {})
        
        # 格式化结果
        result_text = f"Wikipedia search results for '{query}':\n\n"
        results = []
        
        for page_id, page_data in pages.items():
            if page_id == '-1':  # Missing page
                continue
            
            title = page_data.get('title', '')
            extract = page_data.get('extract', '')
            url = page_data.get('fullurl', '')
            
            # 对于完整内容，限制长度但保留更多信息
            if len(extract) > 2000:  # 增加内容长度限制
                extract = extract[:2000] + "..."
            
            if title and extract:
                result_text += f"{len(results)+1}. {title}\n"
                result_text += f"   {extract}\n"
                result_text += f"   URL: {url}\n\n"
                
                results.append({
                    'title': title,
                    'extract': extract,
                    'url': url,
                    'pageid': page_id
                })
        
        print(f"搜索完成，共找到{len(results)}个结果")
        return ToolResult(
            success=True,
            result=result_text.strip(),
            metadata={
                "query": query,
                "results_count": len(results),
                "tool": "wikipedia_search",
                "language": self.language,
                "source": "wikipedia_api",
                "full_content": True  # 标记返回的是完整内容
            }
        )
    
    def _search_with_params(self, base_url: str, query: str, additional_params: dict) -> list:
        """使用指定参数执行搜索并返回结果列表，支持网络错误处理和重试机制"""
        import time
        
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'utf8': 1,
            'srenablerewrites': 1,
            'srinfo': 'suggestion|rewrittenquery'
        }
        params.update(additional_params)
        
        # 执行搜索请求，添加重试机制
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                response = self.session.get(base_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                # 提取搜索结果
                search_results = data.get('query', {}).get('search', [])
                return search_results
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    # 等待一段时间再重试
                    time.sleep(1 * retry_count)  # 递增等待时间
        
        # 所有重试都失败了，返回空列表
        return []
    
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
                    "description": "维基百科搜索关键词"
                }
            },
            "required": ["query"]
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取工具元数据"""
        return {
            "name": "wikipedia_search",
            "description": "Search Wikipedia articles using the official Wikipedia API",
            "parameters": ["query"],
            "timeout": self.timeout,
            "max_results": self.max_results,
            "language": self.language,
            "version": "1.0"
        }


class EnhancedWikipediaSearchTool(BaseTool):
    """增强维基百科搜索工具 - 支持多语言和高级搜索"""
    
    def __init__(self, timeout: int = 10, max_results: int = 5, language: str = "en", 
                 include_images: bool = False, include_references: bool = False, max_retries: int = 3):
        super().__init__(
            name="enhanced_wikipedia_search",
            description="Enhanced Wikipedia search with additional features"
        )
        print("\n使用增强维基百科搜索工具\n")
        self.timeout = timeout
        self.max_results = max_results
        self.language = language
        self.include_images = include_images
        self.include_references = include_references
        self.max_retries = min(max_retries, 3)  # 确保最多重试3次
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AgentFlow/1.0 (Enhanced Wikipedia Search Tool)'
        })
    
    def execute(self, query: str, context: Optional[str] = None) -> ToolResult:
        """执行增强维基百科搜索，支持网络错误处理和重试机制"""
        import time
        print("\n使用增强维基百科搜索工具\n")
        # 首先执行基本搜索
        base_search = WikipediaSearchTool(
            timeout=self.timeout,
            max_results=self.max_results,
            language=self.language
        )
        
        search_result = base_search.execute(query)
        
        if not search_result.success:
            return search_result
        
        # 如果不需要额外功能，直接返回基本搜索结果
        if not self.include_images and not self.include_references:
            return search_result
        
        # 获取页面ID以获取额外信息
        base_url = f"https://{self.language}.wikipedia.org/w/api.php"
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'srlimit': self.max_results,
            'format': 'json',
            'utf8': 1
        }
        
        # 执行搜索请求，添加重试机制
        retry_count = 0
        search_data = None
        last_error = None
        
        while retry_count < self.max_retries and search_data is None:
            try:
                search_response = self.session.get(
                    base_url, 
                    params=search_params, 
                    timeout=self.timeout
                )
                search_response.raise_for_status()
                search_data = search_response.json()
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    # 等待一段时间再重试
                    time.sleep(1 * retry_count)  # 递增等待时间
        
        if search_data is None:
            return ToolResult(
                success=False,
                result="",
                error=f"维基百科搜索失败，已尝试{self.max_retries}次。最后错误: {str(last_error)}"
            )
        
        search_results = search_data.get('query', {}).get('search', [])
        page_ids = [str(result['pageid']) for result in search_results]
        
        # 构建额外参数
        extra_params = {
            'action': 'query',
            'pageids': '|'.join(page_ids),
            'prop': 'extracts|info',
            'explaintext': 1,
            'exintro': 1,
            'exlimit': self.max_results,
            'inprop': 'url|displaytitle',
            'format': 'json',
            'utf8': 1
        }
        
        # 添加图片信息
        if self.include_images:
            extra_params['prop'] += '|images'
            extra_params['imlimit'] = 5
        
        # 添加引用信息
        if self.include_references:
            extra_params['prop'] += '|extlinks'
            extra_params['ellimit'] = 10
        
        # 获取增强信息，添加重试机制
        retry_count = 0
        enhanced_data = None
        last_error = None
        
        while retry_count < self.max_retries and enhanced_data is None:
            try:
                enhanced_response = self.session.get(
                    base_url, 
                    params=extra_params, 
                    timeout=self.timeout
                )
                enhanced_response.raise_for_status()
                enhanced_data = enhanced_response.json()
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    # 等待一段时间再重试
                    time.sleep(1 * retry_count)  # 递增等待时间
        
        if enhanced_data is None:
            return ToolResult(
                success=False,
                result="",
                error=f"获取维基百科增强信息失败，已尝试{self.max_retries}次。最后错误: {str(last_error)}"
            )
        
        # 处理增强结果
        pages = enhanced_data.get('query', {}).get('pages', {})
        
        # 格式化结果
        result_text = f"Enhanced Wikipedia search results for '{query}':\n\n"
        results = []
        
        for page_id, page_data in pages.items():
            if page_id == '-1':  # Missing page
                continue
            
            title = page_data.get('title', '')
            extract = page_data.get('extract', '')
            url = page_data.get('fullurl', '')
            
            # 限制摘要长度
            if len(extract) > 500:
                extract = extract[:500] + "..."
            
            result_text += f"{len(results)+1}. {title}\n"
            result_text += f"   {extract}\n"
            result_text += f"   URL: {url}\n"
            
            # 添加图片信息
            if self.include_images and 'images' in page_data:
                images = page_data['images'][:3]  # 限制图片数量
                if images:
                    result_text += "   Images:\n"
                    for img in images:
                        img_title = img.get('title', '')
                        if img_title:
                            img_url = f"https://{self.language}.wikipedia.org/wiki/File:{quote_plus(img_title.replace('File:', ''))}"
                            result_text += f"     - {img_title}: {img_url}\n"
            
            # 添加外部链接信息
            if self.include_references and 'extlinks' in page_data:
                extlinks = page_data['extlinks'][:3]  # 限制链接数量
                if extlinks:
                    result_text += "   External Links:\n"
                    for link in extlinks:
                        link_url = link.get('*', '')
                        if link_url:
                            result_text += f"     - {link_url}\n"
            
            result_text += "\n"
            
            results.append({
                'title': title,
                'extract': extract,
                'url': url,
                'pageid': page_id,
                'images': page_data.get('images', []) if self.include_images else [],
                'extlinks': page_data.get('extlinks', []) if self.include_references else []
            })
        
        print(f"搜索完成，共找到{len(results)}个结果")
        return ToolResult(
            success=True,
            result=result_text.strip(),
            metadata={
                "query": query,
                "results_count": len(results),
                "tool": "enhanced_wikipedia_search",
                "language": self.language,
                "include_images": self.include_images,
                "include_references": self.include_references,
                "source": "wikipedia_api"
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
                    "description": "维基百科搜索关键词"
                }
            },
            "required": ["query"]
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取工具元数据"""
        return {
            "name": "enhanced_wikipedia_search",
            "description": "Enhanced Wikipedia search with additional features",
            "parameters": ["query"],
            "timeout": self.timeout,
            "max_results": self.max_results,
            "language": self.language,
            "include_images": self.include_images,
            "include_references": self.include_references,
            "version": "1.0"
        }
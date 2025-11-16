"""
API Keys Management
API密钥管理模块
"""

import os
import yaml
from typing import Optional, Dict
from pathlib import Path


class APIKeyManager:
    """API密钥管理类"""
    
    def __init__(self, config: dict = None):
        """
        初始化API密钥管理器
        
        Args:
            config (dict, optional): 配置字典，包含API密钥信息
        """
        # 如果没有提供config，则尝试从YAML配置文件加载
        if config is None:
            config = self._load_config_from_yaml()
        
        self.config = config or {}
        self._serpapi_key = None
        self._google_api_key = None
        self._google_search_engine_id = None
        self._bing_api_key = None
        self._brave_api_key = None
        self._tavily_api_key = None
        self._dashscope_api_key = None
    
    def _load_config_from_yaml(self) -> dict:
        """
        从YAML配置文件加载配置
        
        Returns:
            dict: 配置字典
        """
        # 定义可能的配置文件路径
        possible_paths = [
            Path("/root/agent/test_1/src/configs/config.yaml")
        ]
        
        # 尝试加载配置文件
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        print(f"✅ 成功从 {path} 加载配置")
                        return config
                except Exception as e:
                    print(f"⚠️  从 {path} 加载配置失败: {e}")
        
        print("⚠️  未找到配置文件，使用空配置")
        return {}
    
    @property
    def api_keys(self) -> Dict[str, str]:
        """获取所有API密钥的字典"""
        return {
            'google_api_key': self.get_google_api_key(),
            'google_search_engine_id': self.get_google_search_engine_id(),
            'bing_api_key': self.get_bing_api_key(),
            'brave_api_key': self.get_brave_api_key(),
            'tavily_api_key': self.get_tavily_api_key()
        }
    
    def get_serpapi_key(self) -> Optional[str]:
        """
        获取SerpAPI密钥
        
        Returns:
            Optional[str]: SerpAPI密钥，如果未找到则返回None
        """
        # 优先从实例变量获取
        if self._serpapi_key is not None:
            return self._serpapi_key
            
        # 从配置文件获取
        if self.config and "tools" in self.config and "search_apis" in self.config["tools"]:
            key = self.config["tools"]["search_apis"].get("serpapi_key")
            if key:
                self._serpapi_key = key
                return self._serpapi_key
                
        # 从环境变量获取
        env_key = os.getenv("SERPAPI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if env_key:
            self._serpapi_key = env_key
            return self._serpapi_key
            
        return None
    
    def get_google_api_key(self) -> Optional[str]:
        """
        获取Google API密钥
        
        Returns:
            Optional[str]: Google API密钥，如果未找到则返回None
        """
        # 优先从实例变量获取
        if self._google_api_key is not None:
            return self._google_api_key
            
        # 从配置文件获取
        if self.config and "tools" in self.config and "search_apis" in self.config["tools"]:
            key = self.config["tools"]["search_apis"].get("google_api_key")
            if key:
                self._google_api_key = key
                return self._google_api_key
                
        # 从环境变量获取
        env_key = os.getenv("GOOGLE_API_KEY")
        if env_key:
            self._google_api_key = env_key
            return self._google_api_key
            
        return None
    
    def get_google_search_engine_id(self) -> Optional[str]:
        """
        获取Google搜索引擎ID
        
        Returns:
            Optional[str]: Google搜索引擎ID，如果未找到则返回None
        """
        # 优先从实例变量获取
        if self._google_search_engine_id is not None:
            return self._google_search_engine_id
            
        # 从配置文件获取
        if self.config and "tools" in self.config and "search_apis" in self.config["tools"]:
            engine_id = self.config["tools"]["search_apis"].get("google_search_engine_id")
            if engine_id:
                self._google_search_engine_id = engine_id
                return self._google_search_engine_id
                
        # 从环境变量获取
        env_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        if env_id:
            self._google_search_engine_id = env_id
            return self._google_search_engine_id
            
        return None
    
    def get_bing_api_key(self) -> Optional[str]:
        """
        获取Bing API密钥
        
        Returns:
            Optional[str]: Bing API密钥，如果未找到则返回None
        """
        # 优先从实例变量获取
        if self._bing_api_key is not None:
            return self._bing_api_key
            
        # 从配置文件获取
        if self.config and "tools" in self.config and "search_apis" in self.config["tools"]:
            key = self.config["tools"]["search_apis"].get("bing_api_key")
            if key:
                self._bing_api_key = key
                return self._bing_api_key
                
        # 从环境变量获取
        env_key = os.getenv("BING_API_KEY")
        if env_key:
            self._bing_api_key = env_key
            return self._bing_api_key
            
        return None
    
    def get_brave_api_key(self) -> Optional[str]:
        """
        获取Brave API密钥
        
        Returns:
            Optional[str]: Brave API密钥，如果未找到则返回None
        """
        # 优先从实例变量获取
        if self._brave_api_key is not None:
            return self._brave_api_key
            
        # 从配置文件获取
        if self.config and "tools" in self.config and "search_apis" in self.config["tools"]:
            key = self.config["tools"]["search_apis"].get("brave_api_key")
            if key:
                self._brave_api_key = key
                return self._brave_api_key
                
        # 从环境变量获取
        env_key = os.getenv("BRAVE_API_KEY")
        if env_key:
            self._brave_api_key = env_key
            return self._brave_api_key
            
        return None
    
    def get_tavily_api_key(self) -> Optional[str]:
        """
        获取Tavily API密钥
        
        Returns:
            Optional[str]: Tavily API密钥，如果未找到则返回None
        """
        # 优先从实例变量获取
        if self._tavily_api_key is not None:
            return self._tavily_api_key
            
        # 从配置文件获取
        if self.config and "tools" in self.config and "search_apis" in self.config["tools"]:
            key = self.config["tools"]["search_apis"].get("tavily_api_key")
            if key:
                self._tavily_api_key = key
                return self._tavily_api_key
                
        # 从环境变量获取
        env_key = os.getenv("TAVILY_API_KEY")
        if env_key:
            self._tavily_api_key = env_key
            return self._tavily_api_key
            
        return None
    
    def get_dashscope_api_key(self) -> Optional[str]:
        """
        获取DashScope API密钥
        
        Returns:
            Optional[str]: DashScope API密钥，如果未找到则返回None
        """
        # 优先从实例变量获取
        if self._dashscope_api_key is not None:
            return self._dashscope_api_key
            
        # 从配置文件获取
        if self.config and "tools" in self.config and "search_apis" in self.config["tools"]:
            key = self.config["tools"]["search_apis"].get("dashscope_api_key")
            if key:
                self._dashscope_api_key = key
                return self._dashscope_api_key
                
        # 从环境变量获取
        env_key = os.getenv("DASHSCOPE_API_KEY")
        if env_key:
            self._dashscope_api_key = env_key
            return self._dashscope_api_key
            
        return None
    
    def set_serpapi_key(self, key: str):
        """设置SerpAPI密钥"""
        self._serpapi_key = key
    
    def set_google_api_key(self, key: str):
        """设置Google API密钥"""
        self._google_api_key = key
    
    def set_google_search_engine_id(self, engine_id: str):
        """设置Google搜索引擎ID"""
        self._google_search_engine_id = engine_id
    
    def set_bing_api_key(self, key: str):
        """设置Bing API密钥"""
        self._bing_api_key = key
    
    def set_brave_api_key(self, key: str):
        """设置Brave API密钥"""
        self._brave_api_key = key
    
    def set_tavily_api_key(self, key: str):
        """设置Tavily API密钥"""
        self._tavily_api_key = key
    
    def set_dashscope_api_key(self, key: str):
        """设置DashScope API密钥"""
        self._dashscope_api_key = key
    
    def get_available_services(self) -> Dict[str, bool]:
        """
        获取可用的服务列表及其状态
        
        Returns:
            Dict[str, bool]: 服务名称和可用性状态的映射
        """
        return {
            'google_custom_search': bool(self.get_google_api_key() and self.get_google_search_engine_id()),
            'bing_web_search': bool(self.get_bing_api_key()),
            'brave_search': bool(self.get_brave_api_key()),
            'tavily_search': bool(self.get_tavily_api_key()),
            'dashscope_api': bool(self.get_dashscope_api_key())
        }
    
    def print_setup_instructions(self):
        """打印API密钥设置说明"""
        print("📝 API密钥设置说明:")
        print("要使用付费搜索工具，请设置以下环境变量之一或在配置文件中提供相应密钥:")
        print("")
        print("Google Custom Search:")
        print("  • GOOGLE_API_KEY=your_google_api_key")
        print("  • GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id")
        print("")
        print("Bing Web Search:")
        print("  • BING_API_KEY=your_bing_api_key")
        print("")
        print("Brave Search:")
        print("  • BRAVE_API_KEY=your_brave_api_key")
        print("")
        print("Tavily Search (推荐):")
        print("  • TAVILY_API_KEY=your_tavily_api_key")
        print("")
        print("DashScope API (通义千问):")
        print("  • DASHSCOPE_API_KEY=your_dashscope_api_key")
        print("")
        print("💡 提示: 您也可以通过创建 'search_api_keys.json' 文件来配置密钥")
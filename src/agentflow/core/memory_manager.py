"""
Memory Manager for AgentFlow
记忆管理器：管理多个查询的记忆，提供问题间记忆隔离机制
"""

import uuid
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .memory import Memory, MemoryEntry, MemoryEntryType


@dataclass
class MemorySession:
    """记忆会话类，封装单个查询的记忆和元数据"""
    session_id: str
    query: str
    context: str
    memory: Memory
    created_at: float
    last_accessed: float
    metadata: Dict[str, Any]


class MemoryManager:
    """
    记忆管理器类
    
    职责：
    1. 管理多个查询的记忆
    2. 提供问题间记忆隔离机制
    3. 支持记忆的创建、访问、更新和删除
    4. 提供记忆持久化功能
    """
    
    def __init__(self, max_sessions: int = 10, max_entries_per_session: int = 100):
        """
        初始化记忆管理器
        
        Args:
            max_sessions: 最大会话数
            max_entries_per_session: 每个会话的最大条目数
        """
        self.sessions: Dict[str, MemorySession] = {}
        self.max_sessions = max_sessions
        self.max_entries_per_session = max_entries_per_session
        self.default_timeout = 3600  # 1小时超时
        
    def create_session(self, query: str, context: str = "", required_skills: List[str] = None, 
                      metadata: Dict[str, Any] = None) -> str:
        """
        创建新的记忆会话
        
        Args:
            query: 查询内容
            context: 查询上下文
            required_skills: 所需技能列表
            metadata: 会话元数据
            
        Returns:
            str: 会话ID
        """
        # 生成唯一会话ID
        session_id = str(uuid.uuid4())
        
        # 创建记忆实例
        memory = Memory(max_entries=self.max_entries_per_session)
        
        # 初始化记忆
        memory.initialize(query, context, required_skills)
        
        # 创建会话
        current_time = time.time()
        session = MemorySession(
            session_id=session_id,
            query=query,
            context=context,
            memory=memory,
            created_at=current_time,
            last_accessed=current_time,
            metadata=metadata or {}
        )
        
        # 添加到会话字典
        self.sessions[session_id] = session
        
        # 如果会话数超过限制，删除最旧的会话
        if len(self.sessions) > self.max_sessions:
            self._cleanup_old_sessions()
            
        return session_id
    
    def get_session(self, session_id: str) -> Optional[MemorySession]:
        """
        获取记忆会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            Optional[MemorySession]: 记忆会话，如果不存在则返回None
        """
        if session_id not in self.sessions:
            return None
            
        # 更新最后访问时间
        self.sessions[session_id].last_accessed = time.time()
        return self.sessions[session_id]
    
    def get_memory(self, session_id: str) -> Optional[Memory]:
        """
        获取记忆实例
        
        Args:
            session_id: 会话ID
            
        Returns:
            Optional[Memory]: 记忆实例，如果不存在则返回None
        """
        session = self.get_session(session_id)
        return session.memory if session else None
    
    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """
        更新会话元数据
        
        Args:
            session_id: 会话ID
            metadata: 新的元数据
            
        Returns:
            bool: 是否更新成功
        """
        if session_id not in self.sessions:
            return False
            
        self.sessions[session_id].metadata.update(metadata)
        self.sessions[session_id].last_accessed = time.time()
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除记忆会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否删除成功
        """
        if session_id not in self.sessions:
            return False
            
        del self.sessions[session_id]
        return True
    
    def clear_session(self, session_id: str) -> bool:
        """
        清空会话记忆
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否清空成功
        """
        if session_id not in self.sessions:
            return False
            
        self.sessions[session_id].memory.clear()
        self.sessions[session_id].last_accessed = time.time()
        return True
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有会话的基本信息
        
        Returns:
            List[Dict[str, Any]]: 会话信息列表
        """
        sessions_info = []
        for session_id, session in self.sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "query": session.query,
                "created_at": session.created_at,
                "last_accessed": session.last_accessed,
                "entries_count": len(session.memory.entries),
                "current_turn": session.memory.current_turn,
                "terminated": session.memory.terminated
            })
            
        return sessions_info
    
    def _cleanup_old_sessions(self):
        """清理最旧的会话"""
        if not self.sessions:
            return
            
        # 按最后访问时间排序，删除最旧的会话
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # 删除最旧的会话，直到会话数在限制内
        while len(self.sessions) > self.max_sessions:
            oldest_session_id = sorted_sessions[0][0]
            del self.sessions[oldest_session_id]
            sorted_sessions.pop(0)
    
    def cleanup_expired_sessions(self, timeout: Optional[float] = None):
        """
        清理超时的会话
        
        Args:
            timeout: 超时时间（秒），如果为None则使用默认超时时间
        """
        if timeout is None:
            timeout = self.default_timeout
            
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_accessed > timeout:
                expired_sessions.append(session_id)
                
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    def save_session_to_file(self, session_id: str, file_path: str) -> bool:
        """
        保存会话记忆到文件
        
        Args:
            session_id: 会话ID
            file_path: 文件路径
            
        Returns:
            bool: 是否保存成功
        """
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        
        # 准备保存数据
        data = {
            "session_id": session.session_id,
            "query": session.query,
            "context": session.context,
            "created_at": session.created_at,
            "last_accessed": session.last_accessed,
            "metadata": session.metadata,
            "memory": {}
        }
        
        # 获取记忆数据
        memory_data = {
            "initial_query": session.memory.initial_query,
            "initial_context": session.memory.initial_context,
            "required_skills": session.memory.required_skills,
            "current_turn": session.memory.current_turn,
            "terminated": session.memory.terminated,
            "termination_reason": session.memory.termination_reason,
            "performance_metrics": session.memory.get_performance_metrics(),
            "turn_entry_count": session.memory.turn_entry_count,
            "turn_start_time": session.memory.turn_start_time,
            "entries": [entry.to_dict() for entry in session.memory.entries]
        }
        
        data["memory"] = memory_data
        
        # 保存到文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def load_session_from_file(self, file_path: str) -> Optional[str]:
        """
        从文件加载会话记忆
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[str]: 会话ID，如果加载失败则返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 创建记忆实例
            memory = Memory(max_entries=self.max_entries_per_session)
            
            # 恢复记忆数据
            memory.initial_query = data["memory"].get("initial_query", "")
            memory.initial_context = data["memory"].get("initial_context", "")
            memory.required_skills = data["memory"].get("required_skills", [])
            memory.current_turn = data["memory"].get("current_turn", 0)
            memory.terminated = data["memory"].get("terminated", False)
            memory.termination_reason = data["memory"].get("termination_reason", "")
            memory.turn_entry_count = data["memory"].get("turn_entry_count", {0: 0})
            memory.turn_start_time = data["memory"].get("turn_start_time", {0: time.time()})
            
            # 加载条目
            memory.entries = []
            for entry_data in data["memory"].get("entries", []):
                memory.entries.append(MemoryEntry.from_dict(entry_data))
            
            # 更新使用的工具集合
            memory.used_tools = set(entry.selected_tool for entry in memory.entries)
            
            # 创建会话
            session = MemorySession(
                session_id=data["session_id"],
                query=data["query"],
                context=data["context"],
                memory=memory,
                created_at=data["created_at"],
                last_accessed=data["last_accessed"],
                metadata=data.get("metadata", {})
            )
            
            # 添加到会话字典
            self.sessions[data["session_id"]] = session
            
            return data["session_id"]
        except Exception:
            return None
    
    def get_all_training_data(self) -> List[Dict[str, Any]]:
        """
        获取所有会话的训练数据
        
        Returns:
            List[Dict[str, Any]]: 所有会话的训练数据
        """
        training_data = []
        for session_id, session in self.sessions.items():
            session_training_data = session.memory.get_training_data()
            session_training_data["session_id"] = session_id
            session_training_data["session_metadata"] = session.metadata
            training_data.append(session_training_data)
            
        return training_data
    
    def clear_all_sessions(self):
        """清空所有会话"""
        self.sessions.clear()
"""
通讯协议基础接口模块
定义了所有通讯协议必须实现的接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, AsyncGenerator
import asyncio

class CommunicationProtocol(ABC):
    """
    通讯协议基础接口类
    定义了所有通讯协议必须实现的方法
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        初始化通讯协议
        """
        pass
    
    @abstractmethod
    async def connect(self) -> None:
        """
        建立连接
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        断开连接
        """
        pass
    
    @abstractmethod
    async def send_audio(self, audio_data: bytes) -> None:
        """
        发送音频数据
        
        Args:
            audio_data: 音频数据字节流
        """
        pass
    
    @abstractmethod
    async def receive_audio(self) -> AsyncGenerator[bytes, None]:
        """
        接收音频数据
        
        Yields:
            音频数据字节流
        """
        pass
    
    @abstractmethod
    async def send_text(self, text: str) -> None:
        """
        发送文本数据
        
        Args:
            text: 要发送的文本
        """
        pass
    
    @abstractmethod
    async def receive_text(self) -> AsyncGenerator[str, None]:
        """
        接收文本数据
        
        Yields:
            接收到的文本
        """
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        检查连接状态
        
        Returns:
            是否已连接
        """
        pass
    
    @abstractmethod
    async def set_callbacks(self, callbacks: Dict[str, Callable]) -> None:
        """
        设置回调函数
        
        Args:
            callbacks: 回调函数字典
        """
        pass
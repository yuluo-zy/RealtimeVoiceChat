"""
WebSocket通讯协议实现模块
实现了基于WebSocket的通讯协议
"""

import json
import asyncio
from typing import Dict, Callable, AsyncGenerator, Optional
from fastapi import WebSocket
from .base import CommunicationProtocol
from .config import WebSocketConfig

class WebSocketProtocol(CommunicationProtocol):
    """
    WebSocket通讯协议实现类
    实现了基于WebSocket的实时通讯功能
    """
    
    def __init__(self, config: WebSocketConfig):
        """
        初始化WebSocket协议
        
        Args:
            config: WebSocket配置对象
        """
        self.config = config
        self.websocket: Optional[WebSocket] = None
        self.callbacks: Dict[str, Callable] = {}
        self._connected = False
        self._message_queue = asyncio.Queue()
        self._audio_queue = asyncio.Queue()
    
    async def initialize(self) -> None:
        """初始化WebSocket连接"""
        # WebSocket的初始化在FastAPI的WebSocket路由中完成
        pass
    
    async def connect(self) -> None:
        """建立WebSocket连接"""
        if not self.websocket:
            raise RuntimeError("WebSocket对象未初始化")
        await self.websocket.accept()
        self._connected = True
    
    async def disconnect(self) -> None:
        """断开WebSocket连接"""
        if self.websocket:
            await self.websocket.close()
        self._connected = False
    
    async def send_audio(self, audio_data: bytes) -> None:
        """
        发送音频数据
        
        Args:
            audio_data: 音频数据字节流
        """
        if not self._connected:
            raise RuntimeError("WebSocket未连接")
        await self.websocket.send_bytes(audio_data)
    
    async def receive_audio(self) -> AsyncGenerator[bytes, None]:
        """
        接收音频数据
        
        Yields:
            音频数据字节流
        """
        while self._connected:
            try:
                data = await self.websocket.receive_bytes()
                yield data
            except Exception as e:
                if "connection is closed" in str(e):
                    break
                raise
    
    async def send_text(self, text: str) -> None:
        """
        发送文本数据
        
        Args:
            text: 要发送的文本
        """
        if not self._connected:
            raise RuntimeError("WebSocket未连接")
        await self.websocket.send_text(text)
    
    async def receive_text(self) -> AsyncGenerator[str, None]:
        """
        接收文本数据
        
        Yields:
            接收到的文本
        """
        while self._connected:
            try:
                text = await self.websocket.receive_text()
                yield text
            except Exception as e:
                if "connection is closed" in str(e):
                    break
                raise
    
    async def is_connected(self) -> bool:
        """
        检查连接状态
        
        Returns:
            是否已连接
        """
        return self._connected
    
    async def set_callbacks(self, callbacks: Dict[str, Callable]) -> None:
        """
        设置回调函数
        
        Args:
            callbacks: 回调函数字典
        """
        self.callbacks = callbacks 
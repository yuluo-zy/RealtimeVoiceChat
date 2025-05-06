"""
WebRTC通讯协议实现模块
实现了基于WebRTC的通讯协议
"""

import json
import asyncio
from typing import Dict, Callable, AsyncGenerator, Optional, Any
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from .base import CommunicationProtocol
from .config import WebRTCConfig
import numpy as np

class WebRTCProtocol(CommunicationProtocol):
    """
    WebRTC通讯协议实现类
    实现了基于WebRTC的实时通讯功能
    """
    
    def __init__(self, config: WebRTCConfig):
        """
        初始化WebRTC协议
        
        Args:
            config: WebRTC配置对象
        """
        self.config = config
        self.pc: Optional[RTCPeerConnection] = None
        self.callbacks: Dict[str, Callable] = {}
        self._connected = False
        self._audio_track: Optional[MediaStreamTrack] = None
        self._message_queue = asyncio.Queue()
        self._audio_queue = asyncio.Queue()
        
        # 创建RTC配置
        self.rtc_config = RTCConfiguration(
            iceServers=[
                {"urls": server} for server in config.stun_servers
            ] + [
                {
                    "urls": server["url"],
                    "username": server.get("username"),
                    "credential": server.get("credential")
                } for server in config.turn_servers
            ],
            iceTransportPolicy=config.ice_transport_policy,
            bundlePolicy=config.bundle_policy,
            rtcpMuxPolicy=config.rtcp_mux_policy,
            iceCandidatePoolSize=config.ice_candidate_pool_size
        )
    
    async def initialize(self) -> None:
        """初始化WebRTC连接"""
        self.pc = RTCPeerConnection(configuration=self.rtc_config)
        
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if self.pc.connectionState == "connected":
                self._connected = True
            elif self.pc.connectionState in ["failed", "disconnected", "closed"]:
                self._connected = False
        
        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                self._audio_track = track
                asyncio.create_task(self._handle_audio_track(track))
    
    async def connect(self) -> None:
        """建立WebRTC连接"""
        if not self.pc:
            raise RuntimeError("WebRTC未初始化")
        # WebRTC的连接建立是通过信令服务器完成的
        pass
    
    async def disconnect(self) -> None:
        """断开WebRTC连接"""
        if self.pc:
            await self.pc.close()
        self._connected = False
    
    async def send_audio(self, audio_data: bytes) -> None:
        """
        发送音频数据
        
        Args:
            audio_data: 音频数据字节流
        """
        if not self._connected:
            raise RuntimeError("WebRTC未连接")
            
        if not self.audio_track:
            raise RuntimeError("音频轨道未初始化")
            
        try:
            # 将字节数据转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 创建音频帧
            from aiortc.mediastreams import MediaStreamTrack
            frame = MediaStreamTrack.create_audio_frame(
                audio_array,
                sample_rate=24000,
                num_channels=1
            )
            
            # 发送音频帧
            await self.audio_track.send(frame)
        except Exception as e:
            logger.error(f"发送音频数据时出错: {str(e)}")
            raise
    
    async def receive_audio(self) -> AsyncGenerator[bytes, None]:
        """
        接收音频数据
        
        Yields:
            音频数据字节流
        """
        while self._connected:
            try:
                data = await self._audio_queue.get()
                yield data
            except Exception as e:
                if not self._connected:
                    break
                raise
    
    async def send_text(self, text: str) -> None:
        """
        发送文本数据
        
        Args:
            text: 要发送的文本
        """
        if not self._connected:
            raise RuntimeError("WebRTC未连接")
            
        if not self.data_channel:
            raise RuntimeError("数据通道未初始化")
            
        try:
            # 发送文本消息
            self.data_channel.send(text)
        except Exception as e:
            logger.error(f"发送文本数据时出错: {str(e)}")
            raise
    
    async def receive_text(self) -> AsyncGenerator[str, None]:
        """
        接收文本数据
        
        Yields:
            接收到的文本
        """
        while self._connected:
            try:
                text = await self._message_queue.get()
                yield text
            except Exception as e:
                if not self._connected:
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
    
    async def _handle_audio_track(self, track: MediaStreamTrack) -> None:
        """
        处理音频轨道数据
        
        Args:
            track: 媒体轨道对象
        """
        while self._connected:
            try:
                frame = await track.recv()
                # 处理音频帧数据
                await self._audio_queue.put(frame.to_ndarray().tobytes())
            except Exception as e:
                if not self._connected:
                    break
                raise
    
    async def create_offer(self) -> RTCSessionDescription:
        """
        创建WebRTC offer
        
        Returns:
            RTCSessionDescription对象
        """
        if not self.pc:
            raise RuntimeError("WebRTC未初始化")
            
        # 创建数据通道
        self.data_channel = self.pc.createDataChannel("text")
        self.data_channel.onmessage = lambda msg: asyncio.create_task(self._handle_text_message(msg.data))
        
        # 创建音频轨道
        self.audio_track = self.pc.addTransceiver("audio", direction="sendrecv").track
        
        # 创建offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        return self.pc.localDescription
    
    async def handle_answer(self, answer: RTCSessionDescription) -> None:
        """
        处理WebRTC answer
        
        Args:
            answer: RTCSessionDescription对象
        """
        if not self.pc:
            raise RuntimeError("WebRTC未初始化")
            
        # 设置远程描述
        await self.pc.setRemoteDescription(RTCSessionDescription(
            sdp=answer["sdp"],
            type=answer["type"]
        ))
        
        # 等待连接建立
        while self.pc.connectionState != "connected":
            await asyncio.sleep(0.1)
            if self.pc.connectionState in ["failed", "disconnected", "closed"]:
                raise RuntimeError(f"WebRTC连接失败: {self.pc.connectionState}")
    
    async def _handle_text_message(self, message: str) -> None:
        """
        处理文本消息
        
        Args:
            message: 接收到的文本消息
        """
        try:
            data = json.loads(message)
            if "type" in data and data["type"] in self.callbacks:
                await self.callbacks[data["type"]](data.get("content", ""))
        except json.JSONDecodeError:
            logger.error(f"无效的JSON消息: {message}")
        except Exception as e:
            logger.error(f"处理文本消息时出错: {str(e)}") 
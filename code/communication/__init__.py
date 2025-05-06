"""
通讯模块初始化文件
包含所有通讯相关的接口和实现
"""

from .base import CommunicationProtocol
from .websocket_protocol import WebSocketProtocol
from .webrtc_protocol import WebRTCProtocol
from .config import CommunicationConfig, ProtocolType

__all__ = ['CommunicationProtocol', 'WebSocketProtocol', 'WebRTCProtocol', 'CommunicationConfig', 'ProtocolType'] 
"""
通讯配置模块
定义了通讯相关的配置项
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

class ProtocolType(Enum):
    """通讯协议类型枚举"""
    WEBSOCKET = "websocket"
    WEBRTC = "webrtc"

@dataclass
class WebSocketConfig:
    """WebSocket配置类"""
    host: str = "0.0.0.0"
    port: int = 8000
    path: str = "/ws"
    ping_interval: float = 20.0
    ping_timeout: float = 20.0
    max_message_size: int = 1024 * 1024  # 1MB

@dataclass
class WebRTCConfig:
    """WebRTC配置类"""
    stun_servers: list[str] = None
    turn_servers: list[Dict[str, Any]] = None
    ice_transport_policy: str = "all"
    bundle_policy: str = "max-bundle"
    rtcp_mux_policy: str = "require"
    ice_candidate_pool_size: int = 0
    
    def __post_init__(self):
        if self.stun_servers is None:
            self.stun_servers = ["stun:stun.l.google.com:19302"]
        if self.turn_servers is None:
            self.turn_servers = []

@dataclass
class CommunicationConfig:
    """通讯配置主类"""
    protocol_type: ProtocolType = ProtocolType.WEBSOCKET
    websocket_config: Optional[WebSocketConfig] = None
    webrtc_config: Optional[WebRTCConfig] = None
    
    def __post_init__(self):
        if self.websocket_config is None:
            self.websocket_config = WebSocketConfig()
        if self.webrtc_config is None:
            self.webrtc_config = WebRTCConfig() 
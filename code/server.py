"""
服务器主模块
实现了基于FastAPI的Web服务器，支持WebSocket和WebRTC通讯
"""

from queue import Queue, Empty
import logging
from logsetup import setup_logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, Response, FileResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import json

from communication.config import (
    CommunicationConfig,
    ProtocolType,
    WebSocketConfig,
    WebRTCConfig
)
from communication.websocket_protocol import WebSocketProtocol
from communication.webrtc_protocol import WebRTCProtocol
from speech_pipeline_manager import SpeechPipelineManager
from audio_in import AudioInputProcessor
from upsample_overlap import UpsampleOverlap

# 配置参数
USE_SSL = False
TTS_START_ENGINE = "coqui"
LLM_START_PROVIDER = "ollama"
LLM_START_MODEL = "hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M"
NO_THINK = False
DIRECT_STREAM = False

# 通讯配置
COMM_CONFIG = CommunicationConfig(
    protocol_type=ProtocolType.WEBSOCKET,  # 默认使用WebSocket
    websocket_config=WebSocketConfig(
        host="0.0.0.0",
        port=8000,
        path="/ws"
    ),
    webrtc_config=WebRTCConfig(
        stun_servers=["stun:stun.l.google.com:19302"]
    )
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    管理应用程序的生命周期
    初始化全局组件并在关闭时清理资源
    """
    logger.info("🖥️▶️ 服务器启动中")
    
    # 初始化全局组件
    app.state.SpeechPipelineManager = SpeechPipelineManager(
        tts_engine=TTS_START_ENGINE,
        llm_provider=LLM_START_PROVIDER,
        llm_model=LLM_START_MODEL,
        no_think=NO_THINK
    )
    
    app.state.Upsampler = UpsampleOverlap()
    app.state.AudioInputProcessor = AudioInputProcessor(
        "en",
        is_orpheus=TTS_START_ENGINE=="orpheus",
        pipeline_latency=app.state.SpeechPipelineManager.full_output_pipeline_latency / 1000
    )
    
    yield
    
    logger.info("🖥️⏹️ 服务器关闭中")
    app.state.AudioInputProcessor.shutdown()

# 创建FastAPI应用
app = FastAPI(lifespan=lifespan)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index() -> HTMLResponse:
    """返回主页"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/favicon.ico")
async def favicon():
    """返回网站图标"""
    return FileResponse("static/favicon.ico")

# WebSocket路由
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接处理"""
    protocol = WebSocketProtocol(COMM_CONFIG.websocket_config)
    protocol.websocket = websocket
    
    try:
        await protocol.initialize()
        await protocol.connect()
        
        # 设置回调函数
        callbacks = {
            "on_partial": lambda txt: asyncio.create_task(protocol.send_text(json.dumps({"type": "partial", "text": txt}))),
            "on_final": lambda txt: asyncio.create_task(protocol.send_text(json.dumps({"type": "final", "text": txt}))),
            "on_audio": lambda audio: asyncio.create_task(protocol.send_audio(audio))
        }
        await protocol.set_callbacks(callbacks)
        
        # 处理音频数据
        async for audio_data in protocol.receive_audio():
            await app.state.AudioInputProcessor.process_audio(audio_data)
            
    except WebSocketDisconnect:
        logger.info("WebSocket连接断开")
    except Exception as e:
        logger.error(f"WebSocket错误: {str(e)}")
    finally:
        await protocol.disconnect()

# WebRTC路由
@app.post("/webrtc/offer")
async def webrtc_offer(offer: dict):
    """处理WebRTC offer请求"""
    try:
        protocol = WebRTCProtocol(COMM_CONFIG.webrtc_config)
        await protocol.initialize()
        
        # 设置回调函数
        callbacks = {
            "on_partial": lambda txt: asyncio.create_task(protocol.send_text(json.dumps({"type": "partial", "text": txt}))),
            "on_final": lambda txt: asyncio.create_task(protocol.send_text(json.dumps({"type": "final", "text": txt}))),
            "on_audio": lambda audio: asyncio.create_task(protocol.send_audio(audio))
        }
        await protocol.set_callbacks(callbacks)
        
        # 处理offer
        answer = await protocol.create_offer()
        return {"sdp": answer.sdp, "type": answer.type}
    except Exception as e:
        logger.error(f"WebRTC offer处理错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webrtc/answer")
async def webrtc_answer(answer: dict):
    """处理WebRTC answer请求"""
    try:
        protocol = WebRTCProtocol(COMM_CONFIG.webrtc_config)
        await protocol.handle_answer(answer)
        
        # 处理音频数据
        async for audio_data in protocol.receive_audio():
            await app.state.AudioInputProcessor.process_audio(audio_data)
            
        return {"status": "success"}
    except Exception as e:
        logger.error(f"WebRTC answer处理错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=COMM_CONFIG.websocket_config.host,
        port=COMM_CONFIG.websocket_config.port,
        ssl_keyfile="key.pem" if USE_SSL else None,
        ssl_certfile="cert.pem" if USE_SSL else None
    )

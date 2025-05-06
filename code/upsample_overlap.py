import base64
import numpy as np
from scipy.signal import resample_poly
from typing import Optional

class UpsampleOverlap:
    """
    管理带有重叠处理的音频块上采样。

    该类处理连续的音频块，使用 `scipy.signal.resample_poly` 将音频从24kHz上采样到48kHz，
    并管理块之间的重叠以减轻边界伪影。处理后的上采样音频段以Base64编码字符串的形式返回。
    它维护内部状态以正确处理调用之间的重叠。
    """
    def __init__(self):
        """
        初始化上采样重叠处理器。

        设置跟踪前一个音频块及其重采样版本所需的内部状态，
        以便在处理过程中处理重叠。
        """
        self.previous_chunk: Optional[np.ndarray] = None
        self.resampled_previous_chunk: Optional[np.ndarray] = None

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        处理输入的音频块，进行上采样，并返回相关段作为Base64编码。

        将原始PCM字节（假设为16位有符号整数）块转换为float32 numpy数组，
        进行归一化，并从24kHz上采样到48kHz。它使用前一个块的数据创建重叠，
        重采样组合后的音频，并提取主要对应于当前块的中心部分，
        使用重叠来平滑过渡。更新状态以供下一次调用使用。
        提取的音频段被转换回16位PCM字节并以Base64编码字符串的形式返回。

        参数:
            chunk: 原始音频数据字节（预期为PCM 16位有符号整数格式）。

        返回:
            表示上采样音频段的Base64编码字符串，已针对重叠进行调整。
            如果输入块为空，则返回空字符串。
        """
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        # 优雅地处理潜在的空块
        if audio_int16.size == 0:
             return "" # 对于空输入块返回空字符串

        audio_float = audio_int16.astype(np.float32) / 32768.0

        # 首先独立上采样当前块，用于状态和第一个块的逻辑
        upsampled_current_chunk = resample_poly(audio_float, 48000, 24000)

        if self.previous_chunk is None:
            # 第一个块：输出其上采样版本的前半部分
            half = len(upsampled_current_chunk) // 2
            part = upsampled_current_chunk[:half]
        else:
            # 后续块：将前一个浮点块与当前浮点块组合
            combined = np.concatenate((self.previous_chunk, audio_float))
            # 上采样组合后的块
            up = resample_poly(combined, 48000, 24000)

            # 计算提取中间部分的长度和索引
            # 确保self.resampled_previous_chunk不为None（由于外部if，这里不应该发生）
            assert self.resampled_previous_chunk is not None
            prev_len = len(self.resampled_previous_chunk) # 前一个块的上采样版本的长度
            h_prev = prev_len // 2 # 前一个块的上采样版本的中点索引

            # *** 修正的索引计算（恢复到原始） ***
            # 计算对应于当前块主要贡献的部分的结束索引
            # 这个索引表示组合'up'数组中当前块贡献的中点
            h_cur = (len(up) - prev_len) // 2 + prev_len

            part = up[h_prev:h_cur]

        # 更新下一次迭代的状态
        self.previous_chunk = audio_float
        self.resampled_previous_chunk = upsampled_current_chunk # 存储当前块的上采样版本用于下一次重叠

        # 将提取的部分转换回PCM16字节并编码
        pcm = (part * 32767).astype(np.int16).tobytes()
        return base64.b64encode(pcm).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        在处理完所有块后返回剩余的上采样音频段。

        在最后一次调用 `get_base64_chunk` 后，状态保存了最后一个输入块的上采样版本
        (`self.resampled_previous_chunk`)。此方法返回该完整的上采样块，
        转换为16位PCM字节并以Base64编码。然后清除内部状态。
        这应该在所有输入块都已传递给 `get_base64_chunk` 后调用一次。

        返回:
            包含最终上采样音频块的Base64编码字符串，
            如果没有处理任何块或已经调用了flush，则返回None。
        """
        # *** 修正的flush逻辑（恢复到原始） ***
        if self.resampled_previous_chunk is not None:
            # 按照原始逻辑返回整个最后一个上采样块
            pcm = (self.resampled_previous_chunk * 32767).astype(np.int16).tobytes()

            # 刷新后清除状态
            self.previous_chunk = None
            self.resampled_previous_chunk = None
            return base64.b64encode(pcm).decode('utf-8')
        return None # 如果没有要刷新的内容，返回None
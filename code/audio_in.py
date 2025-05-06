import asyncio
import logging
from typing import Optional, Callable
import numpy as np
from scipy.signal import resample_poly
from transcribe import TranscriptionProcessor

logger = logging.getLogger(__name__)


class AudioInputProcessor:
    """
    管理音频输入，处理转录，并处理相关回调。

    该类接收原始音频块，将其重采样为所需格式（16kHz），
    将其提供给底层的 `TranscriptionProcessor`，并管理用于
    实时转录更新、录音开始事件和静音检测的回调。
    它还在后台任务中运行转录过程。
    """

    _RESAMPLE_RATIO = 3  # 从48kHz（假设输入）到16kHz的重采样比率。

    def __init__(
            self,
            language: str = "en",
            is_orpheus: bool = False,
            silence_active_callback: Optional[Callable[[bool], None]] = None,
            pipeline_latency: float = 0.5,
        ) -> None:
        """
        初始化音频输入处理器。

        参数:
            language: 转录的目标语言代码（例如，"en"）。
            is_orpheus: 指示是否应使用特定模型变体的标志。
            silence_active_callback: 当静音状态改变时调用的可选回调函数。
                                     它接收一个布尔参数（如果静音处于活动状态则为True）。
            pipeline_latency: 处理管道的估计延迟（以秒为单位）。
        """
        self.last_partial_text: Optional[str] = None
        self.transcriber = TranscriptionProcessor(
            language,
            on_recording_start_callback=self._on_recording_start,
            silence_active_callback=self._silence_active_callback,
            is_orpheus=is_orpheus,
            pipeline_latency=pipeline_latency,
        )
        # Flag to indicate if the transcription loop has failed fatally
        self._transcription_failed = False
        self.transcription_task = asyncio.create_task(self._run_transcription_loop())


        self.realtime_callback: Optional[Callable[[str], None]] = None
        self.recording_start_callback: Optional[Callable[[None], None]] = None # Type adjusted
        self.silence_active_callback: Optional[Callable[[bool], None]] = silence_active_callback
        self.interrupted = False # TODO: Consider renaming or clarifying usage (interrupted by user speech?)

        self._setup_callbacks()
        logger.info("👂🚀 AudioInputProcessor initialized.")

    def _silence_active_callback(self, is_active: bool) -> None:
        """内部回调中继，用于静音检测状态。"""
        if self.silence_active_callback:
            self.silence_active_callback(is_active)

    def _on_recording_start(self) -> None:
        """当转录器开始录音时触发的内部回调中继。"""
        if self.recording_start_callback:
            self.recording_start_callback()

    def abort_generation(self) -> None:
        """向底层转录器发出信号，中止任何正在进行的生成过程。"""
        logger.info("👂🛑 Aborting generation requested.")
        self.transcriber.abort_generation()

    def _setup_callbacks(self) -> None:
        """为TranscriptionProcessor实例设置内部回调。"""
        def partial_transcript_callback(text: str) -> None:
            """处理来自转录器的部分转录结果。"""
            if text != self.last_partial_text:
                self.last_partial_text = text
                if self.realtime_callback:
                    self.realtime_callback(text)

        self.transcriber.realtime_transcription_callback = partial_transcript_callback

    async def _run_transcription_loop(self) -> None:
        """
        在后台asyncio任务中持续运行转录循环。

        它重复调用底层的`transcribe_loop`。如果`transcribe_loop`
        正常完成（完成一个周期），此循环将再次调用它。
        如果`transcribe_loop`引发异常，它被视为致命错误，
        设置标志，并且此循环终止。单独处理CancelledError。
        """
        task_name = self.transcription_task.get_name() if hasattr(self.transcription_task, 'get_name') else 'TranscriptionTask'
        logger.info(f"👂▶️ Starting background transcription task ({task_name}).")
        while True: # Loop restored to continuously call transcribe_loop
            try:
                # Run one cycle of the underlying blocking loop
                await asyncio.to_thread(self.transcriber.transcribe_loop)
                # If transcribe_loop returns without error, it means one cycle is complete.
                # The `while True` ensures it will be called again.
                logger.debug("👂✅ TranscriptionProcessor.transcribe_loop completed one cycle.")
                # Add a small sleep to prevent potential tight loop if transcribe_loop returns instantly
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                logger.info(f"👂🚫 Transcription loop ({task_name}) cancelled.")
                # Do not set failure flag on cancellation
                break # Exit the while loop
            except Exception as e:
                # An actual error occurred within transcribe_loop
                logger.error(f"👂💥 Transcription loop ({task_name}) encountered a fatal error: {e}. Loop terminated.", exc_info=True)
                self._transcription_failed = True # Set failure flag
                break # Exit the while loop, stopping retries

        logger.info(f"👂⏹️ Background transcription task ({task_name}) finished.")


    def process_audio_chunk(self, raw_bytes: bytes) -> np.ndarray:
        """
        将原始音频字节（int16）转换为16kHz 16位PCM numpy数组。

        音频被转换为float32以进行精确重采样，然后
        转换回int16，裁剪超出有效范围的值。

        参数:
            raw_bytes: 假定为int16格式的原始音频数据。

        返回:
            包含16kHz下int16格式重采样音频的numpy数组。
            如果输入是静音，则返回零数组。
        """
        raw_audio = np.frombuffer(raw_bytes, dtype=np.int16)

        if np.max(np.abs(raw_audio)) == 0:
            # Calculate expected length after resampling for silence
            expected_len = int(np.ceil(len(raw_audio) / self._RESAMPLE_RATIO))
            return np.zeros(expected_len, dtype=np.int16)

        # Convert to float32 for resampling precision
        audio_float32 = raw_audio.astype(np.float32)

        # Resample using float32 data
        resampled_float = resample_poly(audio_float32, 1, self._RESAMPLE_RATIO)

        # Convert back to int16, clipping to ensure validity
        resampled_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)

        return resampled_int16


    async def process_chunk_queue(self, audio_queue: asyncio.Queue) -> None:
        """
        持续处理从asyncio队列接收的音频块。

        检索音频数据，使用`process_audio_chunk`处理它，并
        将结果提供给转录器，除非被中断或转录
        任务失败。当从队列接收到`None`或发生错误时停止。

        参数:
            audio_queue: 预期产生包含'pcm'（原始音频字节）或None以终止的字典的asyncio队列。
        """
        logger.info("👂▶️ Starting audio chunk processing loop.")
        while True:
            try:
                # Check if the transcription task has permanently failed *before* getting item
                if self._transcription_failed:
                    logger.error("👂�� Transcription task failed previously. Stopping audio processing.")
                    break # Stop processing if transcription backend is down

                # Check if the task finished unexpectedly (e.g., cancelled but not failed)
                # Needs to check self.transcription_task existence as it might be None during shutdown
                if self.transcription_task and self.transcription_task.done() and not self._transcription_failed:
                     # Attempt to check exception status if task is done
                    task_exception = self.transcription_task.exception()
                    if task_exception and not isinstance(task_exception, asyncio.CancelledError):
                        # If there was an exception other than CancelledError, treat it as failed.
                        logger.error(f"👂🛑 Transcription task finished with unexpected error: {task_exception}. Stopping audio processing.", exc_info=task_exception)
                        self._transcription_failed = True # Mark as failed
                        break
                    else:
                         # Finished cleanly or was cancelled
                        logger.warning("👂⏹️ Transcription task is no longer running (completed or cancelled). Stopping audio processing.")
                        break # Stop processing

                audio_data = await audio_queue.get()
                if audio_data is None:
                    logger.info("👂🔌 Received termination signal for audio processing.")
                    break  # Termination signal

                pcm_data = audio_data.pop("pcm")

                # Process audio chunk (resampling happens consistently via float32)
                processed = self.process_audio_chunk(pcm_data)
                if processed.size == 0:
                    continue # Skip empty chunks

                # Feed audio only if not interrupted and transcriber should be running
                if not self.interrupted:
                    # Check failure flag again, as it might have been set between queue.get and here
                     if not self._transcription_failed:
                        # Feed audio to the underlying processor
                        self.transcriber.feed_audio(processed.tobytes(), audio_data)
                     # No 'else' needed here because the checks at the start of the loop handle termination

            except asyncio.CancelledError:
                logger.info("👂🚫 Audio processing task cancelled.")
                break
            except Exception as e:
                # Log general errors during audio chunk processing
                logger.error(f"👂💥 Audio processing error in queue loop: {e}", exc_info=True)
                # Continue processing subsequent chunks after logging the error.
                # Consider adding logic to break if errors persist.
        logger.info("👂⏹️ Audio chunk processing loop finished.")


    def shutdown(self) -> None:
        """
        Initiates shutdown procedures for the audio processor and transcriber.

        Signals the transcriber to shut down and cancels the background
        transcription task.
        """
        logger.info("👂🛑 Shutting down AudioInputProcessor...")
        # Ensure transcriber shutdown is called first to signal the loop
        if hasattr(self.transcriber, 'shutdown'):
             logger.info("👂🛑 Signaling TranscriptionProcessor to shut down.")
             self.transcriber.shutdown()
        else:
             logger.warning("👂⚠️ TranscriptionProcessor does not have a shutdown method.")

        if self.transcription_task and not self.transcription_task.done():
            task_name = self.transcription_task.get_name() if hasattr(self.transcription_task, 'get_name') else 'TranscriptionTask'
            logger.info(f"👂🚫 Cancelling background transcription task ({task_name})...")
            self.transcription_task.cancel()
            # Optional: Add await with timeout here in an async shutdown context
            # try:
            #     await asyncio.wait_for(self.transcription_task, timeout=5.0)
            # except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
            #     logger.warning(f"👂⚠️ Error/Timeout waiting for transcription task {task_name} cancellation: {e}")
        else:
            logger.info("👂✅ Transcription task already done or not running during shutdown.")

        logger.info("👂👋 AudioInputProcessor shutdown sequence initiated.")
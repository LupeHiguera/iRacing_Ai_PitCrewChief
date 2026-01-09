"""
Text-to-speech wrapper for Piper TTS.
"""

import asyncio
import io
import tempfile
from typing import Optional

import sounddevice as sd
import soundfile as sf


class PiperTTS:
    """Async TTS using Piper for voice synthesis."""

    def __init__(self, piper_path: str, model_path: str, max_queue_size: int = 10):
        """
        Initialize Piper TTS.

        Args:
            piper_path: Path to piper executable
            model_path: Path to voice model (.onnx file)
            max_queue_size: Maximum number of queued messages
        """
        self._piper_path = piper_path
        self._model_path = model_path
        self._max_queue_size = max_queue_size

        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._is_speaking: bool = False
        self._running: bool = False

    async def start(self) -> None:
        """Start the background speech worker."""
        if self._worker_task is not None:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Stop the background worker and clear queue."""
        self._running = False

        # Clear the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Cancel worker task
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def speak(self, text: str, priority: bool = False) -> None:
        """
        Queue text for speech.

        Args:
            text: Text to speak
            priority: If True, clears queue and speaks immediately
        """
        if priority:
            # Clear existing queue for priority messages
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await self._queue.put(text)
        else:
            # Skip if already speaking (unless priority)
            if not self._is_speaking:
                try:
                    self._queue.put_nowait(text)
                except asyncio.QueueFull:
                    # Queue full, drop the message
                    pass

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking

    def queue_size(self) -> int:
        """Get number of items in speech queue."""
        return self._queue.qsize()

    async def _worker(self) -> None:
        """Background worker that processes the speech queue."""
        while self._running:
            try:
                # Wait for text with timeout to allow checking _running
                try:
                    text = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                await self._process_speech(text)

            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but keep worker running
                pass

    async def _process_speech(self, text: str) -> None:
        """Process a single speech request."""
        self._is_speaking = True
        try:
            audio_data = await self._synthesize(text)
            if audio_data:
                await self._play_audio(audio_data)
        finally:
            self._is_speaking = False

    async def _synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize text to audio using Piper.

        Returns raw WAV audio data.
        """
        try:
            print(f"[TTS] Synthesizing: {text[:50]}...")
            process = await asyncio.create_subprocess_exec(
                self._piper_path,
                "--model", self._model_path,
                "--output_raw",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate(input=text.encode('utf-8'))

            if process.returncode != 0:
                print(f"[TTS] Piper failed with code {process.returncode}: {stderr.decode()}")
                return None

            print(f"[TTS] Synthesized {len(stdout)} bytes of audio")
            return stdout

        except FileNotFoundError:
            print(f"[TTS] ERROR: Piper not found at {self._piper_path}")
            return None
        except Exception as e:
            print(f"[TTS] ERROR: {e}")
            return None

    async def _play_audio(self, audio_data: bytes) -> None:
        """Play raw audio data."""
        try:
            # Piper outputs raw 16-bit PCM at 22050 Hz
            # Write to temp file for soundfile to read
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            # Read as raw PCM
            import numpy as np
            samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = 22050

            # Play audio (blocking in thread to not block event loop)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: sd.play(samples, sample_rate) or sd.wait()
            )

        except Exception:
            # Silently handle playback errors
            pass
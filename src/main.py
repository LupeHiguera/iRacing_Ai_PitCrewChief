"""
Main integration for iRacing AI Race Strategist.

Ties together telemetry, strategy, LLM, TTS, and logging.
"""

import asyncio
import signal
import time
from typing import Optional

from config import Config
from .telemetry import TelemetryReader, TelemetrySnapshot
from .strategy import StrategyCalculator, StrategyState, Urgency
from .llm_client import LMStudioClient
from .tts import PiperTTS
from .logger import SessionLogger
from .overlay_server import OverlayServer
from .event_detector import EventDetector, RaceEvent, EventPriority


class StrategyEngine:
    """Main engine that orchestrates the race strategist."""

    def __init__(self, config: Config):
        self._config = config
        self._running = False

        # Components (initialized in start())
        self._telemetry: Optional[TelemetryReader] = None
        self._strategy: Optional[StrategyCalculator] = None
        self._llm: Optional[LMStudioClient] = None
        self._tts: Optional[PiperTTS] = None
        self._logger: Optional[SessionLogger] = None
        self._overlay: Optional[OverlayServer] = None
        self._event_detector: Optional[EventDetector] = None

        # State tracking
        self._last_lap: int = 0
        self._last_urgency: Urgency = Urgency.OK
        self._last_llm_call_time: float = 0.0

    async def start(self) -> None:
        """Initialize components and start the engine."""
        print("Initializing Race Strategist...")

        # Initialize components
        self._telemetry = TelemetryReader()
        self._strategy = StrategyCalculator(
            fuel_warning_laps=self._config.fuel_warning_laps,
            fuel_critical_laps=self._config.fuel_critical_laps,
            tire_warning_pct=self._config.tire_warning_pct,
            tire_critical_pct=self._config.tire_critical_pct,
        )
        self._llm = LMStudioClient(
            base_url=self._config.lm_studio_url,
            timeout=self._config.llm_timeout_sec,
        )
        self._tts = PiperTTS(
            piper_path=self._config.piper_path,
            model_path=self._config.tts_model_path,
            max_queue_size=self._config.tts_queue_size,
        )

        if self._config.log_sessions:
            self._logger = SessionLogger(log_dir=str(self._config.session_log_dir))

        # Initialize event detector
        self._event_detector = EventDetector(self._config)

        # Start overlay server
        if self._config.overlay_enabled:
            self._overlay = OverlayServer(
                host=self._config.overlay_host,
                port=self._config.overlay_port,
            )
            await self._overlay.start()

        # Start TTS worker
        await self._tts.start()

        print("Components initialized.")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        self._running = False
        print("\nShutting down...")

        # End logging session
        if self._logger:
            path = self._logger.end_session()
            if path:
                print(f"Session saved to: {path}")

        # Stop TTS
        if self._tts:
            await self._tts.stop()

        # Stop overlay server
        if self._overlay:
            await self._overlay.stop()

        # Close LLM client
        if self._llm:
            await self._llm.close()

        # Disconnect from iRacing
        if self._telemetry:
            self._telemetry.disconnect()

        print("Shutdown complete.")

    async def run(self) -> None:
        """Main run loop."""
        await self.start()

        # Wait for iRacing connection
        print("Waiting for iRacing...")
        connected = await self._wait_for_connection()
        if not connected:
            print("Could not connect to iRacing. Is the sim running?")
            await self.stop()
            return

        print("Connected to iRacing!")

        # Get track and car info
        track = self._telemetry.get_track_name()
        car = self._telemetry.get_car_name()
        print(f"Track: {track} | Car: {car}")

        # Start logging session
        if self._logger:
            self._logger.start_session(track, car)

        # Set overlay session info
        if self._overlay:
            self._overlay.set_session_info(track, car, "Race")

        # Announce startup
        await self._speak("Race strategist online. Good luck out there.")

        self._running = True
        try:
            await self._main_loop()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _main_loop(self) -> None:
        """Main polling loop at 1Hz."""
        print("[DEBUG] Main loop started")
        while self._running:
            try:
                # Get telemetry snapshot
                snapshot = self._telemetry.get_snapshot()
                if snapshot is None:
                    # Lost connection
                    print("Lost iRacing connection, attempting reconnect...")
                    if not await self._wait_for_connection(max_attempts=10):
                        print("Could not reconnect to iRacing.")
                        break
                    continue

                # Skip if not on track
                if not snapshot.is_on_track:
                    await asyncio.sleep(1.0)
                    continue

                # Update strategy
                state = self._strategy.update(snapshot)

                # Log telemetry
                self._log_telemetry(snapshot, state)

                # Broadcast to overlay
                if self._overlay:
                    await self._overlay.broadcast_telemetry(snapshot, state)

                # Debug: Show lap changes
                if snapshot.lap != self._last_lap:
                    print(f"[DEBUG] Lap {snapshot.lap} completed | Fuel: {state.laps_of_fuel:.1f} laps | "
                          f"Urgency: {state.urgency.name}")

                # Detect events
                events = self._event_detector.detect_events(snapshot, state)

                # Process highest priority event (if any)
                if events:
                    top_event = events[0]  # Already sorted by priority

                    # Check global cooldown (except for critical events)
                    current_time = time.time()
                    cooldown_ok = (current_time - self._last_llm_call_time) >= self._config.llm_cooldown_sec
                    is_critical = top_event.priority == EventPriority.CRITICAL

                    if cooldown_ok or is_critical:
                        print(f"[DEBUG] Event detected: {top_event.event_type.name} - {top_event.message}")
                        await self._handle_event(snapshot, state, top_event)

                # Update state tracking
                self._last_lap = snapshot.lap
                self._last_urgency = state.urgency

                # Poll at 1Hz
                await asyncio.sleep(1.0)

            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)

    async def _wait_for_connection(
        self,
        max_attempts: int = 60,
        delay: float = 1.0,
    ) -> bool:
        """Wait for iRacing connection."""
        for _ in range(max_attempts):
            if self._telemetry.connect():
                return True
            await asyncio.sleep(delay)
        return False

    # === LEGACY TRIGGER LOGIC ===
    # Kept for reference/fallback. Now replaced by EventDetector.detect_events()
    # Remove once EventDetector is proven stable.

    def _get_trigger_reason(
        self,
        snapshot: TelemetrySnapshot,
        state: StrategyState,
        last_lap: int,
    ) -> Optional[str]:
        """
        [LEGACY] Determine if we should call the LLM and why.

        Returns trigger reason string if should trigger, None otherwise.
        """
        current_time = time.time()
        cooldown_ok = (current_time - self._last_llm_call_time) >= self._config.llm_cooldown_sec

        # Critical always triggers (bypasses cooldown)
        if state.urgency == Urgency.CRITICAL and self._last_urgency != Urgency.CRITICAL:
            reason = state.pit_reason or "Critical situation"
            return f"CRITICAL: {reason}"

        # Check cooldown for non-critical
        if not cooldown_ok:
            return None

        # Urgency escalation
        if self._urgency_escalated(state.urgency):
            reason = state.pit_reason or f"Urgency → {state.urgency.name}"
            return f"Escalation: {reason}"

        # Periodic update on lap completion
        if snapshot.lap > last_lap and snapshot.lap % self._config.periodic_update_laps == 0:
            return f"Lap {snapshot.lap} update (every {self._config.periodic_update_laps} laps)"

        return None

    def _urgency_escalated(self, new_urgency: Urgency) -> bool:
        """Check if urgency escalated from last state."""
        priority = {
            Urgency.OK: 0,
            Urgency.INFO: 1,
            Urgency.WARNING: 2,
            Urgency.CRITICAL: 3,
        }
        return priority[new_urgency] > priority[self._last_urgency]

    async def _handle_event(
        self,
        snapshot: TelemetrySnapshot,
        state: StrategyState,
        event: RaceEvent,
    ) -> None:
        """Handle a detected event - call LLM or use fallback."""
        # Build prompt with event context
        prompt = self._llm.format_telemetry_prompt(state, snapshot, event)

        # Broadcast "thinking" state to overlay
        if self._overlay:
            await self._overlay.broadcast_ai_thinking(prompt)

        # Try LLM
        response = await self._llm.generate(prompt)

        if response:
            message = response.text
            latency = response.latency_ms
            await self._log_llm_call(prompt, message, latency)
        else:
            # Fallback to event's built-in message
            message = event.message
            latency = 0
            await self._log_llm_call(prompt, f"[FALLBACK] {message}", 0)

        # Broadcast to overlay
        if self._overlay:
            await self._overlay.broadcast_ai_message(
                message=message,
                trigger_reason=f"{event.event_type.name}",
                latency_ms=latency,
                urgency=state.urgency,
                prompt=prompt,
            )

        # Speak the message (priority for critical events)
        priority = event.priority == EventPriority.CRITICAL
        await self._speak(message, priority=priority)

        self._last_llm_call_time = time.time()

    async def _handle_llm_call(
        self,
        snapshot: TelemetrySnapshot,
        state: StrategyState,
        trigger_reason: str = "periodic",
    ) -> None:
        """[LEGACY] Handle an LLM call with fallback. Use _handle_event() instead."""
        prompt = self._llm.format_telemetry_prompt(state, snapshot)

        # Broadcast "thinking" state to overlay
        if self._overlay:
            await self._overlay.broadcast_ai_thinking(prompt)

        # Try LLM
        response = await self._llm.generate(prompt)

        if response:
            message = response.text
            latency = response.latency_ms
            await self._log_llm_call(prompt, message, latency)
        else:
            # Fallback to deterministic message
            message = self._get_fallback_message(state, snapshot)
            latency = 0
            await self._log_llm_call(prompt, f"[FALLBACK] {message}", 0)

        # Broadcast to overlay
        if self._overlay:
            await self._overlay.broadcast_ai_message(
                message=message,
                trigger_reason=trigger_reason,
                latency_ms=latency,
                urgency=state.urgency,
                prompt=prompt,
            )

        # Speak the message
        priority = state.urgency == Urgency.CRITICAL
        await self._speak(message, priority=priority)

        self._last_llm_call_time = time.time()

    def _get_fallback_message(self, state: StrategyState, snapshot: TelemetrySnapshot) -> str:
        """Get a deterministic fallback message when LLM fails."""
        position = snapshot.position

        if state.urgency == Urgency.CRITICAL:
            if state.pit_reason and "fuel" in state.pit_reason.lower():
                return f"Box now! Fuel critical, {state.laps_of_fuel:.1f} laps remaining."
            elif state.pit_reason and "tire" in state.pit_reason.lower():
                return f"Box now! Tires critical, {state.worst_tire_corner} at {state.worst_tire_wear:.0f}%."
            else:
                return "Box this lap! Critical situation."

        elif state.urgency == Urgency.WARNING:
            if state.pit_reason and "fuel" in state.pit_reason.lower():
                return f"Fuel getting low. P{position}, {state.laps_of_fuel:.1f} laps remaining."
            elif state.pit_reason and "tire" in state.pit_reason.lower():
                return f"Tires wearing. {state.worst_tire_corner} at {state.worst_tire_wear:.0f}%. P{position}."
            else:
                return f"Warning: Consider pitting soon. P{position}."

        else:
            # Normal lap update with position and fuel
            return f"Lap {snapshot.lap} complete. P{position}, {state.laps_of_fuel:.1f} laps of fuel."

    async def _speak(self, text: str, priority: bool = False) -> None:
        """Send text to TTS."""
        if self._tts:
            await self._tts.speak(text, priority=priority)

    def _log_telemetry(
        self,
        snapshot: TelemetrySnapshot,
        state: StrategyState,
    ) -> None:
        """Log telemetry to session logger."""
        if self._logger:
            self._logger.log_telemetry(snapshot, state)

    async def _log_llm_call(
        self,
        prompt: str,
        response: str,
        latency_ms: float,
    ) -> None:
        """Log LLM call to session logger."""
        if self._logger:
            self._logger.log_llm_call(prompt, response, latency_ms)


async def main():
    """Entry point."""
    config = Config()
    engine = StrategyEngine(config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        engine._running = False

    try:
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass

    try:
        await engine.run()
    except KeyboardInterrupt:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
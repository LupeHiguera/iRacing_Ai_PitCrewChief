"""
WebSocket overlay server for real-time telemetry display.

Provides a comprehensive overlay for video demos showing:
- Race context (track, car, session info)
- Performance metrics (lap times, delta, incidents)
- Telemetry data (fuel, tires, temps)
- AI debug info (model, prompt, latency, thinking state)
- Visual enhancements (gauges, wear indicators)
"""

import asyncio
import json
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .telemetry import TelemetrySnapshot
from .strategy import StrategyState, Urgency


class OverlayServer:
    """WebSocket server for overlay updates."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self._host = host
        self._port = port
        self._app = FastAPI()
        self._connections: Set[WebSocket] = set()
        self._server = None
        self._server_task: Optional[asyncio.Task] = None

        # Session info (set once at start)
        self._track_name: str = "Unknown Track"
        self._car_name: str = "Unknown Car"
        self._session_type: str = "Practice"

        # Setup routes
        self._setup_routes()

    def set_session_info(self, track: str, car: str, session_type: str = "Race") -> None:
        """Set session info for display."""
        self._track_name = track
        self._car_name = car
        self._session_type = session_type

    def _setup_routes(self) -> None:
        """Configure FastAPI routes."""

        @self._app.get("/")
        async def serve_overlay():
            return FileResponse("static/overlay.html")

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._connections.add(websocket)
            try:
                while True:
                    # Keep connection alive, ignore incoming messages
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self._connections.discard(websocket)

    async def start(self) -> None:
        """Start the overlay server."""
        import uvicorn

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        print(f"Overlay server started at http://{self._host}:{self._port}")

    async def stop(self) -> None:
        """Stop the overlay server."""
        if self._server:
            self._server.should_exit = True
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        # Close all connections
        for ws in list(self._connections):
            try:
                await ws.close()
            except Exception:
                pass
        self._connections.clear()

    async def broadcast_telemetry(
        self,
        snapshot: TelemetrySnapshot,
        state: StrategyState,
    ) -> None:
        """Broadcast telemetry update to all connected clients."""
        if not self._connections:
            return

        # Calculate average tire temp for each corner
        tire_temps = {
            "LF": (snapshot.tire_temp_lf_l + snapshot.tire_temp_lf_m + snapshot.tire_temp_lf_r) / 3,
            "RF": (snapshot.tire_temp_rf_l + snapshot.tire_temp_rf_m + snapshot.tire_temp_rf_r) / 3,
            "LR": (snapshot.tire_temp_lr_l + snapshot.tire_temp_lr_m + snapshot.tire_temp_lr_r) / 3,
            "RR": (snapshot.tire_temp_rr_l + snapshot.tire_temp_rr_m + snapshot.tire_temp_rr_r) / 3,
        }

        # Format session time remaining
        time_remain = snapshot.session_time_remain
        if time_remain > 0:
            mins = int(time_remain // 60)
            secs = int(time_remain % 60)
            time_remain_str = f"{mins}:{secs:02d}"
        else:
            time_remain_str = "--:--"

        data = {
            "type": "telemetry",
            "data": {
                # Race Context
                "track_name": self._track_name,
                "car_name": self._car_name,
                "session_type": self._session_type,
                "session_time_remain": time_remain_str,
                "session_laps_remain": snapshot.session_laps_remain if snapshot.session_laps_remain > 0 else None,

                # Position & Lap
                "lap": snapshot.lap,
                "position": snapshot.position,
                "lap_pct": round(snapshot.lap_pct * 100, 1),

                # Performance
                "last_lap_time": snapshot.last_lap_time if snapshot.last_lap_time > 0 else None,
                "best_lap_time": snapshot.best_lap_time if snapshot.best_lap_time > 0 else None,
                "lap_delta": round(snapshot.lap_delta_to_best, 3) if snapshot.lap_delta_to_best else 0,
                "incident_count": snapshot.incident_count,

                # Fuel
                "fuel_level": round(snapshot.fuel_level, 2),
                "fuel_laps": round(state.laps_of_fuel, 1) if state.laps_of_fuel != float('inf') else 99.9,
                "fuel_per_lap": round(state.fuel_per_lap, 3),
                "fuel_pct": round(snapshot.fuel_level_pct * 100, 1),

                # Tires
                "tire_wear": {
                    "LF": round(snapshot.tire_wear_lf, 1),
                    "RF": round(snapshot.tire_wear_rf, 1),
                    "LR": round(snapshot.tire_wear_lr, 1),
                    "RR": round(snapshot.tire_wear_rr, 1),
                },
                "tire_temps": {
                    "LF": round(tire_temps["LF"], 1),
                    "RF": round(tire_temps["RF"], 1),
                    "LR": round(tire_temps["LR"], 1),
                    "RR": round(tire_temps["RR"], 1),
                },
                "tire_pressures": {
                    "LF": round(snapshot.tire_pressure_lf, 1),
                    "RF": round(snapshot.tire_pressure_rf, 1),
                    "LR": round(snapshot.tire_pressure_lr, 1),
                    "RR": round(snapshot.tire_pressure_rr, 1),
                },
                "worst_tire": state.worst_tire_corner,
                "worst_tire_wear": round(state.worst_tire_wear, 1),

                # Gaps
                "gap_ahead": round(snapshot.gap_ahead_sec, 2) if snapshot.gap_ahead_sec else None,
                "gap_behind": round(snapshot.gap_behind_sec, 2) if snapshot.gap_behind_sec else None,

                # Track Conditions
                "track_temp": round(snapshot.track_temp_c, 1),
                "air_temp": round(snapshot.air_temp_c, 1),

                # Strategy
                "urgency": state.urgency.value,
                "needs_pit": state.needs_pit,
                "pit_reason": state.pit_reason,
                "pit_window": state.pit_window,
            }
        }

        await self._broadcast(data)

    async def broadcast_ai_thinking(self, prompt_preview: str) -> None:
        """Broadcast that AI is currently thinking."""
        if not self._connections:
            return

        # Truncate prompt for display
        if len(prompt_preview) > 150:
            prompt_preview = prompt_preview[:147] + "..."

        data = {
            "type": "ai_thinking",
            "data": {
                "status": "thinking",
                "prompt_preview": prompt_preview,
                "model": "Llama 3.1 8B",
            }
        }

        await self._broadcast(data)

    async def broadcast_ai_message(
        self,
        message: str,
        trigger_reason: str,
        latency_ms: float,
        urgency: Urgency,
        prompt: str = "",
    ) -> None:
        """Broadcast AI message to all connected clients."""
        if not self._connections:
            return

        # Truncate prompt for display
        prompt_preview = prompt
        if len(prompt_preview) > 200:
            prompt_preview = prompt_preview[:197] + "..."

        data = {
            "type": "ai_message",
            "data": {
                "message": message,
                "trigger_reason": trigger_reason,
                "latency_ms": round(latency_ms, 0),
                "urgency": urgency.value,
                "model": "Llama 3.1 8Bvery ",
                "prompt_preview": prompt_preview,
                "status": "complete",
            }
        }

        await self._broadcast(data)

    async def _broadcast(self, data: dict) -> None:
        """Send data to all connected WebSocket clients."""
        if not self._connections:
            return

        message = json.dumps(data)
        dead_connections = set()

        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead_connections.add(ws)

        # Clean up dead connections
        self._connections -= dead_connections

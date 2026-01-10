"""
Event detection system for real-time race engineer callouts.

Monitors telemetry for interesting racing events and triggers appropriate callouts.
Handles state tracking, cooldowns, and event prioritization.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List

from config import Config
from .telemetry import TelemetrySnapshot
from .strategy import StrategyState, Urgency


class EventType(Enum):
    """Types of events that can trigger callouts."""
    # Position
    POSITION_GAINED = auto()
    POSITION_LOST = auto()

    # Gap/Battle
    GAP_CLOSING = auto()
    GAP_DEFEND = auto()
    GAP_SAFE = auto()
    DIRTY_AIR = auto()
    CLEAN_AIR = auto()

    # Tire temps
    TIRE_COLD = auto()
    TIRE_OPTIMAL = auto()
    TIRE_HOT = auto()
    LOCKUP = auto()
    WHEELSPIN = auto()

    # Pace
    PACE_DROPPING = auto()
    PACE_IMPROVING = auto()
    PERSONAL_BEST = auto()

    # Race progress
    RACE_HALFWAY = auto()
    LAPS_REMAINING = auto()
    FINAL_LAP = auto()

    # Flags
    YELLOW_FLAG = auto()
    GREEN_FLAG = auto()

    # Incidents
    INCIDENT = auto()

    # Pit
    PIT_ENTRY = auto()
    PIT_EXIT = auto()

    # Strategy (from existing system)
    FUEL_WARNING = auto()
    FUEL_CRITICAL = auto()
    TIRE_WEAR_WARNING = auto()
    TIRE_WEAR_CRITICAL = auto()
    PERIODIC_UPDATE = auto()


class EventPriority(Enum):
    """Priority levels for events."""
    CRITICAL = 4  # Box now, defend
    HIGH = 3      # Position change, battle
    MEDIUM = 2    # Tire temps, pace
    LOW = 1       # Info, periodic


@dataclass
class RaceEvent:
    """A detected race event."""
    event_type: EventType
    priority: EventPriority
    message: str  # Short description for fallback
    data: dict = field(default_factory=dict)  # Additional context
    timestamp: float = field(default_factory=time.time)


class EventDetector:
    """
    Stateful event detector that monitors telemetry for interesting situations.

    Tracks previous values to detect changes, manages per-event-type cooldowns,
    and batches rapid changes (like position shuffling on lap 1).
    """

    # iRacing session flag bitmasks
    FLAG_YELLOW = 0x8
    FLAG_CAUTION = 0x4000
    FLAG_CAUTION_WAVING = 0x8000

    def __init__(self, config: Config):
        self._config = config

        # Previous state tracking
        self._last_position: Optional[int] = None
        self._last_gap_behind: Optional[float] = None
        self._last_gap_ahead: Optional[float] = None
        self._last_on_pit_road: bool = False
        self._last_incident_count: int = 0
        self._last_session_flags: int = 0
        self._last_best_lap_time: float = 0.0
        self._last_lap: int = 0

        # Tire temp tracking (for spike detection)
        self._last_tire_temps: dict = {}  # corner -> avg temp

        # Position batching
        self._position_change_start_time: Optional[float] = None
        self._position_at_change_start: Optional[int] = None
        self._pending_position_event: Optional[RaceEvent] = None

        # Tire temp state
        self._announced_tire_optimal: bool = False
        self._tire_temp_state: str = "unknown"  # cold, optimal, hot

        # Gap state tracking
        self._was_in_battle: bool = False  # Was gap < gap_close_threshold
        self._was_in_dirty_air: bool = False

        # Race progress tracking
        self._announced_halfway: bool = False
        self._announced_laps_remaining: set = set()
        self._total_laps: Optional[int] = None

        # Pace tracking
        self._lap_times: deque = deque(maxlen=10)  # Recent lap times
        self._pace_baseline: Optional[float] = None  # Average pace before trend

        # Cooldowns (event_type -> last_trigger_time)
        self._cooldowns: dict = {}

    def reset(self) -> None:
        """Reset all state (e.g., for new session)."""
        self._last_position = None
        self._last_gap_behind = None
        self._last_gap_ahead = None
        self._last_on_pit_road = False
        self._last_incident_count = 0
        self._last_session_flags = 0
        self._last_best_lap_time = 0.0
        self._last_lap = 0
        self._last_tire_temps = {}
        self._position_change_start_time = None
        self._position_at_change_start = None
        self._pending_position_event = None
        self._announced_tire_optimal = False
        self._tire_temp_state = "unknown"
        self._was_in_battle = False
        self._was_in_dirty_air = False
        self._announced_halfway = False
        self._announced_laps_remaining = set()
        self._total_laps = None
        self._lap_times.clear()
        self._pace_baseline = None
        self._cooldowns = {}

    def detect_events(
        self,
        snapshot: TelemetrySnapshot,
        state: StrategyState,
    ) -> List[RaceEvent]:
        """
        Analyze telemetry and return list of detected events.

        Events are returned in priority order (highest first).
        """
        events = []

        # Detect various event types
        events.extend(self._detect_position_events(snapshot))
        events.extend(self._detect_gap_events(snapshot))
        events.extend(self._detect_tire_temp_events(snapshot))
        events.extend(self._detect_pace_events(snapshot))
        events.extend(self._detect_race_progress_events(snapshot))
        events.extend(self._detect_flag_events(snapshot))
        events.extend(self._detect_incident_events(snapshot))
        events.extend(self._detect_pit_events(snapshot))
        events.extend(self._detect_strategy_events(state, snapshot))

        # Update state for next iteration
        self._update_state(snapshot)

        # Sort by priority (highest first)
        events.sort(key=lambda e: e.priority.value, reverse=True)

        return events

    def _is_on_cooldown(self, event_type: EventType, cooldown_sec: float) -> bool:
        """Check if an event type is on cooldown."""
        last_time = self._cooldowns.get(event_type, 0)
        return (time.time() - last_time) < cooldown_sec

    def _set_cooldown(self, event_type: EventType) -> None:
        """Mark an event type as triggered (starts cooldown)."""
        self._cooldowns[event_type] = time.time()

    # === Position Events ===

    def _detect_position_events(self, snapshot: TelemetrySnapshot) -> List[RaceEvent]:
        """Detect position changes with batching to avoid lap 1 spam."""
        if not self._config.position_change_enabled:
            return []

        if snapshot.lap < self._config.position_min_lap:
            # Too early, skip
            self._last_position = snapshot.position
            return []

        if self._last_position is None:
            self._last_position = snapshot.position
            return []

        events = []
        current_time = time.time()
        position_changed = snapshot.position != self._last_position

        if position_changed:
            # Start or continue tracking position change
            if self._position_change_start_time is None:
                self._position_change_start_time = current_time
                self._position_at_change_start = self._last_position

            # Update pending event with latest position
            delta = self._position_at_change_start - snapshot.position
            if delta > 0:
                self._pending_position_event = RaceEvent(
                    event_type=EventType.POSITION_GAINED,
                    priority=EventPriority.HIGH,
                    message=f"Gained {delta} position{'s' if delta > 1 else ''}, now P{snapshot.position}",
                    data={"positions_gained": delta, "new_position": snapshot.position}
                )
            elif delta < 0:
                self._pending_position_event = RaceEvent(
                    event_type=EventType.POSITION_LOST,
                    priority=EventPriority.HIGH,
                    message=f"Lost {abs(delta)} position{'s' if abs(delta) > 1 else ''}, now P{snapshot.position}",
                    data={"positions_lost": abs(delta), "new_position": snapshot.position}
                )

        # Check if we should emit the batched position event
        if self._position_change_start_time is not None:
            time_since_change = current_time - self._position_change_start_time

            # Position settled (no change this tick) and settle time passed
            if not position_changed and time_since_change >= self._config.position_settle_time_sec:
                if self._pending_position_event is not None:
                    if not self._is_on_cooldown(self._pending_position_event.event_type,
                                                 self._config.event_cooldown_position):
                        events.append(self._pending_position_event)
                        self._set_cooldown(self._pending_position_event.event_type)

                # Reset batching state
                self._position_change_start_time = None
                self._position_at_change_start = None
                self._pending_position_event = None

        return events

    # === Gap/Battle Events ===

    def _detect_gap_events(self, snapshot: TelemetrySnapshot) -> List[RaceEvent]:
        """Detect gap-based events (battle, dirty air, etc)."""
        events = []

        # Gap behind (car catching us)
        if snapshot.gap_behind_sec is not None:
            gap = snapshot.gap_behind_sec

            # Defend alert (very close)
            if gap < self._config.gap_battle_threshold_sec:
                if not self._is_on_cooldown(EventType.GAP_DEFEND, self._config.event_cooldown_gap):
                    events.append(RaceEvent(
                        event_type=EventType.GAP_DEFEND,
                        priority=EventPriority.CRITICAL,
                        message=f"Defend! {gap:.1f} behind",
                        data={"gap_behind": gap}
                    ))
                    self._set_cooldown(EventType.GAP_DEFEND)
                    self._was_in_battle = True

            # Closing alert
            elif gap < self._config.gap_close_threshold_sec:
                if not self._was_in_battle:
                    if not self._is_on_cooldown(EventType.GAP_CLOSING, self._config.event_cooldown_gap):
                        events.append(RaceEvent(
                            event_type=EventType.GAP_CLOSING,
                            priority=EventPriority.HIGH,
                            message=f"Car behind closing, {gap:.1f} seconds",
                            data={"gap_behind": gap}
                        ))
                        self._set_cooldown(EventType.GAP_CLOSING)
                self._was_in_battle = True

            # Safe/clear (gap opened up after being in battle)
            elif gap > self._config.gap_safe_threshold_sec and self._was_in_battle:
                if not self._is_on_cooldown(EventType.GAP_SAFE, self._config.event_cooldown_gap):
                    events.append(RaceEvent(
                        event_type=EventType.GAP_SAFE,
                        priority=EventPriority.MEDIUM,
                        message=f"Gap stable, {gap:.1f} seconds clear",
                        data={"gap_behind": gap}
                    ))
                    self._set_cooldown(EventType.GAP_SAFE)
                self._was_in_battle = False

        # Gap ahead (dirty air / clean air)
        if snapshot.gap_ahead_sec is not None:
            gap = snapshot.gap_ahead_sec

            # In dirty air
            if gap < self._config.gap_ahead_dirty_air_sec:
                if not self._was_in_dirty_air:
                    if not self._is_on_cooldown(EventType.DIRTY_AIR, self._config.event_cooldown_gap):
                        events.append(RaceEvent(
                            event_type=EventType.DIRTY_AIR,
                            priority=EventPriority.MEDIUM,
                            message=f"In dirty air, manage the temps",
                            data={"gap_ahead": gap}
                        ))
                        self._set_cooldown(EventType.DIRTY_AIR)
                self._was_in_dirty_air = True

            # Clean air (after being in dirty air)
            elif gap > self._config.gap_ahead_clean_air_sec and self._was_in_dirty_air:
                if not self._is_on_cooldown(EventType.CLEAN_AIR, self._config.event_cooldown_gap):
                    events.append(RaceEvent(
                        event_type=EventType.CLEAN_AIR,
                        priority=EventPriority.MEDIUM,
                        message=f"Clean air, push now",
                        data={"gap_ahead": gap}
                    ))
                    self._set_cooldown(EventType.CLEAN_AIR)
                self._was_in_dirty_air = False

        return events

    # === Tire Temperature Events ===

    def _detect_tire_temp_events(self, snapshot: TelemetrySnapshot) -> List[RaceEvent]:
        """Detect tire temperature events (hot, cold, optimal, lockup, wheelspin)."""
        events = []

        # Calculate average temps per corner
        temps = {
            "LF": (snapshot.tire_temp_lf_l + snapshot.tire_temp_lf_m + snapshot.tire_temp_lf_r) / 3,
            "RF": (snapshot.tire_temp_rf_l + snapshot.tire_temp_rf_m + snapshot.tire_temp_rf_r) / 3,
            "LR": (snapshot.tire_temp_lr_l + snapshot.tire_temp_lr_m + snapshot.tire_temp_lr_r) / 3,
            "RR": (snapshot.tire_temp_rr_l + snapshot.tire_temp_rr_m + snapshot.tire_temp_rr_r) / 3,
        }

        # Skip if temps are all zero (car doesn't report temps)
        if all(t == 0 for t in temps.values()):
            return []

        # Lockup/wheelspin detection (sudden temp spike)
        if self._last_tire_temps:
            fronts = ["LF", "RF"]
            rears = ["LR", "RR"]

            for corner in fronts:
                if corner in self._last_tire_temps:
                    delta = temps[corner] - self._last_tire_temps[corner]
                    if delta > self._config.tire_temp_spike_delta_c:
                        if not self._is_on_cooldown(EventType.LOCKUP, 5.0):  # Short cooldown
                            events.append(RaceEvent(
                                event_type=EventType.LOCKUP,
                                priority=EventPriority.HIGH,
                                message=f"Lockup on {corner}! Easy on the brakes",
                                data={"corner": corner, "temp_spike": delta}
                            ))
                            self._set_cooldown(EventType.LOCKUP)
                            break  # Only one lockup event per tick

            for corner in rears:
                if corner in self._last_tire_temps:
                    delta = temps[corner] - self._last_tire_temps[corner]
                    if delta > self._config.tire_temp_spike_delta_c:
                        if not self._is_on_cooldown(EventType.WHEELSPIN, 5.0):
                            events.append(RaceEvent(
                                event_type=EventType.WHEELSPIN,
                                priority=EventPriority.HIGH,
                                message=f"Wheelspin! Smooth on throttle",
                                data={"corner": corner, "temp_spike": delta}
                            ))
                            self._set_cooldown(EventType.WHEELSPIN)
                            break

        # Overall temp state
        avg_temp = sum(temps.values()) / 4
        front_avg = (temps["LF"] + temps["RF"]) / 2
        rear_avg = (temps["LR"] + temps["RR"]) / 2

        # Determine new temp state
        new_state = "optimal"
        hot_corner = None
        cold_corner = None

        for corner, temp in temps.items():
            if temp > self._config.tire_temp_hot_c:
                new_state = "hot"
                hot_corner = corner
                break
            elif temp < self._config.tire_temp_cold_c:
                new_state = "cold"
                cold_corner = corner

        # State transitions
        if new_state != self._tire_temp_state:
            if not self._is_on_cooldown(EventType.TIRE_HOT, self._config.event_cooldown_tire_temp):
                if new_state == "hot":
                    # Determine if fronts or rears
                    if hot_corner in ["LF", "RF"]:
                        msg = "Fronts running hot, ease the braking"
                    else:
                        msg = "Rears overheating, smooth on throttle"
                    events.append(RaceEvent(
                        event_type=EventType.TIRE_HOT,
                        priority=EventPriority.MEDIUM,
                        message=msg,
                        data={"corner": hot_corner, "temp": temps[hot_corner]}
                    ))
                    self._set_cooldown(EventType.TIRE_HOT)

                elif new_state == "cold":
                    if cold_corner in ["LF", "RF"]:
                        msg = "Fronts still cold, be careful"
                    else:
                        msg = "Rears still cold, watch the throttle"
                    events.append(RaceEvent(
                        event_type=EventType.TIRE_COLD,
                        priority=EventPriority.MEDIUM,
                        message=msg,
                        data={"corner": cold_corner, "temp": temps[cold_corner]}
                    ))
                    self._set_cooldown(EventType.TIRE_COLD)

                elif new_state == "optimal" and not self._announced_tire_optimal:
                    events.append(RaceEvent(
                        event_type=EventType.TIRE_OPTIMAL,
                        priority=EventPriority.LOW,
                        message="Tires in the window",
                        data={"avg_temp": avg_temp}
                    ))
                    self._announced_tire_optimal = True

            self._tire_temp_state = new_state

        # Store for next iteration
        self._last_tire_temps = temps.copy()

        return events

    # === Pace Events ===

    def _detect_pace_events(self, snapshot: TelemetrySnapshot) -> List[RaceEvent]:
        """Detect pace trend events and personal bests."""
        events = []

        # Personal best detection
        if self._config.personal_best_callout:
            if (self._last_best_lap_time > 0 and
                snapshot.best_lap_time > 0 and
                snapshot.best_lap_time < self._last_best_lap_time):
                if not self._is_on_cooldown(EventType.PERSONAL_BEST, 30.0):
                    events.append(RaceEvent(
                        event_type=EventType.PERSONAL_BEST,
                        priority=EventPriority.MEDIUM,
                        message=f"Personal best! {snapshot.best_lap_time:.3f}",
                        data={"lap_time": snapshot.best_lap_time}
                    ))
                    self._set_cooldown(EventType.PERSONAL_BEST)

        # Track lap times for pace trend
        if snapshot.lap > self._last_lap and snapshot.last_lap_time > 0:
            self._lap_times.append(snapshot.last_lap_time)

            # Need enough laps for trend analysis
            if len(self._lap_times) >= self._config.pace_trend_laps * 2:
                recent = list(self._lap_times)[-self._config.pace_trend_laps:]
                earlier = list(self._lap_times)[-(self._config.pace_trend_laps * 2):-self._config.pace_trend_laps]

                recent_avg = sum(recent) / len(recent)
                earlier_avg = sum(earlier) / len(earlier)
                delta = recent_avg - earlier_avg

                if delta > self._config.pace_drop_threshold_sec:
                    if not self._is_on_cooldown(EventType.PACE_DROPPING, self._config.event_cooldown_pace):
                        events.append(RaceEvent(
                            event_type=EventType.PACE_DROPPING,
                            priority=EventPriority.MEDIUM,
                            message=f"Pace dropping, {delta:.1f}s slower",
                            data={"delta": delta, "recent_avg": recent_avg}
                        ))
                        self._set_cooldown(EventType.PACE_DROPPING)

                elif delta < -self._config.pace_gain_threshold_sec:
                    if not self._is_on_cooldown(EventType.PACE_IMPROVING, self._config.event_cooldown_pace):
                        events.append(RaceEvent(
                            event_type=EventType.PACE_IMPROVING,
                            priority=EventPriority.LOW,
                            message=f"Found some pace, {abs(delta):.1f}s faster",
                            data={"delta": delta, "recent_avg": recent_avg}
                        ))
                        self._set_cooldown(EventType.PACE_IMPROVING)

        return events

    # === Race Progress Events ===

    def _detect_race_progress_events(self, snapshot: TelemetrySnapshot) -> List[RaceEvent]:
        """Detect race progress milestones (halfway, laps remaining, final lap)."""
        if not self._config.race_progress_enabled:
            return []

        events = []
        laps_remaining = snapshot.session_laps_remain

        # Track total laps (first valid reading)
        if self._total_laps is None and laps_remaining > 0 and snapshot.lap > 0:
            self._total_laps = laps_remaining + snapshot.lap

        # Halfway point
        if (self._config.race_halfway_callout and
            self._total_laps is not None and
            not self._announced_halfway):
            halfway = self._total_laps // 2
            if snapshot.lap >= halfway:
                events.append(RaceEvent(
                    event_type=EventType.RACE_HALFWAY,
                    priority=EventPriority.LOW,
                    message=f"Halfway, {laps_remaining} laps to go",
                    data={"lap": snapshot.lap, "remaining": laps_remaining}
                ))
                self._announced_halfway = True

        # Laps remaining callouts
        if laps_remaining > 0 and laps_remaining in self._config.race_laps_remaining_callouts:
            if laps_remaining not in self._announced_laps_remaining:
                if laps_remaining == 1:
                    events.append(RaceEvent(
                        event_type=EventType.FINAL_LAP,
                        priority=EventPriority.HIGH,
                        message="Final lap! Bring it home",
                        data={"remaining": 1}
                    ))
                else:
                    events.append(RaceEvent(
                        event_type=EventType.LAPS_REMAINING,
                        priority=EventPriority.MEDIUM,
                        message=f"{laps_remaining} laps remaining",
                        data={"remaining": laps_remaining}
                    ))
                self._announced_laps_remaining.add(laps_remaining)

        return events

    # === Flag Events ===

    def _detect_flag_events(self, snapshot: TelemetrySnapshot) -> List[RaceEvent]:
        """Detect flag changes (yellow, green)."""
        if not self._config.flag_callout_enabled:
            return []

        events = []
        flags = snapshot.session_flags

        is_yellow = bool(flags & (self.FLAG_YELLOW | self.FLAG_CAUTION | self.FLAG_CAUTION_WAVING))
        was_yellow = bool(self._last_session_flags & (self.FLAG_YELLOW | self.FLAG_CAUTION | self.FLAG_CAUTION_WAVING))

        if is_yellow and not was_yellow:
            events.append(RaceEvent(
                event_type=EventType.YELLOW_FLAG,
                priority=EventPriority.CRITICAL,
                message="Yellow flag, caution",
                data={"flags": flags}
            ))
        elif not is_yellow and was_yellow:
            events.append(RaceEvent(
                event_type=EventType.GREEN_FLAG,
                priority=EventPriority.HIGH,
                message="Green flag, go!",
                data={"flags": flags}
            ))

        return events

    # === Incident Events ===

    def _detect_incident_events(self, snapshot: TelemetrySnapshot) -> List[RaceEvent]:
        """Detect when incident count increases."""
        if not self._config.incident_callout_enabled:
            return []

        events = []

        if snapshot.incident_count > self._last_incident_count:
            new_incidents = snapshot.incident_count - self._last_incident_count
            events.append(RaceEvent(
                event_type=EventType.INCIDENT,
                priority=EventPriority.MEDIUM,
                message=f"Incident, that's {snapshot.incident_count}x total",
                data={"new_incidents": new_incidents, "total": snapshot.incident_count}
            ))

        return events

    # === Pit Events ===

    def _detect_pit_events(self, snapshot: TelemetrySnapshot) -> List[RaceEvent]:
        """Detect pit entry/exit."""
        events = []

        # Pit entry
        if self._config.pit_entry_callout:
            if snapshot.on_pit_road and not self._last_on_pit_road:
                events.append(RaceEvent(
                    event_type=EventType.PIT_ENTRY,
                    priority=EventPriority.MEDIUM,
                    message="Pit entry, good stop",
                    data={}
                ))

        # Pit exit
        if self._config.pit_exit_callout:
            if not snapshot.on_pit_road and self._last_on_pit_road:
                events.append(RaceEvent(
                    event_type=EventType.PIT_EXIT,
                    priority=EventPriority.MEDIUM,
                    message="Out of the pits, push now",
                    data={}
                ))

        return events

    # === Strategy Events (integrates with existing StrategyState) ===

    def _detect_strategy_events(
        self,
        state: StrategyState,
        snapshot: TelemetrySnapshot,
    ) -> List[RaceEvent]:
        """Convert existing strategy urgency into events."""
        events = []

        # Periodic update on lap completion
        if snapshot.lap > self._last_lap:
            if snapshot.lap % self._config.periodic_update_laps == 0:
                events.append(RaceEvent(
                    event_type=EventType.PERIODIC_UPDATE,
                    priority=EventPriority.LOW,
                    message=f"Lap {snapshot.lap}, P{snapshot.position}, {state.laps_of_fuel:.1f} laps of fuel",
                    data={"lap": snapshot.lap, "position": snapshot.position}
                ))

        # Fuel/tire warnings are handled by existing strategy system
        # We just wrap them as events for consistency
        if state.urgency == Urgency.CRITICAL:
            if state.pit_reason and "fuel" in state.pit_reason.lower():
                events.append(RaceEvent(
                    event_type=EventType.FUEL_CRITICAL,
                    priority=EventPriority.CRITICAL,
                    message=f"Box now! {state.laps_of_fuel:.1f} laps of fuel",
                    data={"laps_of_fuel": state.laps_of_fuel}
                ))
            elif state.pit_reason and "tire" in state.pit_reason.lower():
                events.append(RaceEvent(
                    event_type=EventType.TIRE_WEAR_CRITICAL,
                    priority=EventPriority.CRITICAL,
                    message=f"Tires critical, {state.worst_tire_corner} at {state.worst_tire_wear:.0f}%",
                    data={"corner": state.worst_tire_corner, "wear": state.worst_tire_wear}
                ))

        return events

    # === State Update ===

    def _update_state(self, snapshot: TelemetrySnapshot) -> None:
        """Update tracked state for next iteration."""
        self._last_position = snapshot.position
        self._last_gap_behind = snapshot.gap_behind_sec
        self._last_gap_ahead = snapshot.gap_ahead_sec
        self._last_on_pit_road = snapshot.on_pit_road
        self._last_incident_count = snapshot.incident_count
        self._last_session_flags = snapshot.session_flags
        self._last_best_lap_time = snapshot.best_lap_time
        self._last_lap = snapshot.lap

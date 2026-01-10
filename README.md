# iRacing AI Pit Crew Chief

A real-time AI race engineer that provides voice strategy updates during iRacing sessions. Uses local LLM inference for low-latency responses and Piper TTS for natural speech output.

## Features

- **Real-time telemetry** - Reads fuel, tire wear, lap times, and position from iRacing
- **Smart strategy calculations** - Tracks fuel consumption, calculates pit windows, monitors tire degradation
- **LLM-powered callouts** - Natural language updates via local Mistral 7B inference
- **Voice output** - Text-to-speech using Piper for hands-free racing
- **Session logging** - Captures telemetry and LLM responses for analysis

## Architecture

```
iRacing → Telemetry Reader → Strategy Calculator → LLM Client → TTS → Audio
                                    ↓
                            Session Logger → Fine-tuning Data
```

## Requirements

- Windows (iRacing is Windows-only)
- Python 3.10+
- [iRacing](https://www.iracing.com/) subscription
- [LM Studio](https://lmstudio.ai/) with Mistral 7B Instruct
- [Piper TTS](https://github.com/rhasspy/piper/releases)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/iRacing_Ai_PitCrewChief.git
cd iRacing_Ai_PitCrewChief

# Install dependencies
pip install -r requirements.txt

# Download Piper and a voice model
# Place piper.exe and en_US-lessac-medium.onnx in C:/tools/piper/
```

## Configuration

Edit `config.py` to set paths and thresholds:

```python
@dataclass
class Config:
    piper_path: str = "C:/tools/piper/piper.exe"
    tts_model_path: str = "C:/tools/piper/en_US-lessac-medium.onnx"
    lm_studio_url: str = "http://localhost:1234/v1"

    fuel_warning_laps: float = 5.0
    fuel_critical_laps: float = 2.0
    tire_warning_pct: float = 70.0
    tire_critical_pct: float = 85.0
```

## Usage

1. Start iRacing and load into a session
2. Start LM Studio with Mistral 7B Instruct model
3. Run the strategist:

```bash
python -m src.main
```

The AI will provide voice updates on:
- Fuel status and pit windows
- Tire wear warnings
- Position changes
- Periodic lap updates

## Project Structure

```
src/
├── telemetry.py    # iRacing data via pyirsdk (43 fields)
├── strategy.py     # Fuel/tire calculations
├── llm_client.py   # LM Studio API client
├── tts.py          # Piper TTS wrapper
├── logger.py       # Session logging to gzip JSON
└── main.py         # Main integration

data/
├── sessions/       # Logged race sessions (gzip JSON)
└── synthetic/      # Generated training data (for fine-tuning)

tests/
└── test_*.py       # 30+ unit tests
```

## Data Collection

Sessions are automatically logged to `data/sessions/` as gzip-compressed JSON files.

### Telemetry Fields (43 total)
| Category | Fields |
|----------|--------|
| **Lap/Position** | lap, lap_pct, position, lap_delta_to_best |
| **Fuel** | fuel_level, fuel_level_pct, fuel_use_per_hour |
| **Tire Wear** | tire_wear_lf/rf/lr/rr (% worn) |
| **Tire Temps** | tire_temp_*_l/m/r (12 fields) |
| **Tire Pressure** | tire_pressure_lf/rf/lr/rr |
| **Brake Pressure** | brake_press_lf/rf/lr/rr |
| **Gaps** | gap_ahead_sec, gap_behind_sec |
| **Session** | session_time_remain, session_laps_remain, session_flags |
| **Track/Weather** | track_temp_c, air_temp_c |
| **Status** | is_on_track, on_pit_road, incident_count |

### Session Log Format
```json
{
  "metadata": {"session_id", "start_time", "track", "car"},
  "events": [
    {"timestamp", "lap", "event_type": "telemetry", "data": {...}},
    {"timestamp", "lap", "event_type": "llm", "data": {"prompt", "response", "latency_ms"}}
  ]
}
```

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **pyirsdk** - iRacing telemetry SDK
- **aiohttp** - Async HTTP for LLM calls
- **Piper** - Fast local TTS
- **LM Studio** - Local LLM inference
- **Mistral 7B** - Base language model

## Future Work

- [x] Gap tracking to cars ahead/behind
- [x] Session flags and incident tracking
- [ ] Fine-tune Mistral 7B on race engineer responses using QLoRA
- [ ] Weather and track condition awareness
- [ ] Multi-class race support

## License

MIT

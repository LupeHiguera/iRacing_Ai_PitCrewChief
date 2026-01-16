# iRacing AI Pit Crew Chief

A real-time AI race engineer that provides voice strategy updates during iRacing sessions. Uses local LLM inference for low-latency responses and Piper TTS for natural speech output.

## Features

- **Real-time telemetry** - Reads 43 telemetry fields from iRacing via pyirsdk
- **Smart strategy calculations** - Tracks fuel consumption, calculates pit windows, monitors tire degradation
- **Event detection** - Detects 25 race situations (battles, lockups, position changes, flags, etc.)
- **Car/Track metadata** - 60+ cars and 65+ tracks with corner mappings for contextual callouts
- **LLM-powered callouts** - Natural language updates via local Llama 3.1 8B inference
- **Voice output** - Text-to-speech using Piper for hands-free racing
- **Session logging** - Captures telemetry and LLM responses for fine-tuning data

## Demo

[Youtube](https://www.youtube.com/watch?v=D0YSpfuh97o)

## Architecture

```
iRacing → Telemetry Reader → Strategy Calculator ─┬→ Event Detector → LLM Client → TTS → Audio
                                                  │
                                                  └→ Session Logger → Fine-tuning Data
```

## Event Detection

The AI race engineer detects and responds to 25 real racing situations:

| Category | Events | Example Callout |
|----------|--------|-----------------|
| **Position** | Gained/lost (batched) | "Gained 3 positions, now P6" |
| **Battle** | Gap closing, Defend, Clear | "Defend! 0.5 behind" |
| **Dirty Air** | Following too close | "In dirty air, manage the temps" |
| **Clean Air** | Gap opens up | "Clean air now, push" |
| **Tire Temps** | Cold, Optimal, Hot | "Fronts running hot, ease the braking" |
| **Lockup/Spin** | Sudden temp spike | "Lockup! Easy on the brakes" |
| **Pace** | Dropping, Improving | "Pace dropping, tires going off?" |
| **Race Progress** | Halfway, Laps remaining | "5 laps remaining, bring it home" |
| **Flags** | Yellow, Green | "Yellow flag, caution" |
| **Incidents** | 1x, 4x penalties | "Incident, that's 4x total" |
| **Pit Lane** | Entry, Exit | "Out of the pits, push now" |
| **Personal Best** | New fastest lap | "Personal best! Great lap" |

Position changes are **batched** to avoid lap 1 chaos spam - the system waits for positions to settle before reporting.

## Car & Track Metadata

The system includes metadata for **60+ cars** and **65+ tracks** with corner mappings:

### Car Coverage
| Category | Count | Examples |
|----------|-------|----------|
| Production/Entry | 11 | MX-5 Cup, BMW M2 CSR, Spec Racer Ford |
| TCR/Touring | 4 | Audi RS 3 LMS, Honda Civic Type R TCR |
| GT4 | 6 | Porsche 718 Cayman, Mercedes-AMG GT4 |
| GT3 | 12 | Ferrari 296/488, BMW M4, Porsche 911 GT3 R |
| GTE | 5 | Corvette C8.R, Ferrari 488 GTE |
| GTP/LMDh/Hypercar | 5 | Ferrari 499P, Porsche 963, BMW M Hybrid V8 |
| F1 | 4 | Mercedes W12/W13, McLaren MP4-30 |
| IndyCar/Formula | 9 | Dallara IR18, Super Formula SF23, Dallara F3 |
| Ovals | 26 | Superspeedways, Intermediates, Short Tracks |

### Track Coverage
| Region/Type | Count | Examples |
|-------------|-------|----------|
| European Road | 17 | Spa, Monza, Silverstone, Nurburgring |
| North American Road | 13 | Road America, Laguna Seca, COTA |
| Japanese/Asian | 5 | Suzuka, Fuji, Okayama |
| Ovals | 26 | Daytona, Talladega, Bristol, Martinsville |

Each track includes **corner mappings** with lap percentage positions for contextual callouts.

## Requirements

- Windows (iRacing is Windows-only)
- Python 3.10+
- [iRacing](https://www.iracing.com/) subscription
- [LM Studio](https://lmstudio.ai/) with Llama 3.1 8B Instruct
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

Create a `.env` file or edit `config.py`:

```python
# Paths
PIPER_PATH=C:/tools/piper/piper.exe
TTS_MODEL_PATH=C:/tools/piper/en_US-lessac-medium.onnx
LM_STUDIO_URL=http://localhost:1234/v1

# Strategy thresholds
fuel_warning_laps: float = 5.0
fuel_critical_laps: float = 2.0
tire_warning_pct: float = 70.0
tire_critical_pct: float = 85.0

# Event detection thresholds
gap_battle_threshold_sec: float = 0.8   # "Defend!" when under this
gap_close_threshold_sec: float = 1.5    # "Car closing" alert
tire_temp_hot_c: float = 110.0          # Overheating warning
tire_temp_cold_c: float = 60.0          # Cold tire warning
```

## Usage

1. Start iRacing and load into a session
2. Start LM Studio with Llama 3.1 8B Instruct model
3. Run the strategist:

```bash
python -m src.main
```

The AI will provide voice updates based on race events - not just periodic lap updates, but reactive callouts when interesting things happen.

## Project Structure

```
src/
├── telemetry.py      # iRacing data via pyirsdk (43 fields)
├── strategy.py       # Fuel/tire calculations
├── event_detector.py # Detects 25 race events
├── llm_client.py     # LM Studio API client + JSON format
├── metadata.py       # 60+ cars, 65+ tracks with corner mappings
├── tts.py            # Piper TTS wrapper
├── logger.py         # Session logging to gzip JSON
└── main.py           # Main integration

data/
├── sessions/         # Logged race sessions (gzip JSON)
└── synthetic/        # Generated training data (for fine-tuning)

tests/
├── test_telemetry.py    # 30 tests
├── test_metadata.py     # 35 tests
└── test_llm_client.py   # 45 tests (includes JSON format)
```

## Data Collection

Sessions are automatically logged to `data/sessions/` as gzip-compressed JSON files.

**Current stats:**
- 16 sessions logged
- 3,300+ telemetry samples
- 200+ LLM calls with prompts/responses/latency
- Tracks: Lime Rock, Okayama, Mugello, Monza, Red Bull Ring
- Cars: Mazda MX-5, BMW M2 CS, BMW M4 GT3 EVO

### Telemetry Fields (43 total)

| Category | Fields |
|----------|--------|
| **Lap/Position** | lap, lap_pct, position, lap_delta_to_best |
| **Fuel** | fuel_level, fuel_level_pct, fuel_use_per_hour |
| **Tire Wear** | tire_wear_lf/rf/lr/rr (% worn) |
| **Tire Temps** | tire_temp_*_l/m/r (12 fields, L/M/R per corner) |
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

110+ tests covering telemetry, metadata, and LLM client functionality.

## Tech Stack

- **pyirsdk** - iRacing telemetry SDK
- **aiohttp** - Async HTTP for LLM calls
- **Piper** - Fast local TTS
- **LM Studio** - Local LLM inference
- **Llama 3.1 8B** - Base language model (to be fine-tuned)

## Roadmap

### Week 1: MVP (Complete)
- [x] Telemetry reading (43 fields)
- [x] Strategy calculations (fuel, tire wear)
- [x] LLM integration with LM Studio
- [x] TTS output with Piper
- [x] Session logging
- [x] Gap tracking to cars ahead/behind
- [x] Event detection system (25 event types)
- [x] Car metadata (60+ cars with class, traits, advice style)
- [x] Track metadata (65+ tracks with corner mappings)
- [x] JSON prompt format for fine-tuned model

### Week 2: Data Collection (In Progress)
- [x] 16 sessions logged
- [ ] Generate 10,000+ synthetic training examples
- [ ] Validate response quality and grounding

### Week 3: Fine-tuning
- [ ] Fine-tune Llama 3.1 8B using QLoRA
- [ ] Train on race engineer responses
- [ ] Export to GGUF for LM Studio

### Week 4: Evaluation
- [ ] Compare base vs fine-tuned model
- [ ] Document results
- [ ] Polish and release

## Known Limitations

| Issue | Cause | Workaround |
|-------|-------|------------|
| Tire temps static on some cars | MX-5/M2 lack TPMS | Use GT3/GTE cars |
| Tire wear minimal in short sessions | Not enough laps | Longer stints |
| Lockup detection needs GT3+ | Lower-tier cars don't report temps | Use GT3/GTE/F3 |
| Tire temps may cache in AI races | iRacing SDK quirk | Use official/hosted races |

## License

MIT

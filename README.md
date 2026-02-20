# iRacing AI Pit Crew Chief

A real-time AI race engineer that provides voice strategy updates during iRacing sessions. Fine-tuned Llama 3.1 8B using QLoRA on 9,200+ synthetic race engineer examples. Runs locally with LM Studio for low-latency inference and Piper TTS for natural speech output.

## Demo

[Youtube](https://www.youtube.com/watch?v=D0YSpfuh97o)

## Architecture

```
iRacing SDK
    │
    ▼
Telemetry Reader (43 fields @ 1Hz)
    │
    ▼
Strategy Calculator (fuel, tires, pit window)
    │
    ├──► Event Detector (25 event types)
    │         │
    │         ▼
    │    LLM Client (fine-tuned Llama 3.1 8B via LM Studio)
    │         │
    │         ▼
    │    Piper TTS ──► Audio Output
    │
    └──► Session Logger ──► Fine-tuning Data
```

**End-to-end latency:** 800-1100ms typical (telemetry to speech)

## Fine-Tuning Results

Fine-tuned Llama 3.1 8B Instruct using QLoRA on 9,269 synthetic training examples across 10 race scenario categories. Evaluated on 55 held-out test cases with 12 weighted metrics.

### Base vs Fine-Tuned (55 test cases)

| Metric | Base Llama 3.1 8B | Fine-Tuned |
|--------|-------------------|------------|
| **Average Score** | 29.7 / 100 | **78.5 / 100** |
| **Win Rate** | 0% | **100%** |
| **Concise (< 40 words)** | 0 / 55 | **55 / 55** |
| **TTS Suitable** | 0 / 55 | **55 / 55** |
| **Hallucinations** | 48 / 55 | **0 / 55** |
| **Track References** | 7 / 55 | **27 / 55** |
| **Avg Latency** | 4,881 ms | **3,570 ms** |

### Score by Category

| Category | Base | Fine-Tuned |
|----------|------|------------|
| Gap Management | 43.4 | **95.6** |
| Routine Updates | 39.9 | **87.7** |
| Pace Feedback | 48.8 | **87.0** |
| Fuel Critical | 22.2 | **85.7** |
| Position Battle | 27.4 | **78.0** |
| Tire Cold | 25.2 | **74.8** |
| Tire Critical | 31.8 | **72.8** |
| Pit Approach | 15.0 | **71.0** |
| Tire Warning | 18.2 | **68.4** |
| Fuel Warning | 23.2 | **59.4** |

### Before / After Examples

**Fuel Critical at Spa (lap 18, 1.8 laps of fuel, P4):**
- Base: *"Hey, 3rd place, 85% of the lap..."* (verbose, wrong position, no urgency)
- **Fine-tuned:** *"Box this lap, critical fuel. P4 will close that 4.2 second gap. Manage tires through Pouhon."*

**Tire Warning at Barcelona (FR at 108C, 52% worn):**
- Base: *"Alright, driver! We're on lap 18, 35% through the stint..."* (generic, no action)
- **Fine-tuned:** *"Front right temp one-oh-eight, wear at fifty-two percent. Ease trail braking into Turn 5."*

**Battle at Nurburgring (P3, gap 0.4s to P2):**
- Base: *"Hey driver, we're in a good spot, P3 on lap 10..."* (no urgency, hallucinated names)
- **Fine-tuned:** *"P3, gap four-tenths. Use your strong brakes into Veedol, be aggressive on exit."*

## Features

- **Real-time telemetry** - Reads 43 fields from iRacing via pyirsdk at 1Hz
- **Strategy calculations** - Fuel consumption tracking, pit window estimation, tire degradation monitoring
- **25 event types** - Position changes, battles, lockups, wheelspin, flags, pace trends, and more
- **Car/track awareness** - 60+ cars and 65+ tracks with corner mappings for contextual callouts
- **Fine-tuned LLM** - QLoRA-trained Llama 3.1 8B that infers situations from raw telemetry
- **Voice output** - Piper TTS for hands-free racing
- **Session logging** - Captures telemetry and LLM responses for training data

## Event Detection

The AI detects and responds to 25 race situations:

| Category | Events | Example |
|----------|--------|---------|
| **Position** | Gained/lost (batched) | "Gained 3 positions, now P6" |
| **Battle** | Gap closing, Defend, Clear | "Defend! 0.5 behind" |
| **Dirty/Clean Air** | Following, gap opens | "Clean air now, push" |
| **Tire Temps** | Cold, Optimal, Hot | "Fronts running hot, ease the braking" |
| **Lockup/Spin** | Sudden temp spike | "Lockup! Easy on the brakes" |
| **Pace** | Dropping, Improving, PB | "Personal best! Great lap" |
| **Race Progress** | Halfway, Laps remaining | "5 laps remaining, bring it home" |
| **Flags** | Yellow, Green | "Yellow flag, caution" |
| **Pit Lane** | Entry, Exit | "Out of the pits, push now" |

## Fine-Tuning Details

### Training Setup

| Parameter | Value |
|-----------|-------|
| Base Model | Llama 3.1 8B Instruct |
| Method | QLoRA (4-bit quantization + LoRA adapters) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning Rate | 2e-4 |
| Epochs | ~2.6 (stopped at step 1500/1653) |
| Effective Batch Size | 16 (4 x 4 gradient accumulation) |
| Training Examples | 9,269 |
| Best Eval Loss | 0.1025 (step 1500) |
| Trainable Parameters | ~0.1% of total |
| Hardware | NVIDIA RTX 5060 Ti 16GB |

### Training Data

9,269 synthetic examples generated via Claude API across 10 categories:

| Category | % | Description |
|----------|---|-------------|
| Fuel Critical | 10% | Box now, DNF risk |
| Fuel Warning | 12% | Fuel getting low |
| Tire Critical | 10% | Tires gone |
| Tire Warning | 12% | Overheating, wearing |
| Tire Cold | 8% | Not up to temp |
| Position Battle | 12% | Attacking / defending |
| Gap Management | 10% | Dirty air, clean air, undercut |
| Pit Approach | 8% | Coming into pits |
| Pace Feedback | 8% | PB, slower, improving |
| Routine | 10% | Status updates |

Each example pairs raw telemetry JSON (car, track, temps, gaps, fuel, position) with a concise, TTS-suitable race engineer response. The model learns to infer situations from data rather than being given semantic labels.

### Training Data Format

```json
{
  "instruction": "You are a race engineer. Given the car, track, and telemetry, provide a brief callout.",
  "input": {
    "car": "BMW M4 GT3", "car_class": "GT3",
    "car_traits": ["rear_limited", "strong_brakes", "aero_dependent"],
    "track": "Monza", "track_type": "high_speed",
    "lap": 12, "lap_pct": 0.65, "position": 8,
    "fuel_laps_remaining": 14,
    "tire_wear": {"fl": 15, "fr": 22, "rl": 12, "rr": 14},
    "tire_temps": {"fl": 95, "fr": 102, "rl": 88, "rr": 91},
    "gap_ahead": 2.1, "gap_behind": 0.8,
    "last_lap_time": 91.5, "best_lap_time": 90.8,
    "session_laps_remain": 18, "incident_count": 2, "track_temp_c": 35
  },
  "output": "Front right is hot. Short shift into Lesmo 1, protect that tire through the chicane."
}
```

### Evaluation Framework

55 test cases scored on 12 weighted metrics:

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| Urgency Match | 15% | Response urgency fits the situation |
| Required Element | 15% | Contains must-have keywords (e.g., "box" for fuel critical) |
| Concise | 10% | Under 40 words |
| TTS Suitable | 10% | 10-35 words, no markdown formatting |
| Telemetry Reference | 10% | References actual data values |
| Actionable | 10% | Contains specific action for the driver |
| Track Reference | 8% | Mentions track corners by name |
| Car-Appropriate | 8% | Advice matches car characteristics |
| Specific | 8% | References specific numbers |
| Correct Values | 6% | Numbers match input data |
| Hallucination | -20% penalty | Fake names, wrong systems |

## Requirements

- Windows (iRacing is Windows-only)
- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM (for fine-tuning; inference works with less)
- [iRacing](https://www.iracing.com/) subscription
- [LM Studio](https://lmstudio.ai/) for local inference
- [Piper TTS](https://github.com/rhasspy/piper/releases) for voice output

## Installation

```bash
git clone https://github.com/yourusername/iRacing_Ai_PitCrewChief.git
cd iRacing_Ai_PitCrewChief

pip install -r requirements.txt

# Download Piper and a voice model
# Place piper.exe and en_US-lessac-medium.onnx in C:/tools/piper/
```

## Usage

### Running the Race Engineer

1. Start iRacing and load into a session
2. Start LM Studio with the fine-tuned model (or base Llama 3.1 8B)
3. Run:

```bash
python -m src.main
```

### Fine-Tuning

```bash
# Generate synthetic training data
export ANTHROPIC_API_KEY=your-key
python scripts/generate_data.py

# Fine-tune with QLoRA
python scripts/finetune.py

# Resume from checkpoint if interrupted
python scripts/finetune.py --resume models/race-engineer-lora/checkpoint-800

# Export merged model
python scripts/export_model.py
```

### Evaluation

```bash
# Full comprehensive eval (55 test cases, base vs fine-tuned)
python scripts/eval_comprehensive.py

# Simple eval (8 test cases)
python scripts/evaluate.py
```

### Converting to GGUF for LM Studio

```bash
# After exporting merged model
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release

# Convert and quantize
python llama.cpp/convert_hf_to_gguf.py models/race-engineer-merged --outfile race-engineer.gguf
./llama.cpp/build/bin/llama-quantize race-engineer.gguf race-engineer-q4_k_m.gguf q4_k_m

# Load the .gguf file in LM Studio
```

## Project Structure

```
src/
├── telemetry.py        # iRacing data via pyirsdk (43 fields)
├── strategy.py         # Fuel/tire calculations + pit window
├── event_detector.py   # 25 race event types with cooldowns
├── llm_client.py       # LM Studio API client + JSON prompt format
├── metadata.py         # 60+ cars, 65+ tracks with corner mappings
├── tts.py              # Piper TTS wrapper with priority queue
├── logger.py           # Session logging to gzip JSON
└── main.py             # Main loop with event-driven LLM calls

scripts/
├── generate_data.py    # Synthetic training data via Claude API
├── finetune.py         # QLoRA fine-tuning with checkpoint resume
├── export_model.py     # Merge LoRA adapter with base model
├── eval_comprehensive.py  # 55-case eval with 12 metrics
├── evaluate.py         # Simple 8-case eval
└── clean_data.py       # Training data cleanup

data/
├── sessions/           # 28 logged race sessions (gzip JSON)
├── synthetic/          # 9,269 training examples
└── eval_comprehensive.json  # Evaluation results

models/
├── race-engineer-lora/    # LoRA adapter weights
└── race-engineer-merged/  # Full merged model (for GGUF conversion)

tests/
├── test_telemetry.py   # 30 tests
├── test_metadata.py    # 35 tests
└── test_llm_client.py  # 45 tests
```

## Known Limitations

| Issue | Cause | Workaround |
|-------|-------|------------|
| Tire temps static on some cars | MX-5/M2 lack TPMS | Use GT3/GTE cars |
| Tire wear minimal in short sessions | Not enough laps | Longer stints |
| Lockup detection needs GT3+ | Lower-tier cars don't report temps | Use GT3/GTE/F3 |

## Tech Stack

- **pyirsdk** - iRacing telemetry SDK
- **transformers + peft + bitsandbytes** - QLoRA fine-tuning
- **aiohttp** - Async HTTP for LLM calls
- **Piper** - Fast local TTS
- **LM Studio** - Local LLM inference
- **Llama 3.1 8B Instruct** - Base model

## License

MIT

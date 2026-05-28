# iRacing AI Pit Crew Chief

A real-time AI race engineer that provides voice strategy updates during iRacing sessions. Fine-tuned Llama 3.1 8B using QLoRA on 9,200+ synthetic race engineer examples. Runs locally with LM Studio for low-latency inference and Piper TTS for natural speech output.

**[Project write-up on higuera.io →](https://higuera.io/projects/iracing-strategist)**

## Demo

[YouTube demo](https://www.youtube.com/watch?v=15d633gyyfY)

## Architecture

```
iRacing SDK
    │
    ▼
Telemetry Reader (49 fields @ 1Hz)
    │
    ├──► Tire-State Estimator (per-corner temp/wear from live G-load)
    │
    ▼
Strategy Calculator (fuel, tires, pit window)
    │
    ├──► Event Detector (race events)
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

### Comparison Dashboard — Pick a Scenario

> Every response below is the **actual model output** from `data/eval_comprehensive.json`. Scores are the composite 12-metric scorer (0–100). Want it interactive? See [docs/dashboard.html](docs/dashboard.html) ([live](https://lupehiguera.github.io/iRacing_Ai_PitCrewChief/dashboard.html)).

<details>
<summary><b>🚨 Fuel Critical</b> — Spa, P3, 0.8 laps of fuel &nbsp;·&nbsp; <code>Base 12 → FT 100 &nbsp;(+88)</code></summary>

> **Base Llama 3.1 8B:** *"Hey, 3rd place, 85% of the lap completed. Fuel's looking good, you've got enough to make it to the end. Tire wear's not too bad, but we're a bit soft in the rear. Temps are getting hot, especially on the rears, so be careful not to push too hard."*
> — Verbose, wrong about fuel, no call to box.
>
> **Fine-tuned:** *"Box this lap, critical fuel. P4 will close that 4.2 second gap. Manage rear weight transfer through Bus Stop, trail brake smoothly in."*
> — Correct urgency, names the corner (Bus Stop), references gap, car-appropriate (Porsche rear-engine).

</details>

<details>
<summary><b>⛽ Fuel Warning</b> — Nürburgring GP, P3, 3.8 laps &nbsp;·&nbsp; <code>Base 4 → FT 69 &nbsp;(+65)</code></summary>

> **Base:** *"Hey, driver! We're on lap 22, 42% of the way through the stint. You're currently in third place, 1.5 seconds ahead of the car in front and 2.8 seconds behind the leader. Tire wear is looking good..."*
> — 60+ words, swaps "ahead" and "behind," no strategy call.
>
> **Fine-tuned:** *"Box this lap, box this lap. Fuel critical at three-point-eight remaining. We'll take four tires, P4 on the out-lap. Push through final sector."*
> — Urgency, exact value, projected out-position, action.

</details>

<details>
<summary><b>🔥 Tire Critical</b> — Spa, McLaren 720S GT3, FR destroyed &nbsp;·&nbsp; <code>Base 19 → FT 76 &nbsp;(+57)</code></summary>

> **Base:** *"Alright, driver, we're in a good position, P4 with 6 laps to go. Tyres are looking good, with the fronts not too worn and the rears still relatively fresh..."*
> — Contradicts the input (FR is destroyed), reassures instead of warning.
>
> **Fine-tuned:** *"Front right critical at one-fifteen degrees. Protect it through Blanchimont and Bus Stop. We'll box next time by, six laps fuel remaining."*
> — Specific temp, two corner names, plan.

</details>

<details>
<summary><b>♨️ Tire Warning</b> — Silverstone, Ferrari 296 GT3, rear deg &nbsp;·&nbsp; <code>Base 19 → FT 84 &nbsp;(+65)</code></summary>

> **Base:** *"Hey driver, it's your engineer. We're in a good spot, P5 with 18 laps to go. The tires are looking good, FL at 35, FR at 38, RL at 55, and RR at 58. Temperatures are a bit high, FL at 92, FR at 94, RL at 104, and RR at 106."*
> — Reads telemetry like a dashboard; no advice.
>
> **Fine-tuned:** *"Rear tires at one-oh-four, two degrees over. Ease the aggression through Maggotts-Becketts, manage that rear downforce. Twenty-two to go, fuel's good."*
> — Translates data into a specific cornering instruction.

</details>

<details>
<summary><b>❄️ Tire Cold</b> — Monza, green track, P10 battle &nbsp;·&nbsp; <code>Base 34 → FT 100 &nbsp;(+66)</code></summary>

> **Base:** *"Alright, driver, here's the situation. We're on lap 3, 58% complete. You're currently in 10th position, with 28 laps of fuel left. Tire wear is moderate, with the rears showing some signs of degradation..."*
> — Misses the actual issue (cold tires) entirely.
>
> **Fine-tuned:** *"Tires still cold, especially left side. Push hard through Lesmo 1 and 2, use the downforce. Gap behind closing, nine-tenths."*
> — Correct diagnosis, prescribes warm-up sectors, surfaces threat.

</details>

<details>
<summary><b>⚔️ Position Battle</b> — Road America, defending P2, 0.3s &nbsp;·&nbsp; <code>Base 4 → FT 69 &nbsp;(+65)</code></summary>

> **Base:** *"Alright, driver, we're looking good here. You're 2nd place with 8 laps to go. Your tires are holding up well, **flintstones** are at 28% wear..."*
> — Hallucinated word ("flintstones"), zero urgency despite 0.3s gap.
>
> **Fine-tuned:** *"Car behind three-tenths, defend into Canada Corner. Use your brakes late and rotate the car, he's pushing too hard. Fuel's tight, eight laps remaining."*
> — Tactical, corner-specific, identifies opponent error.

</details>

<details>
<summary><b>🌬️ Gap Management</b> — Road America, building gap to P5 &nbsp;·&nbsp; <code>Base 34 → FT 100 &nbsp;(+66)</code></summary>

> **Base:** *"Hey, you're doing a great job out there! You're in P4, 5.2 seconds behind the leader and 2.8 seconds ahead of the car behind you..."*
> — Praise, no action.
>
> **Fine-tuned:** *"P4, gap behind closing two-eight. Push now, use those strong brakes into Canada Corner. We need one-two-six to close the five seconds ahead."*
> — Target lap time, specific corner, car-trait reference.

</details>

<details>
<summary><b>🅿️ Pit Approach</b> — Monza, splash-and-dash, 1.2 laps &nbsp;·&nbsp; <code>Base 4 → FT 92 &nbsp;(+88)</code></summary>

> **Base:** *"Hey driver, it's your engineer. We're in a good position, P4, with 1.2 laps left on the tank. Tyres are looking okay..."*
> — "Good position" while running out of fuel.
>
> **Fine-tuned:** *"Box this lap, critical fuel. Push through Ascari, then straight in. P4 secure, five second gap behind."*
> — Names the corner before pit entry; confirms it's safe to keep position.

</details>

<details>
<summary><b>📈 Pace Feedback</b> — Nürburgring, improving trend &nbsp;·&nbsp; <code>Base 34 → FT 84 &nbsp;(+50)</code></summary>

> **Base:** *"Hey, you're looking good out there. The car's handling is solid, as expected. You're 7th on the grid, and the gaps are manageable..."*
> — Vague encouragement.
>
> **Fine-tuned:** *"P7, gap ahead 3.8. Just set fastest lap, keep this pace. Front right running hot at 92, manage it through Ford Kurve. Fuel's good to the end.*"
> — Confirms the trend, flags emerging tire concern, prescribes a corner.

</details>

<details>
<summary><b>🏁 Routine</b> — Monza, clean run mid-stint &nbsp;·&nbsp; <code>Base 34 → FT 92 &nbsp;(+58)</code></summary>

> **Base:** *"Alright, driver, this is your engineer. We're on lap 12, 35% through the stint. You're in fourth position, with a gap of 2.8 seconds to the leader..."*
> — Sounds like a status readout.
>
> **Fine-tuned:** *"P4, gap ahead two-eight. Front right temp climbing to 94. Ease entry aggression through Roggia, let the aero work. Fuel's good to the end."*
> — Even routine updates name corners and prescribe action.

</details>

#### What the dashboard shows

| Failure mode (base) | Fine-tuned behavior |
|---------------------|---------------------|
| 60+ word stat dumps | Under 25 words, TTS-ready |
| Reads numbers without interpreting | Translates numbers into corner-specific actions |
| Contradicts inputs ("tires look good" when destroyed) | Stays grounded in actual telemetry |
| Hallucinates words and positions | Zero hallucinations across 55 cases |
| No urgency markers | "Box this lap," "Defend," matched to situation |
| Car-agnostic generic advice | References car traits (Porsche rear-engine, BMW brakes) |

## Features

- **Real-time telemetry** - Reads 49 fields from iRacing via pyirsdk at 1Hz
- **Tire-state estimation** - iRacing exposes no live tire-temp channel, so per-corner temp/wear is reconstructed from live load (lateral/longitudinal G, brake, speed) and anchored to the real pit-stop measurements
- **Strategy calculations** - Fuel consumption tracking, pit window estimation, tire degradation monitoring
- **Race event detection** - Position changes, battles, gap/dirty-air, tire temps, flags, pace trends, pit lane
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
| **Tire Temps** (estimated) | Cold, Optimal, Hot | "Fronts running hot, ease the braking" |
| **Pace** | Dropping, Improving, PB | "Pace dropping, tires may be going off" |
| **Race Progress** | Halfway, Laps remaining | "5 laps remaining, bring it home" |
| **Flags** | Yellow, Green | "Yellow flag, caution" |
| **Pit Lane** | Entry, Exit | "Out of the pits, push now" |

> Tire-temp events fire off the [tire-state estimate](#tire-state-estimation). Lockup/wheelspin detection exists in the code but is disabled on iRacing — it needs sub-second temp spikes the SDK never streams.

## Tire State Estimation

iRacing's tire telemetry is a trap: `tireTempC*` and `tireWear*` only refresh when the pit
crew measures the tires — at spawn and each pit stop — and are **frozen during a stint** (tire
pressure reads `0`). I confirmed this from the session logs: a 24-lap GT3 run updated tire temps
exactly twice (spawn + the lap after a pit stop). Feeding the model that frozen cold value made
it report "tires cold" for the entire race.

Since iRacing *does* stream the load signals the tires respond to, `src/tire_model.py`
reconstructs a per-corner temp/wear estimate instead:

- **Heat from work, not pedals** — lateral and longitudinal **G-forces** drive heating (cornering
  loads the outer pair, braking the fronts, acceleration the rears). Using G rather than throttle
  avoids cooking the rears on long full-throttle straights, where steady throttle does no tire work.
- **Cooling** relaxes toward the live track/air temp, with extra airflow cooling at speed.
- **Anchored to ground truth** — at spawn and every pit exit the estimate resets to iRacing's real
  crew measurement, so each stint starts from a true value and extrapolates from there.
- **Wear** accrues with load, scaled by each car's degradation trait.

Estimated values feed the model prompt and the tire events, and show as **ESTIMATED** in the
overlay (never passed off as live). On a validation pit stop at Monza the estimate tracked the
correct relative pattern (hottest corner, left/right bias) and ran ~10-15°C above the crew's
*carcass* measurement — expected, since the estimate approximates *surface* temp and the tires
cool on the in-lap. `main.py` logs estimate-vs-measured error at each stop for ongoing calibration.

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
git clone https://github.com/LupeHiguera/iRacing_Ai_PitCrewChief.git
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
├── telemetry.py        # iRacing data via pyirsdk (49 fields)
├── tire_model.py       # Per-corner tire temp/wear estimator (live G-load + pit anchor)
├── strategy.py         # Fuel/tire calculations + pit window
├── event_detector.py   # Race event types with cooldowns
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
├── clean_data.py       # Training data cleanup
└── diag_tire_temps.py  # Diagnostic: dump live tire telemetry update rates

data/
├── sessions/           # 28 logged race sessions (gzip JSON)
├── synthetic/          # 9,269 training examples
└── eval_comprehensive.json  # Evaluation results

models/
├── race-engineer-lora/    # LoRA adapter weights
└── race-engineer-merged/  # Full merged model (for GGUF conversion)

tests/
├── test_telemetry.py       # 30 tests
├── test_metadata.py        # 35 tests
├── test_llm_client.py      # 46 tests
├── test_tire_model.py      # 13 tests (estimator behavior)
├── test_event_detector.py  # 4 tests (estimate-driven tire events)
└── test_main.py            # 21 tests
```

## Known Limitations

| Issue | Cause | Workaround |
|-------|-------|------------|
| No live tire temps/wear (all cars) | iRacing's SDK only refreshes them when the crew measures the tires — at spawn and pit stops | [Tire-state estimator](#tire-state-estimation) reconstructs them from live G-load, anchored to the pit measurements |
| Estimate reads ~10-15°C above the pit measurement | Estimator approximates surface temp; tires also cool on the in-lap, and the crew measures carcass temp | Relative pattern (hot corner, L/R bias) is accurate; per-car calibration from pit checks is possible |
| Lockup/wheelspin events disabled | Need sub-second temp spikes the SDK never streams | Pace degradation is used as the live tire-wear proxy instead |

## Tech Stack

- **pyirsdk** - iRacing telemetry SDK
- **transformers + peft + bitsandbytes** - QLoRA fine-tuning
- **aiohttp** - Async HTTP for LLM calls
- **Piper** - Fast local TTS
- **LM Studio** - Local LLM inference
- **Llama 3.1 8B Instruct** - Base model

## License

MIT

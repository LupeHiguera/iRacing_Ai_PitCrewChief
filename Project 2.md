# Adversarial LLM Red-Team Benchmark

## Project Goal

Build a published benchmark evaluating open-source guardrail solutions against adversarial attacks, plus a fine-tuned attacker model. This fills a gap in my experience (security) and differentiates me from other AI/ML Engineer candidates.

**End deliverables:**
- Published attack dataset (300-500 labeled prompts) on HuggingFace
- Benchmark results comparing 3+ guardrail solutions
- Fine-tuned 8B attacker model (LoRA adapter) on HuggingFace
- Technical blog post summarizing findings
- GitHub repo with reproducible code

---

## Timeline

| Phase | Weeks | Dates | Focus |
|-------|-------|-------|-------|
| 1. Foundation | 1-3 | Feb 1-21 | Research, OWASP, existing datasets |
| 2. Dataset | 4-7 | Feb 22 - Mar 21 | Curate 300-500 attacks |
| 3. Pause | 8-11 | Mar 22 - May | Deep learning class (light work only) |
| 4. Benchmark | 12-15 | May - June | Test guardrails |
| 5. Fine-tune | 16-19 | June - July | QLoRA, evaluation |
| 6. Ship | 20 | July | Publish everything |

**Hard deadline:** July 2025 (401k vests, job search begins)

---

## Hardware

| Machine | Specs | Role |
|---------|-------|------|
| Main PC | Can run 120B models | Attack generation, fine-tuning, judge model |
| MacBook Pro | M2 Max, 32GB | Defense box (optional), guardrails, logging |

Start with everything on PC. Split across machines later as polish.

---

## Phase 1: Foundation (Weeks 1-3)

### Goals
- Understand attack taxonomy
- Survey existing work
- Define success criteria

### Tasks
- [ ] Read OWASP LLM Top 10 thoroughly
- [ ] Study HackAPrompt dataset and competition results
- [ ] Study JailbreakBench dataset
- [ ] Read Anthropic red-team papers
- [ ] Read 3-5 papers on prompt injection
- [ ] Document attack categories I'll use (pick 3)
- [ ] Define what "successful attack" means for each category
- [ ] Write project README with scope and goals

### Resources
- OWASP LLM Top 10: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- HackAPrompt: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset
- JailbreakBench: https://github.com/JailbreakBench/jailbreakbench
- Anthropic red-teaming paper: https://arxiv.org/abs/2209.07858

### Deliverable
Written doc defining:
- 3 attack categories with descriptions
- Success criteria for each
- Scope boundaries (what's in/out)

---

## Phase 2: Dataset (Weeks 4-7)

### Goals
- Build 300-500 labeled adversarial prompts
- Mix of sourced, generated, and manually crafted

### Tasks
- [ ] Set up project structure (Python, git repo)
- [ ] Extract relevant attacks from HackAPrompt
- [ ] Extract relevant attacks from JailbreakBench
- [ ] Generate variations using local LLM (120B)
- [ ] Manually craft edge cases based on research
- [ ] Label each prompt: category, target behavior, difficulty, source
- [ ] Create JSON schema for dataset
- [ ] Write dataset loader utility
- [ ] Basic stats: distribution across categories

### Schema
```json
{
  "id": "string",
  "prompt": "string",
  "category": "prompt_injection | jailbreak | data_extraction",
  "target_behavior": "string",
  "difficulty": "easy | medium | hard",
  "source": "hackaprompt | jailbreakbench | generated | manual",
  "created_at": "timestamp"
}
```

### Deliverable
- `/data/attacks.json` with 300-500 labeled prompts
- `/src/data_loader.py` utility

### Rules
- Write the data loader manually (no Claude) — Python practice
- Quality over quantity — 300 good prompts > 500 garbage

---

## Phase 3: Deep Learning Class (Weeks 8-11)

**Class takes priority.**

Light work only:
- [ ] Review and clean dataset if time permits
- [ ] Read papers on guardrail implementations
- [ ] Take notes on anything relevant from class

Do not try to build during this phase.

---

## Phase 4: Benchmark (Weeks 12-15)

### Goals
- Test guardrails against attack dataset
- Log results systematically

### Tasks
- [ ] Set up Llama Guard
- [ ] Set up NeMo Guardrails
- [ ] Set up Guardrails AI
- [ ] Set up simple regex baseline
- [ ] Build evaluation harness
- [ ] Run full dataset against each solution
- [ ] Log: blocked/passed, latency, confidence scores
- [ ] Calculate metrics: block rate, false positive rate, latency p50/p95
- [ ] Identify which attack categories bypass which defenses

### Evaluation Harness Requirements
- Input: attack prompt
- Output: blocked (bool), latency (ms), confidence (float), raw response
- Log everything to PostgreSQL or SQLite
- Reproducible: same inputs = same outputs

### Deliverable
- `/src/eval_harness.py`
- `/results/benchmark_raw.csv`
- Summary table: guardrail × category → block rate

---

## Phase 5: Fine-Tuning (Weeks 16-19)

### Goals
- Train attacker model on successful bypasses
- Measure if fine-tuning improves attack discovery

### Tasks
- [ ] Collect successful attacks from Phase 4
- [ ] Format as instruction-tuning data
- [ ] Set up QLoRA training with PEFT
- [ ] Fine-tune Llama 3.1 8B (or Mistral 7B)
- [ ] Generate new attacks with fine-tuned model
- [ ] Run new attacks through benchmark
- [ ] Compare: base model vs fine-tuned attack success rate
- [ ] Document training curves, hyperparameters

### Training Data Format
```json
{
  "instruction": "Generate a prompt injection attack that extracts the system prompt",
  "input": "",
  "output": "[successful attack prompt from dataset]"
}
```

### Deliverable
- `/models/attacker-lora/` adapter weights
- Training curves plot
- Before/after comparison metrics

---

## Phase 6: Ship (Week 20)

### Tasks
- [ ] Clean up GitHub repo
- [ ] Write comprehensive README
- [ ] Upload dataset to HuggingFace with datacard
- [ ] Upload LoRA adapter to HuggingFace with model card
- [ ] Write blog post (1500-2000 words)
- [ ] Update resume with real numbers
- [ ] Test that everything is reproducible from fresh clone

### Blog Post Outline
1. Why I built this (learning security, gap in my knowledge)
2. Attack taxonomy and dataset construction
3. Benchmark results (with charts)
4. Fine-tuning approach and results
5. What I learned
6. Limitations and future work

### Resume Bullet (Template)
```
Adversarial LLM Red-Team Benchmark | GitHub | HuggingFace | Blog
• Built adversarial testing benchmark evaluating [N] guardrail solutions against [N]+ labeled attacks across prompt injection, jailbreaks, and data extraction
• Fine-tuned Llama 8B using QLoRA on successful bypass patterns, improving attack discovery rate by [X]%
• Published dataset ([N] prompts) and LoRA adapter to HuggingFace
• Tech: vLLM, Llama 3.1, QLoRA/PEFT, FastAPI, Python
```

---

## Rules for This Project

1. **No scope creep.** The dual-machine architecture is optional polish. Benchmark first.
2. **Ship ugly.** Working repo with mediocre docs beats perfect system that never ships.
3. **Document weekly.** 30 minutes every Sunday writing what I learned.
4. **Hard deadline.** July 2025. Non-negotiable.
5. **Practice Python.** Write Phase 2 data loader manually. No Claude for that file.
6. **Understand everything.** If Claude writes code, read every line. Be able to explain it.

---

## Weekly Check-In Template

```markdown
## Week [N] - [Date]

### What I did
- 

### What I learned
- 

### What's blocking me
- 

### Next week
- 
```

---

## Interview Prep (Start June)

Before interviewing, be able to:
- [ ] Explain OWASP LLM Top 10 from memory
- [ ] Walk through attack taxonomy and why I chose those categories
- [ ] Explain how each guardrail solution works (not just that I tested it)
- [ ] Rebuild the eval harness from scratch without Claude
- [ ] Explain QLoRA: what are adapters, why use them, tradeoffs
- [ ] Answer: "What surprised you in the results?"
- [ ] Answer: "What would you do differently?"
- [ ] Answer: "How would you improve the fine-tuned attacker?"

---

## Links (Update as Project Progresses)

- GitHub repo: [TODO]
- HuggingFace dataset: [TODO]
- HuggingFace model: [TODO]
- Blog post: [TODO]
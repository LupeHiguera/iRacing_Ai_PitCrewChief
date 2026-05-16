# Dashboard

`dashboard.html` is a standalone single-file dashboard for the iRacing AI Pit Crew Chief eval results. It reads `eval_comprehensive.json` (the canonical eval output from `scripts/eval_comprehensive.py`) and renders a side-by-side comparison of the base Llama 3.1 8B and the QLoRA fine-tuned model across 55 race-engineer scenarios.

## Hosting on GitHub Pages

In repo settings → Pages, set source to `main` branch, `/docs` folder. The dashboard will be served at:

`https://lupehiguera.github.io/iRacing_Ai_PitCrewChief/dashboard.html`

## Local preview

```bash
python -m http.server 8000 -d docs
# then open http://localhost:8000/dashboard.html
```

## Updating the data

After re-running the eval:

```bash
cp data/eval_comprehensive.json docs/eval_comprehensive.json
```

## Embedding in higuera.io

```html
<iframe src="https://lupehiguera.github.io/iRacing_Ai_PitCrewChief/dashboard.html"
        style="width:100%;height:1200px;border:0;border-radius:12px"
        loading="lazy"
        title="iRacing AI Pit Crew Chief — eval dashboard">
</iframe>
```

Or use the React component at `docs/portfolio-component.tsx` for a native React/Next.js embed.

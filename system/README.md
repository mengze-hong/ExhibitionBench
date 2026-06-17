# system/

Interactive demo application built with **NiceGUI** — a browser-based UI for exploring ExhibitionBench in real time.

## Features

- **Three tabs**: MEIP (item completion), ECD (coherence detection), TES (theme retrieval)
- Real benchmark samples loaded from `data/`
- Live inference via any OpenAI-compatible endpoint
- BM25 pre-built index for instant retrieval comparison
- Curator feedback logging (`logs/feedback_nicegui.jsonl`) for future annotation collection
- Human evaluation interface with Likert-scale ratings

## Setup

```bash
pip install nicegui rank_bm25 openai

# Set API credentials
export LLM_API_BASE="https://api.openai.com"
export LLM_API_KEY="sk-..."

# Run demo (default port 7861)
python system/nicegui_app.py

# Custom port
python system/nicegui_app.py --port 8080
```

Open `http://localhost:7861` in your browser.

## Deployment Scenarios

### Tourism Guide (latency-bound)
Configure with a fast model (DeepSeek-V3, Doubao-Lite) to achieve P95 latency under 5 s.
Suitable for on-device visitor-facing kiosk applications.

### Curation Back-Office (quality-bound)
Configure with a frontier model (Gemini 3.1 Pro, Claude Opus 4.6) for async batch curation.
Curator reviews top-3 model suggestions before finalising exhibition programmes.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_API_BASE` | `http://localhost:4000` | API base URL |
| `LLM_API_KEY` | (required) | API key |
| `EXHIBITIONBENCH_MODEL` | `gpt-4o` | Model name for inference |

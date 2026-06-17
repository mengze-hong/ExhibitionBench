# ExhibitionBench

A multi-task LLM benchmark for museum exhibition curation, built from 23,658 real objects across 5 open-access museum collections.

---

## Tasks

| Task | Description | Metric | Size |
|---|---|---|---|
| **MEIP** — Museum Exhibition Item Prediction | Given a theme + context objects, pick the best-fitting candidate from 10 options | MRR, Hit@1 | 1,409 queries |
| **TES** — Thematic Exhibition Selection | Rank 50 candidate exhibitions by thematic relevance | NDCG@10, MRR | 283 queries |
| **ECD** — Exhibition Coherence Discrimination | Identify the coherent sequence from a pair (4 difficulty levels) | PairAcc, Macro | 500 pairs |

---

## Key Results

| Model | MEIP MRR | TES NDCG@10 | ECD Macro | Latency (s) | Cost ($/1k) |
|---|---|---|---|---|---|
| Gemini 3.1 Pro | **0.685** | 0.408 | **0.876** | 13.3 | 12.55 |
| Claude Opus 4.6 | 0.621 | **0.437** | 0.836 | 7.1 | 28.75 |
| GPT-5.2 | 0.619 | 0.387 | 0.774 | 7.0 | 2.01 |
| Doubao-Seed-2.0-Pro | 0.642 | 0.410 | 0.852 | 27.2 | 3.93 |
| DeepSeek-V3 | 0.598 | 0.390 | 0.800 | **5.5** | **0.33** |
| Qwen2.5-72B (open-weight) | 0.733 | 0.391 | 0.674 | 8.4 | 0.69 |
| BM25 | 0.449 | 0.347 | 0.864 | <0.1 | ~0 |

Full results: [`results_summary/deployment_summary.json`](results_summary/deployment_summary.json)

---

## Repository Structure

```
ExhibitionBench/
├── data/
│   ├── meip_samples.jsonl        # 1,409 MEIP queries
│   ├── tes_samples.jsonl         # 283 TES queries
│   ├── ecd_samples.jsonl         # 500 ECD pairs (800 samples, 4 levels)
│   ├── objects.jsonl             # 23,658 museum objects
│   ├── exhibitions.jsonl         # 300 exhibition records
│   └── kg.json                   # CIDOC-CRM knowledge-graph triples
│
├── evaluation/
│   ├── sota_eval.py              # Evaluate any model via OpenAI-compatible API
│   ├── openllm_baseline.py       # Lightweight evaluator for open-weight models
│   ├── meip_eval.py              # MEIP metric computation
│   ├── tes_eval.py               # TES metric computation
│   └── ecd_generator.py          # ECD sample generation utilities
│
├── baselines/
│   ├── bm25_baseline.py          # BM25 term-overlap ranking
│   ├── embedding_baseline.py     # SBERT cosine-similarity ranking
│   └── rag_kg_baseline.py        # RAG + CIDOC-CRM KG triples
│
├── analysis/
│   ├── contamination_ablation.py # Dataset contamination check
│   ├── cultural_bias.py          # Per-region accuracy breakdown
│   ├── cultural_bias_multi_model.py
│   ├── error_analysis.py         # Error taxonomy
│   ├── fewshot_mechanism.py      # 0/1/3-shot mechanism analysis
│   └── metadata_ablation.py      # Object metadata sensitivity (L0-L5)
│
├── system/
│   └── nicegui_app.py            # Interactive demo (NiceGUI, port 7861)
│
├── scripts/
│   ├── run_pipeline.py           # End-to-end pipeline orchestration
│   ├── build_samples.py          # Rebuild benchmark from raw data
│   └── compile_results.py        # Aggregate results to tables / LaTeX
│
├── results/
│   ├── main_table/               # Zero-shot results, all 30 models × 3 tasks
│   ├── fewshot/                  # Shot 1/3 results, all models × 3 tasks
│   ├── fewshot_analysis/         # Few-shot mechanism analysis outputs
│   ├── ablation_cot/             # CoT prompting ablation
│   ├── ablation_vision/          # Multimodal (text+image) ablation
│   ├── metadata_ablation/        # Metadata sensitivity outputs
│   ├── contamination/            # Contamination check outputs
│   ├── cultural_bias/            # Cultural bias per-region outputs
│   ├── baselines_pred/           # BM25 / SBERT / RAG prediction files
│   ├── manifests/                # Experiment completeness reports
│   └── tables/                   # LaTeX / CSV summary tables
│
├── results_summary/
│   └── deployment_summary.json   # Per-model latency, cost, accuracy summary
│
├── .env.example                  # API credential template
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/ExhibitionBench
cd ExhibitionBench
pip install -r requirements.txt
cp .env.example .env   # fill in your API base + key
```

Supports any OpenAI-compatible endpoint: OpenAI, Anthropic (via proxy), Groq, Together AI, local Ollama, local vLLM.

---

## Evaluation

### Proprietary models

```bash
source .env
python evaluation/sota_eval.py --task all --model gpt-4o --workers 50 --save-raw
```

### Open-weight models (Ollama / vLLM / Groq / Together)

```bash
python evaluation/openllm_baseline.py \
    --api-base https://api.groq.com/openai/v1 \
    --api-key $GROQ_API_KEY \
    --model llama-3.3-70b-versatile \
    --tasks meip tes ecd
```

### Non-LLM baselines

```bash
python baselines/bm25_baseline.py meip --input data/meip_samples.jsonl
python baselines/embedding_baseline.py --task meip
```

### Compile results table

```bash
python scripts/compile_results.py --shot 0 --latex
```

---

## Interactive Demo

```bash
source .env
python system/nicegui_app.py --port 7861
# Open http://localhost:7861
```

Three task tabs (MEIP / ECD / TES), real benchmark samples, live inference, feedback logging.

---

## Data Sources

| Source | License | Objects |
|---|---|---|
| Metropolitan Museum of Art | CC0 1.0 | ~8,200 |
| Art Institute of Chicago | CC0 1.0 | ~5,400 |
| Victoria and Albert Museum | CC BY 4.0 | ~3,900 |
| Cleveland Museum of Art | CC0 1.0 | ~3,800 |
| Smithsonian Institution | CC0 / CC BY | ~2,358 |

Benchmark data: **CC BY 4.0**. Code: **MIT**.

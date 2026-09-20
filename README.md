<div align="center">

<img src="assets/figures/exhibition-emnlp-2026-banner.png" width="100%" alt="ExhibitionBench accepted at EMNLP 2026 Industry Track" />

<h1 align="center">ExhibitionBench: A Multi-Source, Multi-Task Benchmark for Evaluating LLMs as Exhibition Curation Assistants</h1>

**News:** ExhibitionBench is accepted at **EMNLP 2026 Industry Track**

</div>

---

## Overview

A multi-task LLM benchmark for museum exhibition curation, built from 23,658 real objects across five open-access museum collections.

ExhibitionBench evaluates whether models can complete an exhibition, retrieve
thematically relevant exhibitions, and detect curatorial incoherence. The
repository includes the released benchmark data, model evaluators, retrieval
baselines, analysis scripts, experiment outputs, and an interactive demo.

## Quick Start

```bash
git clone https://github.com/mengze-hong/ExhibitionBench.git
cd ExhibitionBench
python -m pip install -r requirements.txt

# Read-only validation of all released benchmark files
python scripts/validate_data.py
```

The validation command checks JSONL parsing, IDs, candidate counts, gold
references, and ECD sequence structure without modifying any data.

---

## Tasks

| Task | Description | Metric | Size |
|---|---|---|---|
| **MEIP** — Museum Exhibition Item Prediction | Given a theme + context objects, pick the best-fitting candidate from 10 options | MRR, Hit@1 | 1,409 queries |
| **TES** — Thematic Exhibition Selection | Rank 50 candidate exhibitions by thematic relevance | NDCG@10, MRR | 283 queries |
| **ECD** — Exhibition Coherence Discrimination | Identify the coherent sequence from a pair (4 difficulty levels) | PairAcc, Macro | 500 pairs |

---

## Repository Structure

```
ExhibitionBench/
├── data/
│   ├── meip_samples.jsonl        # 1,409 MEIP queries
│   ├── tes_samples.jsonl         # 283 TES queries
│   ├── ecd_samples.jsonl         # 500 ECD pairs (4 levels)
│   ├── objects.jsonl             # 23,658 museum objects
│   ├── exhibitions.jsonl         # 300 exhibition records
│   └── kg.json                   # CIDOC-CRM knowledge-graph triples
│
├── evaluation/
│   ├── sota_eval.py              # Canonical MEIP / TES / ECD evaluator
│   ├── openllm_baseline.py       # Lightweight evaluator for open-weight models
│   ├── meip_eval.py              # MEIP metric computation
│   └── ecd_generator.py          # ECD sample generation utilities
│
├── baselines/
│   ├── data_utils.py             # Shared schema compatibility helpers
│   ├── bm25_baseline.py          # BM25 term-overlap ranking
│   ├── embedding_baseline.py     # SBERT cosine-similarity ranking
│   ├── ecd_baseline.py           # ECD BM25/random baselines (500 pairs)
│   └── rag_kg_baseline.py        # RAG + CIDOC-CRM KG triples
│
├── analysis/
│   ├── contamination_ablation.py # Dataset contamination check
│   ├── cultural_bias.py          # MEIP per-region accuracy breakdown
│   ├── cultural_bias_multi_model.py
│   ├── error_analysis.py         # Error taxonomy
│   ├── fewshot_mechanism.py      # 0/1/3-shot mechanism analysis
│   └── metadata_ablation.py      # Object metadata sensitivity (L0-L5)
│
├── system/
│   └── nicegui_app.py            # Interactive demo (NiceGUI, port 7861)
│
├── scripts/
│   ├── validate_data.py          # Read-only benchmark integrity checks
│   ├── run_pipeline.py           # End-to-end pipeline orchestration
│   └── compile_results.py        # Aggregate results to tables / LaTeX
│
├── results/
│   ├── fewshot_analysis/         # Few-shot mechanism analysis outputs
│   ├── ablation_cot/             # CoT prompting ablation
│   ├── ablation_vision/          # Multimodal (text+image) ablation
│   ├── metadata_ablation/        # Metadata sensitivity outputs
│   ├── contamination/            # Contamination check outputs
│   ├── cultural_bias/            # Cultural bias per-region outputs
│   ├── baselines_pred/           # BM25 / SBERT / RAG prediction files
│   ├── tables/                   # LaTeX / CSV summary tables
│   └── deployment_summary.json   # Per-model latency, cost, accuracy summary
│
├── .env.example                  # API credential template
└── requirements.txt
```

---

## Environment Configuration

```bash
cp .env.example .env
# Edit .env and provide the endpoint and key used by your selected model.
source .env
```

The evaluators support OpenAI-compatible endpoints, including hosted gateways,
Groq, Together AI, local Ollama, and local vLLM. The primary and open-weight
endpoints are configured independently; only the endpoint used by the selected
model requires credentials.

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
| Metropolitan Museum of Art | CC0 1.0 | 1,221 |
| Art Institute of Chicago | CC0 1.0 | 7,270 |
| Victoria and Albert Museum | CC BY 4.0 | 4,545 |
| Cleveland Museum of Art | CC0 1.0 | 4,339 |
| Europeana | Source-specific rights statements | 6,283 |

Benchmark data: **CC BY 4.0**. Code: **MIT**.

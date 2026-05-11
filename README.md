# ExhibitionBench

A benchmark for evaluating the cultural-heritage understanding of Large Language Models in the museum-exhibition domain. ExhibitionBench probes whether LLMs (and their multimodal variants) can reason about real museum collections — recognising exhibits, recovering thematic exhibition structure, and ordering artefacts along a curatorial timeline.

The benchmark is built from public collection APIs of the **Art Institute of Chicago (AIC)**, the **Cleveland Museum of Art (CMA)**, and the **Victoria and Albert Museum (V&A)**, and supports zero-shot, few-shot, and multimodal evaluation across 20+ frontier closed- and open-source LLMs.

---

## Tasks

| ID | Task | Description | Metric |
|----|------|-------------|--------|
| **MEIP** | Museum Exhibit Identification & Placement | Given an exhibition theme + metadata, identify the correct exhibit from a candidate set and place it in the right gallery | MRR, Hit@1 |
| **TES** | Thematic Exhibition Structuring | Cluster a pool of exhibits into coherent thematic exhibitions | nDCG@10 |
| **ECD** | Exhibition Chronological Decision | Pairwise comparison of exhibit creation order under a given theme | macro pair-accuracy |

---

## Repository layout

```
ExhibitionBench/
├── baselines/        # All evaluation scripts (closed-source, open-source, multimodal, retrieval)
├── benchmark/        # Dataset construction & task-specific evaluators
├── analysis/         # Error analysis, cultural-bias study, ablations, contamination check
├── data/             # Curated benchmark JSONL files (large raw dumps are git-ignored)
├── results/          # Per-model score JSONs and aggregated tables
├── system/           # Gradio demo
├── paper/            # LaTeX sources and bibliography
├── scripts/          # Convenience runners
├── collect_*.py      # Multi-museum data-collection pipelines
├── requirements.txt
└── README.md
```

A more detailed walkthrough of every file lives in [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md), and a project hand-over note is in [`HANDOVER.md`](HANDOVER.md).

---

## Installation

```bash
git clone https://github.com/mengze-hong/ExhibitionBench.git
cd ExhibitionBench

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional, for JS-rendered scrapes
playwright install chromium
```

Tested with Python ≥ 3.10, PyTorch ≥ 2.2, transformers ≥ 4.40.

---

## Data

The curated benchmark splits are JSONL files under `data/`:

| File | #samples | Purpose |
|------|---------|---------|
| `meip_samples_v3_fixed.jsonl` | 1,409 | MEIP (canonical version — use this) |
| `tes_samples_v3.jsonl`        |   283 | TES |
| `ecd_samples_v3.jsonl`        |   800 | ECD |
| `exhibitions_v3.jsonl`        |     – | Exhibition metadata |
| `objects_v3.jsonl`            | 23,658 | Exhibit metadata pool |
| `kg.json`                     |     – | Cultural knowledge graph |

> Large raw API dumps (`data/raw/`, per-museum `*_objects.jsonl`, superseded `_v1`/`_v2` files) are **excluded from the repo** via [`.gitignore`](.gitignore) — regenerate them with the `collect_*.py` scripts if needed. Local model checkpoints in `models/` are also excluded; download from the official release page of each model.

To rebuild the dataset from scratch:

```bash
# 1. Scrape multi-museum collections
python collect_multi_source.py
python collect_expand_v3.py        # expand to v3 schema

# 2. Build task samples
python benchmark/build_samples.py
python benchmark/rebuild_samples.py
python benchmark/ecd_generator.py
python benchmark/fix_met_meip.py   # produces *_v3_fixed.jsonl
```

---

## Running evaluations

### Closed-source frontier models (LiteLLM proxy)

```bash
# All three tasks, two models
python baselines/sota_eval.py \
    --models gpt-5.2 claude-opus-4.6 \
    --tasks  meip tes ecd

# Single task with high concurrency
python baselines/sota_eval.py --models gpt-5.2 --tasks meip --workers 150

# Sweep every configured model
python baselines/sota_eval.py --models all --tasks meip
```

Supported models include: GPT-5 / 5.1 / 5.2 / 5-mini, Claude Opus 4.5–4.6 + Sonnet 4.5, Gemini 2.5 Pro/Flash + 3 Pro/Flash preview, Doubao Seed 1.6 / 2.0-Pro, DeepSeek V3.2 / R1, Kimi K2.5, MiniMax M2.5, GLM-5, Qwen-Plus.

Output: `results/{task}_{model}_shot{N}.json` — keys: `mrr`, `hit@1` (MEIP); `ndcg@10` (TES); `macro_pairaccc` (ECD).

### Multimodal evaluation (image-grounded MEIP)

```bash
python baselines/multimodal_eval.py --models all --workers 150
```

Supported vision models: `gpt-5.2`, `claude-opus-4.6`, `gemini-2.5-pro`, `gemini-2.5-flash`, `doubao-seed-1.6-vision-250815`. Output suffix: `_vision_shot0.json`.

### Open-source models (vLLM / Groq / Ollama)

```bash
python baselines/openllm_baseline.py --models Qwen2.5-7B-Instruct Llama-3.1-8B-Instruct
```

### Few-shot ablation

```bash
python baselines/gpt_fewshot.py --shots 0 1 3 5 --models gemini-2.5-flash
```

### Retrieval baselines

```bash
python baselines/bm25_baseline.py
python baselines/embedding_baseline.py
python baselines/rag_kg_baseline.py
```

---

## Analysis

```bash
python analysis/error_analysis.py
python analysis/cultural_bias_multi_model.py
python analysis/metadata_ablation.py
python analysis/contamination_ablation.py
python analysis/tes_leakage_analysis.py
python analysis/fewshot_mechanism.py
python analysis/summarize_analysis.py
```

---

## Headline results (zero-shot, MEIP / `_v3_fixed`)

| Model | MRR | Hit@1 |
|-------|----:|------:|
| doubao-seed-2.0-pro | 0.642 | 0.539 |
| claude-opus-4.6     | 0.621 | 0.510 |
| gpt-5.2             | 0.619 | 0.510 |
| gemini-2.5-pro      | 0.615 | 0.503 |
| gpt-5               | 0.599 | 0.483 |
| deepseek-v3.2       | 0.594 | 0.476 |
| claude-sonnet-4.5   | 0.571 | 0.453 |
| gemini-2.5-flash    | 0.552 | 0.429 |
| deepseek-r1         | 0.529 | 0.391 |
| kimi-k2.5           | 0.506 | 0.369 |
| minimax-m2.5        | 0.469 | 0.319 |
| glm-5               | 0.392 | 0.227 |

Full main tables in `results/sota_main_table_shot0.csv` and `results/latex_main_table_shot0.tex`.

---

## Demo

```bash
python system/app.py    # Gradio UI
```

---

## Citation

If you use ExhibitionBench in your research, please cite:

```bibtex
@misc{exhibitionbench2026,
  title  = {ExhibitionBench: Probing Cultural-Heritage Understanding in Large Language Models},
  author = {Hong, Mengze and others},
  year   = {2026},
  url    = {https://github.com/mengze-hong/ExhibitionBench}
}
```

---

## License

Code is released under the MIT License (see `LICENSE`).
Museum metadata and images are subject to the licensing terms of the respective institutions (AIC, CMA, V&A) — please consult each museum's API terms before redistribution.

---

## Acknowledgements

We thank the open data programmes of the Art Institute of Chicago, the Cleveland Museum of Art, and the Victoria and Albert Museum, whose public collection APIs make this benchmark possible.

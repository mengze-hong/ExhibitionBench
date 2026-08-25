# results/

Pre-computed predictions, evaluation outputs, analysis artifacts, and summary
statistics from ExhibitionBench experiments. Model evaluation JSON files at
the root of this directory are the canonical inputs consumed by
`scripts/compile_results.py`.

## Organization

| Path | Description |
|---|---|
| Root `*.json` | Canonical per-model task results and compiled summaries |
| `main_table/` | Additional zero-shot artifacts used in the main comparison |
| `fewshot/` | Additional few-shot evaluation artifacts |
| `baselines_pred/` | BM25, embedding, and RAG baseline predictions |
| `ablation_cot/` | Chain-of-thought ablation outputs |
| `ablation_vision/` | Vision-input ablation outputs |
| `metadata_ablation/` | Metadata sensitivity results |
| `contamination/` | Contamination analysis outputs |
| `cultural_bias/` | Culture-wise analysis outputs |
| `tables/` | Camera-ready-aligned CSV and LaTeX summary tables |
| `deployment_summary.json` | Per-model quality, latency, token, and cost summary |

## Deployment Summary Format

Each model entry in `deployment_summary.json` reports the metrics used for
quality and deployment comparisons:

```json
{
  "gpt-5.2": {
    "avg_lat": 7.0,
    "total_tok": 1500,
    "cost_per_1k": 2.01,
    "meip_mrr": 0.619,
    "ecd_m": 0.774,
    "tes_ndcg": 0.387
  }
}
```

Raw per-sample traces produced with `--save-raw` are written to
`results/raw_responses/` and are excluded from Git by default.
Newly compiled tables are written to the ignored `results/tables/generated/`
directory, so routine validation never overwrites the frozen released tables.
For zero-shot MEIP, the compiler prefers the corrected full-set `v3fixed`
outputs used in the camera-ready paper. For TES, it prefers the anonymized
`noleak` outputs used in the paper.

# scripts/

Utility scripts for release validation and pipeline orchestration.

## Files

| Script | Purpose |
|---|---|
| `compile_results.py` | Aggregate per-model result JSONs into summary tables and LaTeX |
| `run_pipeline.py` | End-to-end pipeline: manifest check → quality control → missing backfill → recompile |
| `validate_data.py` | Read-only schema, reference, and candidate-integrity validation |

## Validate and Run the Released Benchmark

```bash
# Validate the release without modifying any data
python scripts/validate_data.py

# 1. Run all evaluations on the released benchmark files (replace with your model list)
python evaluation/sota_eval.py \
    --task all --model gpt-4o claude-3-5-sonnet-20241022 \
    --max-samples 500 --workers 50 --save-raw

# 2. Run non-LLM baselines
python baselines/bm25_baseline.py meip \
    --input data/meip_samples.jsonl --output results/baselines_pred/bm25_meip_pred.jsonl
python baselines/embedding_baseline.py meip \
    --input data/meip_samples.jsonl --output results/baselines_pred/sbert_meip_pred.jsonl

# 3. Compile result tables
python scripts/compile_results.py --shot 0 --latex

# 4. Run analysis
python analysis/cultural_bias_multi_model.py
python analysis/metadata_ablation.py
```

## run_pipeline.py — Automated Pipeline

```bash
# Validate released data and recompile all result tables
python scripts/run_pipeline.py --stage all

# Only recompile tables
python scripts/run_pipeline.py --stage recompile
```

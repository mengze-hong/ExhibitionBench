# results_summary/

Pre-computed summary statistics from the full evaluation run.

## Files

| File | Description |
|---|---|
| `deployment_summary.json` | Per-model latency, token usage, cost/1k decisions, MEIP MRR, ECD Macro, TES NDCG for 27 models |

## Format — deployment_summary.json

```json
{
  "gpt-5.2": {
    "avg_lat": 7.0,
    "total_tok": 1500,
    "cost_per_1k": 2.01,
    "meip_mrr": 0.619,
    "ecd_m": 0.774,
    "tes_ndcg": 0.387
  },
  ...
}
```

## Deployment Pareto

For the two main deployment scenarios:

**Tourism Guide (latency-first, P95 < 5 s target):**
- Best: DeepSeek-V3 — MRR 0.60, avg latency 5.5 s, cost $0.33/1k

**Curation Back-Office (quality-first, async):**
- Best: Gemini 3.1 Pro — MRR 0.685, ECD Macro 0.876, cost $12.55/1k
- Budget pick: Doubao-Seed-2.0-Pro — MRR 0.642, ECD 0.852, cost $3.93/1k

Full analysis in the companion paper.

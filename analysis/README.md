# analysis/

Post-hoc analysis scripts for ExhibitionBench experiments.

## Files

| Script | What it computes |
|---|---|
| `contamination_ablation.py` | Dataset contamination check: compares performance on pre-/post-training-cutoff exhibitions |
| `cultural_bias.py` | Per-region accuracy breakdown (East Asia, South Asia, Western Europe, Americas, etc.) |
| `cultural_bias_multi_model.py` | Cross-model cultural bias heatmap |
| `error_analysis.py` | Parse failure taxonomy (invalid output, wrong ID format, truncated response) |
| `fewshot_mechanism.py` | Few-shot ablation: 0-shot vs 1-shot vs 3-shot across tasks |
| `metadata_ablation.py` | Metadata sensitivity: L0 (title only) to L5 (full enriched object description) |

## Usage

```bash
# Contamination check (reads results/*.json automatically)
python analysis/contamination_ablation.py

# Cultural bias for a single model
python analysis/cultural_bias.py --model gpt-4o

# Cross-model bias comparison
python analysis/cultural_bias_multi_model.py

# Few-shot mechanism analysis
python analysis/fewshot_mechanism.py

# Metadata ablation (L0 to L5)
python analysis/metadata_ablation.py --model gpt-4o
```

## Key Findings

- **H1 Cultural Bias**: Western models show 8 to 12 pp gap between Western-European and East-Asian collections on MEIP.
- **H2 Metadata Sensitivity**: Adding cultural context (L2) yields +7 MRR; additional fields plateau after L3.
- **H3 Few-shot Effect**: 1-shot +2 MRR for mid-tier models; frontier models (Gemini 3.1 Pro) show no gain.
- **Contamination**: No significant performance delta on post-cutoff exhibitions, suggesting genuine curation reasoning rather than memorisation.

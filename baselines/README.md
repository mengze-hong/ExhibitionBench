# baselines/

Non-LLM retrieval baselines for ExhibitionBench.

## Files

| Script | Method | Tasks |
|---|---|---|
| `bm25_baseline.py` | BM25 term-overlap ranking | MEIP, TES |
| `embedding_baseline.py` | SBERT cosine-similarity ranking | MEIP, TES |
| `rag_kg_baseline.py` | RAG + CIDOC-CRM knowledge-graph triples | MEIP |

## Usage

```bash
# BM25
python baselines/bm25_baseline.py meip \
    --input data/meip_samples.jsonl \
    --output results/bm25_meip_pred.jsonl

python baselines/bm25_baseline.py tes \
    --input data/tes_samples.jsonl \
    --output results/bm25_tes_pred.jsonl

# SBERT (requires sentence-transformers)
python baselines/embedding_baseline.py \
    --task meip \
    --model all-MiniLM-L6-v2

# RAG+KG (requires OpenAI-compatible endpoint for GPT prompting)
python baselines/rag_kg_baseline.py \
    --task meip \
    --kg data/kg.json
```

## Results (zero-shot, full evaluation set)

| Baseline | MEIP MRR | MEIP Hit@1 | ECD Macro |
|---|---|---|---|
| Random | 0.287 | 0.098 | 0.514 |
| BM25 | 0.449 | 0.281 | 0.864 |
| SBERT | pending | pending | pending |
| RAG+KG | pending | pending | n/a |

BM25 surprisingly strong on ECD (L1/L2) due to high surface-form overlap in coherent sequences.

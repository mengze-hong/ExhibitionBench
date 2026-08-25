# baselines/

Non-LLM retrieval baselines for ExhibitionBench.

## Files

| Script | Method | Tasks |
|---|---|---|
| `bm25_baseline.py` | BM25 term-overlap ranking | MEIP, TES |
| `embedding_baseline.py` | SBERT cosine-similarity ranking | MEIP, TES |
| `rag_kg_baseline.py` | RAG + CIDOC-CRM knowledge-graph triples | MEIP |

The MEIP baselines accept both released schemas: embedded `candidates`
objects and ID-only `candidate_ids`. ID-only records are resolved through
`data/objects.jsonl`; use `--objects` to select another metadata file.

## Usage

```bash
# BM25
python baselines/bm25_baseline.py meip \
    --input data/meip_samples.jsonl \
    --output results/baselines_pred/bm25_meip_pred.jsonl

python baselines/bm25_baseline.py tes \
    --input data/tes_samples.jsonl \
    --output results/baselines_pred/bm25_tes_pred.jsonl

# SBERT (requires sentence-transformers)
python baselines/embedding_baseline.py meip \
    --input data/meip_samples.jsonl \
    --output results/baselines_pred/sbert_meip_pred.jsonl

# RAG+KG (requires OpenAI-compatible endpoint for GPT prompting)
python baselines/rag_kg_baseline.py meip \
    --input data/meip_samples.jsonl \
    --output results/baselines_pred/rag_kg_meip_pred.jsonl \
    --kg data/kg.json
```

## Camera-ready Results (zero-shot)

| Baseline | MEIP MRR | MEIP Hit@1 | TES NDCG@10 | TES MRR | ECD Macro |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.449 | 0.281 | 0.347 | 0.328 | 0.864 |
| SBERT | 0.859 | 0.780 | 0.282 | 0.255 | 0.588 |
| RAG+KG (GPT-5.2) | 0.911 | 0.853 | 0.282 | 0.255 | 0.593 |

Values are rounded exactly as in the camera-ready main table. BM25 is strong on
ECD L1/L2 because surface-form and metadata cues are informative at those levels.

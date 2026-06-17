# data/

Benchmark data for **ExhibitionBench** — three tasks derived from real museum collections.

## Files

| File | Size | Description |
|---|---|---|
| `objects.jsonl` | ~13 MB | 23,658 museum objects (id, title, culture, date, medium, description, source) |
| `exhibitions.jsonl` | ~700 KB | Exhibition metadata from 5 museum sources |
| `meip_samples.jsonl` | ~4.3 MB | 1,409 MEIP queries (10-way ranking, gold_id) |
| `tes_samples.jsonl` | ~9.5 MB | 283 TES queries (50-way ranking, gold_ids, leak-free anonymisation) |
| `ecd_samples.jsonl` | ~2.3 MB | 500 ECD pairs (4 difficulty levels L1–L4, 125 per level, positive + negative sequences) |
| `kg.json` | ~600 KB | Knowledge-graph triples (CIDOC-CRM) used by RAG+KG baseline |

## Sources

Objects were collected from five public museum APIs / open datasets:

- **Metropolitan Museum of Art** — Open Access collection (CC0)
- **Art Institute of Chicago** — Open Access API (CC0)  
- **Victoria and Albert Museum** — V&A API (CC BY)
- **Cleveland Museum of Art** — Open Access collection (CC0)
- **Smithsonian Institution** — Open Access (CC0 / CC BY)

## Format

Each MEIP sample:
```json
{
  "id": "meip_000001",
  "exhibition_theme": "Japanese Woodblock Prints",
  "context": ["met_36491", "met_45677", "aic_185432"],
  "candidates": [{"id": "met_12345", "title": "...", ...}, ...],
  "gold_id": "met_12345"
}
```

Each ECD sample:
```json
{
  "id": "ecd_000001",
  "level": 2,
  "positive": {"theme": "...", "items": [...]},
  "negative": {"theme": "...", "items": [...]}
}
```

# evaluation/

Core evaluation scripts for all three ExhibitionBench tasks.

`sota_eval.py` is the canonical evaluator and the source of truth for released
prompting, parsing, and metrics. Final-protocol TES ranks 50 anonymous
exhibition candidates (`EX_001`--`EX_050`) and returns the top 10. The earlier
object-set TES harness has been removed because it implements a different,
obsolete task.

## Files

| Script | Purpose |
|---|---|
| `sota_eval.py` | Canonical evaluator — runs MEIP / TES / ECD on any model via OpenAI-compatible API |
| `openllm_baseline.py` | Lightweight evaluator for open-weight models (Ollama, vLLM, Groq, Together AI) |
| `meip_eval.py` | Standalone MEIP metric computation (MRR, Hit@1) |
| `ecd_generator.py` | ECD sample generation utilities |

## Quick Start

### Proprietary models (GPT / Claude / Gemini)

```bash
# Set credentials (copy .env.example → .env first)
export LLM_API_BASE="https://api.openai.com"
export LLM_API_KEY="sk-..."

# Run all three tasks on GPT-4o
python evaluation/sota_eval.py --task all --model gpt-4o --max-samples 200

# Run only MEIP with zero-shot + CoT
python evaluation/sota_eval.py --task meip --model gpt-4o --cot
```

### Open-weight models (Ollama / vLLM / Groq / Together AI)

```bash
# Local Ollama
python evaluation/openllm_baseline.py \
    --api-base http://localhost:11434/v1 \
    --api-key ollama \
    --model llama3:70b \
    --tasks meip tes ecd

# Groq (free tier)
python evaluation/openllm_baseline.py \
    --api-base https://api.groq.com/openai/v1 \
    --api-key $GROQ_API_KEY \
    --model llama-3.3-70b-versatile \
    --tasks meip tes ecd

# Together AI
python evaluation/openllm_baseline.py \
    --api-base https://api.together.xyz/v1 \
    --api-key $TOGETHER_API_KEY \
    --model meta-llama/Llama-3.1-8B-Instruct-Turbo \
    --tasks meip tes ecd
```

## Output

Results are saved to `results/{task}_{model}_shot{n}.json`:

```json
{
  "task": "meip",
  "model": "gpt-4o",
  "shot": 0,
  "n_samples": 500,
  "mrr": 0.612,
  "hit@1": 0.491,
  "avg_latency_sec": 4.2,
  "total_tokens": 812000
}
```

## CLI Reference — sota_eval.py

```
--task       meip | tes | ecd | all
--model      model name (see MODELS dict) or 'all'
--max-samples  limit samples per task (default: full set)
--shot       few-shot examples: 0=zero-shot, 1, 3 (default: 0)
--cot        enable Chain-of-Thought prompting
--workers    concurrent threads (default: 100; reduce for rate-limited endpoints)
--force      re-run even if result file exists
--resume     resume from partial raw_responses JSONL
--save-raw   save per-sample traces to results/raw_responses/
--summary    print aggregated table of existing results
```

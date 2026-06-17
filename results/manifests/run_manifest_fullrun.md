# ExhibitionBench Fullrun Manifest

- models: 20
- tasks: 3 (meip/tes/ecd)
- shots: [0, 1, 3]
- matrix size: 180

## Frozen Models
- gpt-5.2
- gpt-5
- gpt-5.1
- claude-opus-4.6
- claude-opus-4.5
- claude-sonnet-4.5
- gemini-2.5-pro
- gemini-2.5-flash
- gemini-3-pro-preview
- gemini-3-flash-preview
- gemini-3.1-pro-preview
- deepseek-r1
- deepseek-v3.2
- deepseek-v3
- doubao-seed-2.0-pro
- doubao-seed-2.0-lite
- doubao-seed-1.6
- kimi-k2.5
- glm-5
- minimax-m2.5

## Command Examples
- `python baselines/sota_eval.py --models gpt-5.2 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force`
- `python baselines/sota_eval.py --models gpt-5.2 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force`
- `python baselines/sota_eval.py --models gpt-5.2 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force`
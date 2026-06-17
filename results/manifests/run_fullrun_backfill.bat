@echo off
setlocal enabledelayedexpansion
python baselines/sota_eval.py --models gpt-5.2 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.2 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.1 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.1 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.1 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.6 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.6 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.6 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.5 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.5 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.5 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-sonnet-4.5 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-sonnet-4.5 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-sonnet-4.5 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-pro --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-pro --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-pro --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-flash --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-flash --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-flash --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-pro-preview --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-pro-preview --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-pro-preview --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-flash-preview --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-flash-preview --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-flash-preview --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3.1-pro-preview --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3.1-pro-preview --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3.1-pro-preview --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-r1 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-r1 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-r1 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3.2 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3.2 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3.2 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-pro --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-pro --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-pro --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-lite --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-lite --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-lite --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-1.6 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-1.6 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-1.6 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models kimi-k2.5 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models kimi-k2.5 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models kimi-k2.5 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models glm-5 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models glm-5 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models glm-5 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models minimax-m2.5 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models minimax-m2.5 --tasks tes --shot 0 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models minimax-m2.5 --tasks ecd --shot 0 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.2 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.2 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.2 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.1 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.1 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.1 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.6 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.6 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.6 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.5 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.5 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.5 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-sonnet-4.5 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-sonnet-4.5 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-sonnet-4.5 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-pro --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-pro --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-pro --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-flash --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-flash --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-flash --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-pro-preview --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-pro-preview --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-pro-preview --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-flash-preview --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-flash-preview --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-flash-preview --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3.1-pro-preview --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3.1-pro-preview --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3.1-pro-preview --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-r1 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-r1 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-r1 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3.2 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3.2 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3.2 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-pro --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-pro --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-pro --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-lite --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-lite --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-lite --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-1.6 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-1.6 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-1.6 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models kimi-k2.5 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models kimi-k2.5 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models kimi-k2.5 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models glm-5 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models glm-5 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models glm-5 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models minimax-m2.5 --tasks meip --shot 1 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models minimax-m2.5 --tasks tes --shot 1 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models minimax-m2.5 --tasks ecd --shot 1 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.2 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.2 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.2 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.1 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.1 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.1 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.6 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.6 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.6 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.5 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.5 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-opus-4.5 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-sonnet-4.5 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-sonnet-4.5 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models claude-sonnet-4.5 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-pro --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-pro --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-pro --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-flash --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-flash --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-2.5-flash --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-pro-preview --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-pro-preview --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-pro-preview --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-flash-preview --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-flash-preview --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3-flash-preview --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3.1-pro-preview --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3.1-pro-preview --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gemini-3.1-pro-preview --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-r1 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-r1 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-r1 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3.2 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3.2 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3.2 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models deepseek-v3 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-pro --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-pro --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-pro --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-lite --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-lite --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-2.0-lite --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-1.6 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-1.6 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models doubao-seed-1.6 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models kimi-k2.5 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models kimi-k2.5 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models kimi-k2.5 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models glm-5 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models glm-5 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models glm-5 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models minimax-m2.5 --tasks meip --shot 3 --max-samples 1409 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models minimax-m2.5 --tasks tes --shot 3 --tes-noleak --max-samples 283 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models minimax-m2.5 --tasks ecd --shot 3 --max-samples 800 --tag fullrun --save-raw --force
python baselines/sota_eval.py --models gpt-5.2 --tasks meip --shot 0 --max-samples 1409 --tag fullrun --save-raw --force
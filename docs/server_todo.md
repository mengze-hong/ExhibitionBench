# ExhibitionBench — 服务器开源模型推理 TODO

> 创建时间：2026-05-02  
> 目标：在 GPU 服务器上部署 vLLM，完成 ExhibitionBench 三个任务（MEIP / TES / ECD）的开源模型 baseline 评测  
> 评测脚本：`baselines/openllm_baseline.py`（兼容 vLLM / Ollama / Together AI，结果格式与 sota_eval.py 完全一致）

---

## 一、优先候选模型列表

| 优先级 | 模型 | HuggingFace ID | 显存需求 | 备注 |
|--------|------|----------------|----------|------|
| P0 | **Qwen2.5-72B-Instruct** | `Qwen/Qwen2.5-72B-Instruct` | ~80GB (4×A100 或 2×H100) | 最强开源，对应闭源 doubao/glm |
| P0 | **Llama-3.3-70B-Instruct** | `meta-llama/Llama-3.3-70B-Instruct` | ~80GB | Meta 最新旗舰 |
| P0 | **Llama-3.1-8B-Instruct** | `meta-llama/Llama-3.1-8B-Instruct` | ~16GB | 轻量高效 baseline |
| P1 | **Qwen2.5-7B-Instruct** | `Qwen/Qwen2.5-7B-Instruct` | ~16GB | Qwen 小模型 |
| P1 | **Mistral-7B-Instruct-v0.3** | `mistralai/Mistral-7B-Instruct-v0.3` | ~16GB | 欧洲开源 baseline |
| P1 | **Mixtral-8x7B-Instruct** | `mistralai/Mixtral-8x7B-Instruct-v0.1` | ~90GB | MoE，可选 |
| P2 | **DeepSeek-R1-Distill-Qwen-14B** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | ~28GB | 注意：R1 推理链问题见 §4 |
| P2 | **Qwen2.5-32B-Instruct** | `Qwen/Qwen2.5-32B-Instruct` | ~40GB | 中等规模 |
| P3 | **gemma-2-27b-it** | `google/gemma-2-27b-it` | ~40GB | Google 开源 |
| P3 | **Yi-1.5-34B-Chat** | `01-ai/Yi-1.5-34B-Chat` | ~40GB | 中文理解能力强 |

> **最低优先级组合（快速跑完论文 baseline）**：Llama-3.1-8B + Llama-3.3-70B + Qwen2.5-72B

---

## 二、环境依赖

```bash
# Python 3.10+
pip install vllm openai tqdm

# 或用 conda
conda create -n exhibbench python=3.10
conda activate exhibbench
pip install vllm openai tqdm rank_bm25 sentence-transformers
```

---

## 三、启动 vLLM 服务

```bash
# 单卡（8B 模型）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --max-model-len 4096 \
    --dtype bfloat16

# 多卡张量并行（70B 模型，4卡）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000 \
    --max-model-len 4096 \
    --dtype bfloat16

# 72B Qwen（2卡 H100 或 4卡 A100）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000 \
    --max-model-len 4096 \
    --dtype bfloat16
```

---

## 四、评测命令

### 4.1 全量三任务评测（推荐）

```bash
cd /path/to/展览馆llm

# Llama-3.1-8B（本地 vLLM，workers=32 对应并发量）
python baselines/openllm_baseline.py \
    --api-base http://localhost:8000 \
    --api-key vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tasks meip tes ecd \
    --workers 32

# Llama-3.3-70B
python baselines/openllm_baseline.py \
    --api-base http://localhost:8000 \
    --api-key vllm \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --tasks meip tes ecd \
    --workers 32

# Qwen2.5-72B
python baselines/openllm_baseline.py \
    --api-base http://localhost:8000 \
    --api-key vllm \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tasks meip tes ecd \
    --workers 32

# Qwen2.5-7B
python baselines/openllm_baseline.py \
    --api-base http://localhost:8000 \
    --api-key vllm \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tasks meip tes ecd \
    --workers 32

# Mistral-7B
python baselines/openllm_baseline.py \
    --api-base http://localhost:8000 \
    --api-key vllm \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --tasks meip tes ecd \
    --workers 32
```

### 4.2 快速冒烟测试（每个任务 50 样本）

```bash
python baselines/openllm_baseline.py \
    --api-base http://localhost:8000 \
    --api-key vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tasks meip tes ecd \
    --max-samples 50 \
    --workers 8
```

### 4.3 备用：Groq 免费 API（无服务器时）

```bash
pip install groq

python baselines/openllm_baseline.py \
    --api-base https://api.groq.com/openai/v1 \
    --api-key $GROQ_API_KEY \
    --model llama-3.3-70b-versatile \
    --tasks meip tes ecd \
    --workers 10
```

### 4.4 ⚠️ 推理链模型（R1 系列）特殊处理

DeepSeek-R1 等推理链模型会输出很长的 `<think>...</think>` 块，导致 ECD 解析失败（macro_pairaccc ≈ 0.49 随机）。  
**临时方案**：运行时只跑 MEIP 和 TES，跳过 ECD：

```bash
python baselines/openllm_baseline.py \
    --api-base http://localhost:8000 \
    --api-key vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --tasks meip tes \
    --workers 16
```

---

## 五、需要上传的文件清单

### 5.1 核心代码

```
baselines/
├── openllm_baseline.py   ← 主评测脚本（支持 vLLM/Groq/Ollama）
├── sota_eval.py          ← 参考（可选，LiteLLM 专用）
└── multimodal_eval.py    ← 视觉评测（可选，需视觉模型）

results/
└── compile_sota_results.py  ← 汇总结果用
```

### 5.2 数据文件（必须上传）

```
data/
├── meip_samples_v3_fixed.jsonl   ← MEIP 评测集（1409 样本，4.3MB）⚠️ 用这个！
├── tes_samples_v3.jsonl          ← TES 评测集（283 样本，9.7MB）
├── ecd_samples_v3.jsonl          ← ECD 评测集（800 样本，2.3MB）
└── objects_v3.jsonl              ← 展品数据库（23658 条，13MB）
```

> ⚠️ **注意**：`find_data_file()` 自动选 `_v3 > _v2 > bare` 版本。  
> MEIP 请确保服务器上存在 `data/meip_samples_v3_fixed.jsonl`（不是 `meip_samples_v3.jsonl`）。  
> 或者手动在 `openllm_baseline.py` 的 `find_data_file()` 中加 `_v3_fixed` 优先级。

### 5.3 不需要上传

```
data/raw/               ← 原始爬取数据，太大
data/*_v2.jsonl         ← 旧版本
data/objects.jsonl      ← 旧版本
results/                ← 服务器跑完后回传
logs/                   ← 不需要
system/                 ← Gradio demo，不需要
```

---

## 六、结果文件命名规范

`openllm_baseline.py` 输出到 `results/` 目录，命名格式：

```
results/{task}_{model_slug}_shot0.json
```

例如：
- `results/meip_meta-llama_Llama-3_3-70B-Instruct_shot0.json`
- `results/tes_Qwen_Qwen2_5-72B-Instruct_shot0.json`
- `results/ecd_mistralai_Mistral-7B-Instruct-v0_3_shot0.json`

跑完后把 `results/` 目录打包回传：

```bash
tar -czf openllm_results_$(date +%Y%m%d).tar.gz results/*_shot0.json
```

---

## 七、汇总结果

回传到本机后，运行（或等修复键名 bug 后再跑）：

```bash
cd "C:\Users\mengzehong\Desktop\展览馆llm"
python results/compile_sota_results.py
```

> ⚠️ **已知 bug**：`compile_results.py` 键名不匹配（TES 应读 `ndcg@10`，ECD 应读 `macro_pairaccc`）  
> 修复前可手动读 JSON 查看结果：  
> ```bash
> python -c "import json; d=json.load(open('results/meip_XXX_shot0.json')); print(d)"
> ```

---

## 八、预期评测时间估算

| 模型规模 | 任务 | 样本数 | vLLM TPS | 预计时间 |
|----------|------|--------|----------|----------|
| 8B | MEIP | 1409 | ~150 tok/s | ~20 min |
| 8B | TES | 283 | ~150 tok/s | ~5 min |
| 8B | ECD | 800 | ~150 tok/s | ~10 min |
| 70B | MEIP | 1409 | ~40 tok/s | ~60 min |
| 70B | TES | 283 | ~40 tok/s | ~15 min |
| 70B | ECD | 800 | ~40 tok/s | ~30 min |

> 以 `--workers 32` 并发为基础估算，实际取决于 GPU 型号和 batch size。

---

## 九、检查清单（跑之前）

- [ ] vLLM 服务已启动，`curl http://localhost:8000/v1/models` 可访问
- [ ] 四个数据文件已上传到 `data/` 目录
- [ ] Python 依赖已安装（`openai`, `vllm`, `tqdm`）
- [ ] `nohup` 跑后台，日志重定向：`nohup python ... > logs/run.log 2>&1 &`
- [ ] 每个模型跑完后确认 `results/` 下有对应 JSON 文件

---

_自动生成 by Claude — ExhibitionBench 项目，2026-05-02_

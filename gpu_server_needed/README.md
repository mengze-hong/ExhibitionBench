# GPU Server Setup Guide — ExhibitionBench Open-Weight Eval

This folder contains everything you need to run open-weight LLM evaluations on a GPU server.

---

## 硬件要求

| 模型规模 | 最低配置 | 推荐配置 |
|---------|---------|---------|
| 7B/8B   | 1×A100 40GB | 1×A100 80GB |
| 70B/72B | 4×A100 80GB | 2×H100 80GB |

> 所有模型使用 bfloat16，max_model_len=4096。

---

## 模型优先级

### P0（必跑）
| 模型 | HuggingFace ID | TP | 需要 HF_TOKEN |
|------|---------------|----|--------------|
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | 1 | ✅ 需要 |
| Llama-3.3-70B | `meta-llama/Llama-3.3-70B-Instruct` | 4 | ✅ 需要 |
| Qwen2.5-72B | `Qwen/Qwen2.5-72B-Instruct` | 4 | ❌ 不需要 |

### P1（可选）
| 模型 | HuggingFace ID | TP | 说明 |
|------|---------------|----|------|
| Qwen3-8B | `Qwen/Qwen3-8B` | 1 | 需要 `transformers>=4.51` |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | 1 | 已有结果，可跳过 |

---

## Step-by-Step 部署指南

### Step 1: 克隆代码库

```bash
git clone https://github.com/mengze-hong/ExhibitionBench.git
cd ExhibitionBench
```

### Step 2: 安装依赖

```bash
# 安装 GPU 服务器专用依赖
pip install -r gpu_server_needed/requirements_gpu.txt

# 验证 vLLM 安装
python -c "import vllm; print(vllm.__version__)"
```

> ⚠️ 如果跑 Qwen3-8B，需要：`pip install transformers>=4.51`

### Step 3: 上传数据文件

将以下文件上传到服务器（**必须**）：

```
data/
├── meip_samples_v3_fixed.jsonl   ← MEIP 评测数据（必须用 fixed 版本！）
├── tes_samples_v3.jsonl           ← TES 评测数据
├── ecd_samples_v3.jsonl           ← ECD 评测数据
├── objects.jsonl                  ← 展品元数据（TES 需要）
└── exhibitions.jsonl              ← 展览数据
```

**上传命令示例（从本地机器）**：
```bash
scp -r data/ user@gpu-server:/path/to/ExhibitionBench/
```

> ⚠️ **重要**：必须用 `meip_samples_v3_fixed.jsonl`，而非 `meip_samples_v3.jsonl`。
> 旧版本中有数据污染问题（query_theme 泄露候选集），fixed 版本已修复。

### Step 4: 设置环境变量

```bash
# Meta Llama 模型需要 HuggingFace token（需申请 Llama 访问权限）
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# （可选）加速 HuggingFace 下载
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install hf_transfer
```

申请 Llama 访问权限：https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

### Step 5: 运行评测

**方式一：一键全跑（推荐）**

```bash
cd ExhibitionBench

# 完整评测（全量，约 2-4 小时）
bash gpu_server_needed/run.sh

# 快速冒烟测试（每任务 50 样本，约 10 分钟）
bash gpu_server_needed/run.sh smoke
```

**方式二：单独跑某个模型**

```bash
# 先启动 vLLM 服务
bash scripts/start_vllm.sh start Qwen/Qwen2.5-72B-Instruct 4

# 等待服务就绪后，运行评测
python baselines/openllm_baseline.py \
    --api-base http://localhost:8000/v1 \
    --model-slug qwen2.5-72b \
    --tasks meip tes ecd \
    --shot 0

# 停止 vLLM 服务
bash scripts/start_vllm.sh stop
```

**方式三：使用完整流水线脚本**

```bash
# 用法: bash scripts/run_openllm_eval.sh <model_id> <slug> [tp_size] [max_samples]
bash scripts/run_openllm_eval.sh meta-llama/Llama-3.1-8B-Instruct llama-3.1-8b 1
bash scripts/run_openllm_eval.sh Qwen/Qwen2.5-72B-Instruct qwen2.5-72b 4
bash scripts/run_openllm_eval.sh meta-llama/Llama-3.3-70B-Instruct llama-3.3-70b 4
```

---

## 注意事项

### ⚠️ 推理链模型跳过 ECD
R1 类推理模型（chain-of-thought）在 ECD 任务上因长推理链导致解析失败，**跳过 ECD**：

```bash
python baselines/openllm_baseline.py \
    --api-base http://localhost:8000/v1 \
    --model-slug <slug> \
    --tasks meip tes   # 故意不包含 ecd
```

### ⚠️ 磁盘空间
- 8B 模型：~16GB
- 70B/72B 模型：~80-140GB（取决于量化）
- 确保有 200GB+ 可用磁盘

### ⚠️ 显存需求
- 不要使用 `--quantization awq/gptq` 除非 80GB×4 也不够，量化会影响结果可比性

---

## 结果回传

评测完成后，打包 results/ 目录：

```bash
cd ExhibitionBench

# 打包所有结果（包括日志）
tar -czf exhib_results_$(hostname)_$(date +%Y%m%d).tar.gz \
    results/*_shot0.json \
    logs/vllm_*.log \
    logs/openllm_*.log

# 查看打包内容
tar -tzf exhib_results_*.tar.gz | head -30
```

**命名规范**：
- 结果文件自动命名为 `results/{task}_{slug}_shot0.json`
- 示例：`results/meip_llama-3.1-8b_shot0.json`

**回传命令**：
```bash
scp exhib_results_*.tar.gz user@local-machine:/path/to/results/
```

---

## 验证结果

回传后，在本地运行：
```bash
cd ExhibitionBench
python results/compile_sota_results.py --latex
```

检查输出中是否有新模型的数据（之前显示 `—` 的行应该有数值了）。

---

## 文件清单（本文件夹）

```
gpu_server_needed/
├── README.md               ← 本文件
├── requirements_gpu.txt    ← GPU 服务器 Python 依赖
└── run.sh                  ← 一键运行脚本（调用 scripts/run_all_openllm.sh）
```

关联脚本（在主目录 scripts/）：
- `scripts/start_vllm.sh` — vLLM 服务管理
- `scripts/run_openllm_eval.sh` — 单模型评测流水线
- `scripts/run_all_openllm.sh` — 批量跑全部模型
- `baselines/openllm_baseline.py` — 评测主脚本

---

## 快速排错

| 问题 | 原因 | 解决 |
|------|------|------|
| `CUDA out of memory` | TP 太小 | 增大 `--tensor-parallel-size` |
| `401 Unauthorized` (HuggingFace) | HF_TOKEN 未设置或无权限 | 申请模型访问权限并重新设置 token |
| `vLLM service did not start` | 模型下载慢或 GPU 不足 | 检查 `logs/vllm_*.log`，增大等待时间 |
| `No data file found` | 数据未上传 | 检查 `data/` 目录，上传缺失文件 |
| `meip_samples_v3.jsonl` 找不到 | 用了 fixed 版本路径 | 正确：用 `meip_samples_v3_fixed.jsonl` |
| ECD 结果 n=0 | 推理模型解析失败 | 跳过 ECD：不传 `--tasks ecd` |

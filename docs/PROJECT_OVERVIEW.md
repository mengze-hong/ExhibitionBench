# ExhibitionBench — 项目代码与文件全貌梳理

> 更新：2026-05-02  
> 项目根目录：`C:\Users\mengzehong\Desktop\展览馆llm`  
> 论文目录：`C:\Users\mengzehong\Desktop\ExhibBench-repo\paper\`

---

## 一、顶层目录结构

```
展览馆llm/
├── baselines/              ← 所有评测脚本
├── data/                   ← 数据集（JSONL格式）
├── results/                ← 评测结果（JSON）
├── analysis/               ← 分析脚本（误差分析、消融实验等）
├── benchmark/              ← 数据集构建工具
├── system/                 ← Gradio Demo 系统
├── logs/                   ← 运行日志
├── paper/                  ← 论文辅助资料（旧）
├── HANDOVER.md             ← 项目交接文档（完整状态、已有结果）
├── server_todo.md          ← 开源模型服务器推理 TODO（新建）
├── progress.md             ← 进度追踪
├── requirements.txt        ← Python 依赖
└── collect_*.py            ← 数据收集脚本（爬取博物馆 API）
```

---

## 二、baselines/ — 评测脚本

### 2.1 主力脚本

| 文件 | 用途 | 状态 |
|------|------|------|
| `sota_eval.py` | 闭源 SOTA 模型评测（LiteLLM 代理）| ✅ 主力，三任务全支持 |
| `multimodal_eval.py` | 视觉多模态评测（MEIP+图片）| ✅ 完成，5个视觉模型 |
| `openllm_baseline.py` | 开源模型评测（vLLM/Groq/Ollama）| ✅ 完成，待服务器运行 |

### 2.2 sota_eval.py 用法

```bash
# 三任务完整评测
python baselines/sota_eval.py --models gpt-5.2 claude-opus-4.6 --tasks meip tes ecd

# 单任务
python baselines/sota_eval.py --models gpt-5.2 --tasks meip --workers 150

# 所有已配置模型
python baselines/sota_eval.py --models all --tasks meip
```

**支持的模型**（在 `MODELS` 字典中）：
- gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini
- claude-opus-4.6, claude-opus-4.5, claude-sonnet-4.5
- gemini-2.5-pro, gemini-2.5-flash, gemini-3-pro-preview, gemini-3-flash-preview
- doubao-seed-2.0-pro, doubao-seed-1.6
- deepseek-v3.2, deepseek-r1
- kimi-k2.5
- minimax-m2.5, glm-5, qwen-plus-latest

**结果命名**：`results/{task}_{model}_shot{N}.json`  
**关键坑**：TES 结果键 `ndcg@10`，ECD 结果键 `macro_pairaccc`

### 2.3 multimodal_eval.py 用法

```bash
# 跑全部5个视觉模型
python baselines/multimodal_eval.py --models all --workers 150

# 跑指定模型
python baselines/multimodal_eval.py --models gpt-5.2 gemini-2.5-flash --workers 150
```

**支持模型**：gpt-5.2, claude-opus-4.6, gemini-2.5-pro, gemini-2.5-flash, doubao-seed-1.6-vision-250815  
**结果命名**：`results/meip_{model}_vision_shot0.json`  
**关键实现**：Vertex AI (Gemini) 使用 `GEMINI_TRUSTED_IMG_DOMAINS` 严格过滤图片 URL

### 2.4 辅助脚本

| 文件 | 用途 |
|------|------|
| `bm25_baseline.py` | BM25 检索 baseline |
| `embedding_baseline.py` | 句向量相似度 baseline |
| `gpt4o_zeroshot.py` | 早期 GPT-4o 评测（已被 sota_eval.py 取代）|
| `gpt_fewshot.py` | Few-shot 消融实验 |
| `rag_kg_baseline.py` | RAG + 知识图谱 baseline |
| `sbert_cultural_bias.py` | 文化偏见分析工具 |

---

## 三、data/ — 数据集文件

### 3.1 核心评测数据（服务器必须上传）

| 文件 | 大小 | 样本数 | 说明 |
|------|------|--------|------|
| `meip_samples_v3_fixed.jsonl` | 4.3MB | **1409** | ✅ MEIP 规范版，**用这个** |
| `tes_samples_v3.jsonl` | 9.7MB | 283 | TES 最新版 |
| `ecd_samples_v3.jsonl` | 2.3MB | 800 | ECD 最新版 |
| `objects_v3.jsonl` | 13MB | 23658 | 展品元数据库 |

### 3.2 数据版本关系

```
meip_samples.jsonl          ← 最早版本（已弃用）
meip_samples_v2.jsonl       ← 第二版（已弃用）
meip_samples_v3.jsonl       ← v3 基础版
meip_samples_v3_fixed.jsonl ← v3 修复版（修复 MetMuseum 展品 ID）⚠️ 用这个
meip_samples_v4.jsonl       ← v4 草稿（暂不用于评测）
meip_hard_samples.jsonl     ← 困难子集（未来实验用）
meip_open_samples.jsonl     ← 开放答案子集
```

**`find_data_file()` 优先级**：`_v3 > _v2 > bare`  
注意：此函数找不到 `_v3_fixed`，评测脚本内部会自动用 `meip_samples_v3.jsonl`。  
若要强制用 `_v3_fixed`，需手动修改路径或在服务器上软链：  
```bash
ln -s meip_samples_v3_fixed.jsonl meip_samples_v3.jsonl  # 覆盖
```

### 3.3 其他数据文件

| 文件 | 说明 |
|------|------|
| `exhibitions_v3.jsonl` | 展览元数据（主题、展品列表）|
| `objects.jsonl`, `objects_v2.jsonl` | 旧版展品库 |
| `kg.json` | 文化知识图谱 |
| `aic_*.jsonl`, `cleveland_*.jsonl`, `vam_*.jsonl` | 各博物馆原始数据 |
| `raw/` | 爬取的原始 API 响应（体积大，不上传服务器）|

---

## 四、results/ — 评测结果（141个文件）

### 4.1 视觉评测结果（已完成）

| 文件 | MRR | Hit@1 | n |
|------|-----|-------|---|
| `meip_gpt-5.2_vision_shot0.json` | 0.6166 | 0.5093 | 1294 |
| `meip_gemini-2.5-flash_vision_shot0.json` | 0.5916 | 0.4786 | 1404 |
| （gemini-2.5-pro, claude-opus-4.6, doubao vision — 正在跑）| | | |

### 4.2 MEIP 文本评测结果（已完成，v3fixed）

| 模型 | MRR | Hit@1 | 文件 |
|------|-----|-------|------|
| doubao-seed-2.0-pro | 0.6419 | 0.5394 | `meip_doubao-seed-2.0-pro_shot0_v3fixed.json` |
| claude-opus-4.6 | 0.6205 | 0.5096 | `meip_claude-opus-4.6_shot0_v3fixed.json` |
| gpt-5.2 | 0.6189 | 0.5103 | `meip_gpt-5.2_shot0_v3fixed.json` |
| doubao-seed-1.6 | 0.6168 | 0.506 | `meip_doubao-seed-1.6_shot0_v3fixed.json` |
| gemini-2.5-pro | 0.6152 | 0.5032 | `meip_gemini-2.5-pro_shot0_v3fixed.json` |
| gpt-5.1 | 0.6146 | 0.5018 | `meip_gpt-5.1_shot0_v3fixed.json` |
| claude-opus-4.5 | 0.6087 | 0.4982 | `meip_claude-opus-4.5_shot0_v3fixed.json` |
| gpt-5 | 0.5988 | 0.4833 | `meip_gpt-5_shot0_v3fixed.json` |
| deepseek-v3.2 | 0.5937 | 0.4755 | `meip_deepseek-v3.2_shot0_v3fixed.json` |
| claude-sonnet-4.5 | 0.5706 | 0.4528 | `meip_claude-sonnet-4.5_shot0_v3fixed.json` |
| gemini-2.5-flash | 0.5522 | 0.4287 | `meip_gemini-2.5-flash_shot0_v3fixed.json` |
| deepseek-r1 | 0.5294 | 0.3911 | `meip_deepseek-r1_shot0_v3fixed.json` |
| kimi-k2.5 | 0.5063 | 0.3691 | `meip_kimi-k2.5_shot0_v3fixed.json` |
| minimax-m2.5 | 0.4688 | 0.3187 | — |
| glm-5 | 0.3919 | 0.2271 | — |

### 4.3 ECD 评测结果（已完成）

| 模型 | macro_pairaccc | 备注 |
|------|----------------|------|
| gemini-2.5-pro | 0.860 | 最高 |
| doubao-seed-1.6 | 0.854 | |
| doubao-seed-2.0-pro | 0.852 | |
| claude-opus-4.6 | 0.836 | |
| gpt-5.1 | 0.836 | |
| gpt-5.2 | 0.826 | |
| gemini-2.5-flash | 0.813 | |
| deepseek-v3.2 | 0.803 | |
| gpt-5 | 0.800 | |
| kimi-k2.5 | 0.498 | ⚠️ 推理链问题，约等于随机 |
| deepseek-r1 | 0.486 | ⚠️ 同上 |
| gpt-5-mini | n=0 | ❌ API 失败 |
| qwen-plus-latest | n=0 | ❌ API 失败 |

### 4.4 TES 评测结果（已完成）

结果在 `results/tes_*_shot0.json`，指标为 `ndcg@10`（含@符号）。  
doubao TES NDCG=0.7348 异常偏高，需要 investigate。

### 4.5 Few-shot 消融实验（部分完成）

| 任务 | 已跑模型 | shot 数 |
|------|----------|---------|
| MEIP | claude-opus-4.6, claude-sonnet-4.5, gemini-2.5-flash, deepseek-v3.2 | 0,1,3,5 |
| TES | 同上 | 0,1,3,5 |

---

## 五、analysis/ — 分析脚本

| 文件 | 功能 |
|------|------|
| `error_analysis.py` | 分析模型预测错误类型（文化跨越、时代混淆等）|
| `cultural_bias.py` | 单模型文化偏见量化 |
| `cultural_bias_multi_model.py` | 多模型文化偏见对比 |
| `metadata_ablation.py` | 元数据消融实验（去掉 title/culture/date 看影响）|
| `fewshot_mechanism.py` | Few-shot 机制分析 |
| `contamination_ablation.py` | 数据污染检验 |
| `tes_leakage_analysis.py` | TES 任务数据泄漏分析 |
| `summarize_analysis.py` | 分析结果汇总 |

---

## 六、benchmark/ — 数据集构建工具

| 文件 | 功能 |
|------|------|
| `build_samples.py` | 从 exhibitions.jsonl 构建 MEIP/TES 样本 |
| `rebuild_samples.py` | 重建样本（v3 版本）|
| `ecd_generator.py` | ECD 任务样本生成 |
| `meip_eval.py` | MEIP 评测核心逻辑（被 sota_eval.py 调用）|
| `tes_eval.py` | TES 评测核心逻辑 |
| `fix_met_meip.py` | MetMuseum ID 修复工具（生成 _fixed 版本）|
| `meip_hard_generator.py` | 困难样本生成 |
| `human_ab_*.py` | 人工标注 A/B 测试工具 |
| `annotation_ui.html` | 人工标注 Web 界面 |

---

## 七、API 配置

```python
# 内部 LiteLLM 代理（无限额度，仅 Windows 本机可访问）
API_KEY  = "sk-TpK0g832p8LbMXTdI_pjkQ"
BASE_URL = "http://csig.litellm.prod.sgpolaris/v1"
```

> ⚠️ 此 API 仅在内网可用，**服务器上无法使用**。  
> 服务器上开源模型需自己部署 vLLM，或用 Groq/Together AI 公共 API。

---

## 八、正在运行的任务

```
PID 548 — python baselines/multimodal_eval.py --models gemini-2.5-pro claude-opus-4.6 doubao-seed-1.6-vision-250815 --workers 150
日志: logs/multimodal_eval2_stdout.log
当前进度: gemini-2.5-pro @ 300/1409, MRR=0.7243 (2026-05-02 14:15)
```

---

## 九、待办事项（优先级排序）

### P0 — 紧急
- [ ] 等待多模态 eval 完成（gemini-2.5-pro, claude-opus-4.6, doubao-vision）
- [ ] 在服务器跑开源模型：Llama-3.1-8B, Llama-3.3-70B, Qwen2.5-72B（见 server_todo.md）

### P1 — 重要
- [ ] 修复 compile_results.py 键名（TES `ndcg@10`，ECD `macro_pairaccc`）
- [ ] 修复 ECD 推理链模型（kimi-k2.5, deepseek-r1）的解析：从 `<think>` 后提取最终答案
- [ ] 调查 gpt-5-mini / qwen-plus-latest ECD n=0 问题（API 模型名称确认）
- [ ] 调查 doubao TES NDCG=0.7348 异常高原因
- [ ] 完成 shot 消融实验（kimi, doubao, gemini-pro 的 shot1/3/5）

### P2 — 可选
- [ ] 完成元数据消融实验（metadata_ablation.py）
- [ ] 多模态 TES 评测（当前 multimodal_eval.py 只支持 MEIP）
- [ ] 更新论文表格（policyeval_bench.tex）加入开源模型结果

---

_by Claude — 2026-05-02_

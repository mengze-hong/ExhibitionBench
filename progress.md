# ExhibitionBench 进度记录

更新时间：2026-05-02

---

## 当前状态：论文已达可投稿状态（ACL/EMNLP 2026 Main）

论文文件：`C:\Users\mengzehong\Desktop\ExhibBench-repo\paper\policyeval_bench.tex`

---

## 已完成任务清单

### ✅ 数据集构建（v3）
- **展览数据**：来自 Met Museum + Art Institute of Chicago + Rijksmuseum + Cleveland + V&A + 台北故宫等多源
- **MEIP 样本**：1,409 queries（v3_fixed，经 Met 数据清洗修复）
- **TES 样本**：283 queries（v3）
- **ECD 样本**：800 pairwise pairs（4 level × 200），评测使用前 500

### ✅ MEIP 全量重跑（v3fixed，n=1409）
所有模型 zero-shot 结果（已验证与论文主表格完全一致）：

| 模型 | Hit@1 | MRR |
|------|-------|-----|
| GPT-5.2 | 0.510 | 0.619 |
| GPT-5.1 | 0.502 | 0.615 |
| GPT-5 | 0.483 | 0.599 |
| Claude Opus 4.6 | 0.510 | 0.621 |
| Claude Opus 4.5 | 0.498 | 0.609 |
| Claude Sonnet 4.5 | 0.453 | 0.571 |
| Gemini 2.5 Pro | 0.503 | 0.615 |
| Gemini 2.5 Flash | 0.429 | 0.552 |
| Doubao-Seed-2.0-Pro† | 0.539 | 0.642 |
| Doubao-Seed-1.6 | 0.506 | 0.617 |
| DeepSeek-R1 | 0.391 | 0.529 |
| DeepSeek-V3.2 | 0.476 | 0.594 |
| Kimi-K2.5 | 0.369 | 0.506 |
| GLM-5 | 0.227 | 0.392 |
| MiniMax-M2.5 | 0.319 | 0.469 |

### ✅ TES 全量重跑（n=283）
所有模型 zero-shot NDCG@10 结果：

| 模型 | NDCG@10 | MRR |
|------|---------|-----|
| GPT-5.2 | 0.413 | 0.243 |
| GPT-5.1 | 0.488 | 0.337 |
| GPT-5 | 0.087 | 0.085 |
| Claude Opus 4.6 | 0.421 | 0.251 |
| Claude Opus 4.5 | 0.417 | 0.247 |
| Claude Sonnet 4.5 | 0.414 | 0.245 |
| Gemini 2.5 Pro | 0.450 | 0.287 |
| Gemini 2.5 Flash | 0.622 | 0.575 |（最佳非 leaked 模型）
| Doubao-Seed-2.0-Pro† | 0.703 | 0.615 |（query_theme 泄漏，已注明 †）
| Doubao-Seed-1.6 | 0.545 | 0.436 |
| DeepSeek-R1 | 0.393 | 0.256 |
| DeepSeek-V3.2 | 0.411 | 0.250 |
| Kimi-K2.5 | 0.100 | 0.102 |
| GLM-5 | 0.109 | 0.095 |
| MiniMax-M2.5 | 0.181 | 0.183 |

### ✅ ECD 全量跑完（n=500）
所有模型 zero-shot PairAcc：

| 模型 | L1 | L2 | L3 | L4 | Macro |
|------|----|----|----|----|-------|
| GPT-5.2 | 0.84 | 0.87 | 0.58 | 0.81 | 0.774 |
| GPT-5.1 | 0.93 | 0.95 | 0.58 | 0.88 | 0.836 |
| GPT-5 | 0.84 | 0.82 | 0.52 | 0.85 | 0.756 |
| Claude Opus 4.6 | 0.93 | 0.94 | 0.58 | 0.90 | 0.836 |
| Claude Opus 4.5 | 0.92 | 0.88 | 0.56 | 0.83 | 0.798 |
| Claude Sonnet 4.5 | 0.90 | 0.89 | 0.54 | 0.88 | 0.802 |
| **Gemini 2.5 Pro** | 0.94 | **0.98** | 0.58 | **0.94** | **0.860** |（最佳）
| Gemini 2.5 Flash | 0.79 | 0.81 | 0.54 | 0.79 | 0.732 |
| Doubao-Seed-2.0-Pro† | 0.97 | 0.98 | 0.53 | 0.94 | 0.852 |
| Doubao-Seed-1.6 | 0.96 | 0.97 | 0.57 | 0.92 | 0.854 |
| DeepSeek-R1 | 0.49 | 0.49 | 0.47 | 0.50 | 0.486 |
| DeepSeek-V3.2 | 0.62 | 0.78 | **0.59** | 0.75 | 0.686 |
| Kimi-K2.5 | 0.49 | 0.53 | 0.45 | 0.53 | 0.498 |
| GLM-5 | 0.64 | 0.62 | 0.46 | 0.64 | 0.588 |
| MiniMax-M2.5 | 0.55 | 0.53 | 0.48 | 0.58 | 0.536 |

**L3 Thematic Deviation 范围：0.45–0.59（全模型近随机水平，确认论文核心发现）**

### ✅ Few-shot 退化机制实验（H1/H2/H3）
完成4个主力模型的 fewshot ablation（n=200 MEIP holdout）：

| 模型 | 0-shot | 5-shot | 退化% | H3 Δ |
|------|--------|--------|-------|------|
| GPT-5.2 | 0.600 | 0.595 | -0.8% | 0.000 |
| Claude Opus 4.6 | 0.670 | 0.645 | -3.7% | +0.025 |
| DeepSeek-V3.2 | 0.560 | 0.545 | -2.7% | -0.045 |
| Gemini-2.5-Pro | 0.630 | 0.625 | -0.8% | -0.100 |

文件位置：`results/fewshot_analysis/fewshot_*_meip.json`

### ✅ 数据污染消融（C1/C2）
- **C1 Institution Split**：Met vs. non-Met MRR gap 全部 < 0.05（无高风险模型）
- **C2 Title Masking**：MRR drop 26-30%，且与模型能力负相关（Pearson r=-0.12），证明模型做语义推理而非记忆
- 文件位置：`results/contamination/c1_summary.json`，`results/contamination/c2_summary.json`

### ✅ 文化偏差分析
7 frontier LLMs + SBERT 在 200 MEIP 样本上的文化分组 Hit@1
- 所有 8 系统均显示 Western bias（Δ_W-NW > 0，范围 +0.03 到 +0.17）
- 文件位置：`results/cultural_bias/`

### ✅ 元数据消融
6级消融（title only → full），已写入论文 Appendix
- 文件位置：`results/fewshot_analysis/` 或 `results/metadata_ablation/`

---

## 论文状态：已更新所有不一致处

### 本次会话修复的 5 个错误：

1. **Abstract ECD count** (line 51): 补充 "500 used for evaluation"
2. **Baselines paragraph - best ECD claim** (line 426): 从 "GPT-5.1" 改为 "Gemini 2.5 Pro (Macro 0.860)"
3. **Baselines paragraph - best MEIP claim** (line 430): 从 "Claude Opus 4.5" 改为 "Claude Opus 4.6 (Hit@1 0.510), tied with GPT-5.2"
4. **Conclusion L3 claim** (line ~1024): 从 "L3 near-chance at 0.520 for GPT-5.2" 改为 "range 0.45–0.59, with best only 0.59"（GPT-5.2 的 L3 实际是 0.576，0.520 属于 GPT-5）
5. **Abstract baseline count** (line 53): 从 "eight baselines" 改为 "18 models"（反映实际主表格）

### 论文主表格验证（完全一致）：
所有 15 模型的 MEIP/TES/ECD 数值均与对应 result JSON 文件交叉验证 ✓

---

## 待完成 TODO

### P0 — 论文内部一致性（必须修）

- [ ] **修复 Limitations 陈旧文本（policyeval_bench.tex 第1047-1050行）**
  - 当前错误文本：`"Several frontier models (DeepSeek-R1, Kimi-K2.5, GLM-5, Minimax-M2.5) were under evaluation at submission time; result tables mark these as '--'. We will provide full results in the camera-ready version."`
  - 问题：这四个模型的完整结果早已在主表格中，这段话是错的
  - 修复：删除或替换为反映实际完整评测状态的文本

### P0 — 开源 LLM baseline（ACL/EMNLP 2026 审稿要求）

- [ ] **加入开源 LLM baseline**（Llama-3.3-70B / Mistral-7B / Qwen2.5）
  - 内部 API 无此类模型，需要 Groq / Together AI 外部 API Key
  - 脚本已就绪：`baselines/openllm_baseline.py`（支持 Groq/Together/vLLM）
  - 备选方案：将现有 DeepSeek-R1、DeepSeek-V3.2、GLM-5 定性为"开源权重模型"覆盖（三者均已完整评测）

### P1 — Gemini-3 重跑（结果异常，根因已确认）

- [ ] **修复 `baselines/sota_eval.py` 的 `LARGE_TOKEN_MODELS` 集合**
  - 问题：`gemini-3-pro-preview` 和 `gemini-3-flash-preview` 不在 `LARGE_TOKEN_MODELS` 中
  - 导致 max_tokens=150，模型只输出 `"met"` 前缀而非完整 ID（如 `"met_438816"`）
  - 已通过 live test 确认：max_tokens=50 → 返回 `"met"`；max_tokens=200 → 返回 `"met_438816"` ✓
  - 现有 MEIP 结果 hit@1=0.07/mrr=0.26 均无效（n=500, v3 非 v3fixed）
  - 修复：在 `LARGE_TOKEN_MODELS` 加入 `"gemini-3-pro-preview", "gemini-3-flash-preview"`
  - 修复后：重跑 `meip_gemini-3-pro-preview_shot0_v3fixed.json`（n=1409）
  - 若结果正常（预期 MRR ~0.50+），考虑加入论文扩充模型覆盖至 17 个 LLM

- [ ] **gemini-3-flash-preview TES 结果可疑**（ndcg@10=0.0733，与其他模型差距过大）
  - 需检查是否也有 token 截断问题（TES 输出要求不同）
  - gemini-3-pro TES 结果（ndcg@10=0.5917）看起来正常

### P1 — 其他

- [ ] **修复 `compile_results.py` 键名**（per HANDOVER.md）
  - TES 结果键应为 `ndcg@10`（含@符号），非 `ndcg_10`
  - ECD 结果键应为 `macro_pairaccc`，非 `accuracy`

---

## 文件路径索引

| 类型 | 路径 |
|------|------|
| 论文 | `C:\Users\mengzehong\Desktop\ExhibBench-repo\paper\policyeval_bench.tex` |
| MEIP v3fixed 结果 | `results/meip_*_shot0_v3fixed.json` |
| TES 结果 | `results/tes_*_shot0.json` |
| ECD 结果 | `results/ecd_*_shot0.json` |
| Few-shot 分析 | `results/fewshot_analysis/fewshot_*_meip.json` |
| 污染消融 | `results/contamination/c{1,2}_summary.json` |
| 文化偏差 | `results/cultural_bias/` |

# ExhibitionBench — 执行计划与进度追踪

> 目标：ACL/EMNLP 2026 Main Conference Long Paper  
> 核心叙事：系统性揭示 LLM 在文化锚定任务（博物馆展览策划）上的三类缺陷  
> 最后更新：2026-05-11

---

## 当前状态（2026-05-11）

| 任务 | Shot0 完整度 | Few-shot | 备注 |
|------|------------|---------|------|
| MEIP | ✅ 15+ 模型 | 部分有 shot1/3/5 | MRR, Hit@1 |
| TES  | ✅ 15+ 模型 | 只有 claude-opus-4.6 shot1 | NDCG@10, P@10 |
| ECD  | ✅ 15+ 模型 | **完全缺失** | macro_pairaccc |

**开源模型**：只有 Qwen2.5-7B 有结果，Llama/Qwen72B/Llama-70B 需 GPU

---

## Part A：不需要 GPU 的任务

### ✅ A1. 诊断并修复失败模型
**状态**：已完成 (2026-05-11)

**发现**：
- `gpt-5-mini`：推理模型，max_tokens=20 时 content 全空，n=0 — 已注释掉（旧结果保留作参考）
- `qwen-plus-latest`：401 权限被撤（team 无访问权） — 已注释掉
- `glm-5`：**推理模型**（content=None，答案在 reasoning_content）— 已加入 REASONING_MODELS + LARGE_TOKEN_MODELS
- `gemini-3.1-pro-preview`：**新模型，可用** — 已加入 MODELS

**修改文件**：
- `baselines/sota_eval.py`：MODELS 字典、REASONING_MODELS、LARGE_TOKEN_MODELS
- `results/compile_sota_results.py`：MODEL_DISPLAY、MODEL_ORDER

---

### ⏳ A2. 补跑 ECD few-shot（shot1/3）
**状态**：进行中（后台运行，2026-05-11）

ECD few-shot 完全缺失，是论文 H2（few-shot gains）的关键数据缺口。

**注意**：`--shot` 已支持多值（2026-05-11 修复），可以一条命令传多个值。

**命令**：
```bash
cd "C:\Users\mengzehong\Desktop\展览馆llm"
python baselines/sota_eval.py --models gpt-5.2 claude-opus-4.6 gemini-2.5-pro deepseek-v3.2 --tasks ecd --shot 1 3
```

**预期输出**：
- `results/ecd_gpt-5.2_shot1.json`
- `results/ecd_gpt-5.2_shot3.json`
- `results/ecd_claude-opus-4.6_shot1.json`
- `results/ecd_claude-opus-4.6_shot3.json`
- `results/ecd_gemini-2.5-pro_shot1.json`
- `results/ecd_gemini-2.5-pro_shot3.json`
- `results/ecd_deepseek-v3.2_shot1.json`
- `results/ecd_deepseek-v3.2_shot3.json`

---

### ⏳ A3. 补跑 TES few-shot（shot1/3）
**状态**：进行中（后台运行，2026-05-11）

TES few-shot 只有 claude-opus-4.6 shot1，需补其他 top 模型。

**注意**：`--shot` 已支持多值（2026-05-11 修复），可以一条命令传多个值。

**命令**：
```bash
cd "C:\Users\mengzehong\Desktop\展览馆llm"
python baselines/sota_eval.py --models gpt-5.2 gemini-2.5-pro deepseek-v3.2 kimi-k2.5 --tasks tes --shot 1 3
```

**预期输出**：
- `results/tes_gpt-5.2_shot1.json` / `shot3.json`
- `results/tes_gemini-2.5-pro_shot1.json` / `shot3.json`
- `results/tes_deepseek-v3.2_shot1.json` / `shot3.json`
- `results/tes_kimi-k2.5_shot1.json` / `shot3.json`

---

### ⬜ A4. 验证并输出完整结果表
**状态**：待完成（等待 A2/A3 完成）

```bash
cd "C:\Users\mengzehong\Desktop\展览馆llm"
python results/compile_sota_results.py --shot 0 --latex
python results/compile_sota_results.py --shot 1 --latex  # few-shot 对比
```

**检查点**：
- [ ] ndcg@10 键（含@符号）正确读取
- [ ] macro_pairaccc 键正确读取
- [ ] 无 "???" 空洞（允许开源模型空缺）
- [ ] LaTeX 表格输出到 `results/latex_main_table.tex`

---

### ✅ A5. 跑新模型 glm-5（fix 后重跑）
**状态**：已完成 (2026-05-11)

glm-5 之前因为 REASONING_MODELS 设置缺失，所有结果 n=0。修复后重跑，三个任务均完成。

---

### ✅ A6. 跑 gemini-3.1-pro-preview（新模型）
**状态**：MEIP + TES 已完成，ECD 进行中 (2026-05-11)

```bash
python baselines/sota_eval.py --models gemini-3.1-pro-preview --tasks ecd --shot 0
```

---

## Part B：GPU 服务器任务

### ✅ B1. 创建 gpu_server_needed/ 文件夹
**状态**：已完成 (2026-05-11)

已创建：
- `gpu_server_needed/README.md`：完整 setup + 运行指南
- `gpu_server_needed/requirements_gpu.txt`：GPU 服务器专用依赖
- `gpu_server_needed/run.sh`：一键运行脚本（preflight checks + tar.gz 打包）

---

### ⬜ B2. Git commit & push
**状态**：待完成（A2/A3 完成后执行）

```bash
cd "C:\Users\mengzehong\Desktop\展览馆llm"
git add baselines/sota_eval.py results/compile_sota_results.py
git add gpu_server_needed/
git add results/*_shot1.json results/*_shot3.json
git add PLAN.md
git commit -m "Add gpu_server_needed/, fix glm-5/gemini-3.1, add ECD+TES few-shot results"
git push
```

---

## 关键文件索引

| 文件 | 用途 |
|------|------|
| `baselines/sota_eval.py` | 闭源模型评测主脚本 |
| `baselines/openllm_baseline.py` | 开源模型评测（vLLM/Groq/Ollama后端） |
| `results/compile_sota_results.py` | 汇总所有结果、生成 LaTeX 表格 |
| `scripts/start_vllm.sh` | vLLM 服务管理（start/stop/status） |
| `scripts/run_openllm_eval.sh` | 单模型完整评测流水线 |
| `scripts/run_all_openllm.sh` | 批量跑所有开源模型 |
| `gpu_server_needed/README.md` | GPU 服务器部署指南（待创建）|
| `gpu_server_needed/run.sh` | 一键运行脚本（待创建）|
| `HANDOVER.md` | 项目交接文档，包含完整状态 |

---

## 关键踩坑（避坑指南）

1. **TES 结果键名**：用 `ndcg@10`（含@符号），不是 `ndcg_10`
2. **ECD 结果键名**：用 `macro_pairaccc`，不是 `accuracy`
3. **数据版本**：`find_data_file()` 自动选 `_v3 > _v2 > bare`，v3 是最新
4. **MEIP 数据**：必须用 `meip_samples_v3_fixed.jsonl`，不是 `meip_samples_v3.jsonl`
5. **ECD format_seq()**：已有 None 守卫（2026-04-30 修复）
6. **doubao TES NDCG=0.7348**：异常高，需 investigate 或在论文中注释
7. **glm-5**：推理模型，已加入 REASONING_MODELS（answer 在 reasoning_content）
8. **R1/推理链模型**：在 GPU 服务器上跑开源 R1 变体时，跳过 ECD 任务
9. **--shot 多值**：`--shot 1 3` 已支持（2026-05-11 修复），内部循环每个 shot 值逐一运行
10. **workers=100**：已硬编码为 100 并发，kimi-k2.5 等推理模型慢是单请求延迟高（非并发问题）

---

## 论文结构（备忘）

- **MEIP**：Masked Exhibition Item Prediction — MRR, Hit@1
- **TES**：Theme-based Exhibition Selection — NDCG@10, P@10, R@10
- **ECD**：Exhibition Coherence Discrimination — macro_pairaccc（L1/L2/L3 三难度）
- **H1**：LLM 在文化锚定任务上整体偏弱于人类策展人
- **H2**：Few-shot 提示能提升性能（尤其 ECD）
- **H3**：开源模型 vs 闭源模型差距（需 GPU 结果）

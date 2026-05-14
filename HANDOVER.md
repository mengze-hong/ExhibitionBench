# ExhibitionBench — AI 接手交接文档

> 版本：v2.0 | 日期：2026-05-13 | 面向：任何接手本项目的 AI 助手

---

## 一、项目定位

**ExhibitionBench** 是面向博物馆展览策划的 LLM 评测基准，目标发表于 **ACL/EMNLP 2026 Main Conference**。  
核心贡献：系统性揭示 LLM 在三类文化锚定任务（MEIP / TES / ECD）上的四类系统性缺陷。

- **论文目录**：`C:\Users\mengzehong\Desktop\ExhibBench-repo\paper\`
- **代码/数据目录**：`C:\Users\mengzehong\Desktop\展览馆llm\`

---

## 二、目录结构

```
C:\Users\mengzehong\Desktop\展览馆llm\
├── data\
│   ├── objects_v3.jsonl              ← 主展品库（23,658件，5数据源）
│   ├── exhibitions_v3.jsonl          ← 展览数据（300个）
│   ├── meip_samples_v3_fixed.jsonl   ← MEIP评测集（1,409样本，10-way）
│   ├── ecd_samples_v3.jsonl          ← ECD评测集（500样本，4级扰动）
│   ├── tes_samples_v3.jsonl          ← TES评测集（283样本，50-way）
│   ├── kg.json                       ← 知识图谱
│   ├── raw/                          ← 原始API抓取（.gitignored）
│   └── archive/                      ← 旧版v1/v2数据（.gitignored）
├── baselines\
│   └── sota_eval.py              ← 主评测脚本（唯一入口）
├── results\                      ← 所有模型评测结果
│   ├── compile_sota_results.py   ← 汇总脚本（生成主表/LaTeX/CSV）
│   ├── *.json                    ← 21模型 × 3任务 × 3shot 结果
│   ├── tables/                   ← 生成的CSV/LaTeX表格（.gitignored，可重新生成）
│   ├── baselines_pred/           ← 早期baseline预测JSONL（.gitignored）
│   ├── contamination/            ← 污染分析结果
│   ├── cultural_bias/            ← 文化偏见分析结果
│   ├── fewshot_analysis/         ← few-shot消融结果
│   └── metadata_ablation/        ← 元数据消融结果
├── scripts\
│   ├── data_collection/          ← 数据采集脚本（collect_*.py）
│   ├── utils/                    ← 工具脚本（run_until_done.py等）
│   ├── compute_human_eval_metrics.py
│   └── generate_human_eval_outputs.py
├── analysis\                     ← 分析脚本（文化偏见、误差分析等）
├── benchmark\                    ← 评测集构建脚本
├── docs\                         ← 项目文档（进度、计划、review）
├── human_eval\                   ← 人工评测数据
├── system\                       ← Demo系统（nicegui_app.py）
├── paper\                        ← 本地论文备份
├── logs\                         ← 运行日志（.gitignored）
├── gpu_server_needed\            ← GPU服务器开源LLM相关
├── README.md
├── HANDOVER.md
└── requirements.txt
C:\Users\mengzehong\Desktop\ExhibBench-repo\
├── paper\
│   ├── exhibitionbench.tex       ← 论文正文（唯一来源）
│   ├── references.bib
│   └── results\
│       └── latex_vision_table.tex ← Vision消融表（\input引用）
└── HANDOVER.md                   ← 旧版（已弃用）
```

---

## 三、API 与运行环境

```python
api_key  = "sk-TpK0g832p8LbMXTdI_pjkQ"
base_url = "http://csig.litellm.prod.sgpolaris/v1"
model    = "gpt-5.2"   # 默认
```

```bash
cd "C:\Users\mengzehong\Desktop\展览馆llm"
python baselines/sota_eval.py --models MODEL --tasks meip tes ecd
python baselines/sota_eval.py --models MODEL --tasks meip tes ecd --shot 1  # few-shot
python baselines/sota_eval.py --models MODEL --tasks meip --force            # 强制重跑
```

---

## 四、三个评测任务

| 任务 | 说明 | 样本数 | 关键 JSON 键 |
|------|------|--------|-------------|
| **MEIP** | 掩码展品预测，10-way ranking | 1,409 | `mrr`, `hit@1` |
| **TES** | 主题展览选择，50-way ranking | 283 | `ndcg@10`（含@！）, `mrr` |
| **ECD** | 展览连贯性判别，4级扰动 | 500 | `macro_pairaccc`, `pairaccc_L1/L2/L3/L4` |

**TES noleak 协议**：候选展览用匿名 ID（EX_001…EX_050）隐藏 theme/title/description，只保留样品展品列表。结果文件命名为 `tes_{model}_shot{n}_noleak.json`。`compile_sota_results.py` 自动优先读 `_noleak` 版本。

---

## 五、当前结果状态（截至 2026-05-13）

### 覆盖范围

- **21 个前沿 LLM** 全部完成 shot0（meip/tes/ecd 三任务）
- **shot1 / shot3** 全部完成（21 模型 × 3 任务）
- TES 全部 21 个 noleak 文件存在（`tes_*_shot0_noleak.json`）

### MEIP 主要结果（shot0，MRR）

| 模型 | MRR | Hit@1 |
|------|-----|-------|
| **Gemini 3.1 Pro** | **0.827** | 0.782 |
| Gemini 3 Pro | 0.820 | 0.774 |
| Claude Opus 4.6 | 0.781 | 0.728 |
| GPT-5.1 | 0.748 | 0.686 |
| Gemini 2.5 Pro | 0.748 | 0.686 |
| GPT-5.2 | 0.746 | 0.682 |
| Gemini 3 Flash | 0.748 | 0.688 |
| ... | ... | ... |

### TES 主要结果（shot0，NDCG@10，noleak协议）

| 模型 | NDCG@10 | MRR |
|------|---------|-----|
| **Claude Opus 4.6** | **0.437** | 0.290 |
| Claude Sonnet 4.5 | 0.418 | 0.266 |
| Doubao Seed 2.0 Pro | 0.410 | 0.256 |
| GPT-5.1 | 0.406 | 0.255 |
| Gemini 2.5 Pro | 0.408 | 0.254 |
| ... | ... | ... |

### ECD 主要结果（shot0，macro_pairaccc）

| 模型 | Macro | L1 | L2 | L3 | L4 |
|------|-------|----|----|----|----|
| **Gemini 3.1 Pro** | **0.876** | 0.98 | 0.98 | 0.58 | 0.98 |
| Claude Opus 4.6 | 0.836 | 0.93 | 0.94 | 0.58 | 0.90 |
| GPT-5.1 | 0.836 | 0.93 | 0.95 | 0.58 | 0.88 |
| ... | ... | ... | ... | ... | ... |
| **L3 全员趋近随机** | 0.45–0.63 | — | — | ← 核心发现 | — |

完整表格：`python results/compile_sota_results.py`

---

## 六、关键踩坑记录（必读）

1. **TES 键名**：`ndcg@10`（含@），不是 `ndcg_10`
2. **ECD 键名**：`macro_pairaccc`，不是 `accuracy`
3. **`find_data_file()`** 自动选 `_v3 > _v2 > bare` 版本，当前 v3 最新
4. **TES noleak**：`compile_sota_results.py` 自动优先用 `_noleak` 文件；所有21个已存在
5. **CoT 崩溃**：Claude-Opus-4.6 开 CoT 后 MEIP MRR 从 0.781 → 0.314（format非合规）
6. **开源LLM**：内部 LiteLLM API 不提供 Llama/Qwen/Mistral 访问，只有 DeepSeek-R1/V3 系列

---

## 七、论文状态（截至 2026-05-13）

**Git repo**：`C:\Users\mengzehong\Desktop\ExhibBench-repo`（branch: main）

最新 commit：`217f1a5` Fix abstract/intro: correct stale +6-16% → actual gains

### 已完成章节

- **tab:main**：21 模型完整结果表，SoTA 格高亮，所有数值与磁盘一致
- **MEIP / TES / ECD 分析**：全部完成，含文化偏见分析
- **Few-shot 消融**（tab:shotablation）：6 模型 × 4 shot，含 GPT-5.2/Claude/Gemini-2.5/3-Pro/3.1-Pro
- **CoT 消融**（tab:cot）：3 模型，CoT 普遍损害 ECD（-14~-34pp），Claude MEIP 崩溃
- **Vision 消融**（tab:vision-ablation）：6 模型，图像平均 +0.003 MRR，无显著收益
- **文化偏见分析**：Western vs Non-Western（tab:cultural）
- **标签格式实验**（tab:h3）：4 模型 shuffled-label，验证 H3 Format Conformity
- **Limitations / Ethics**：完成
- **Abstract / Introduction / Conclusion**：4 个发现，与实验节一致

### 四个核心发现（论文叙事）

1. Few-shot gains on MEIP **heterogeneous**（Gemini-2.5-Pro +2.8%；GPT-5.2/Claude-Opus-4.6 saturated）
2. CoT **universally hurts ECD** (-14~-34pp Macro)；Claude MEIP collapse (-46.7pp)
3. **Systematic Western preference**（Δ_W-NW ∈ [+0.03, +0.17]）
4. **Coherence insensitivity**（ECD L3 best 0.63，range 0.45–0.63）

---

## 八、待办事项

### P1（论文发出前）

- [ ] **添加 Doubao Seed 1.6 Thinking 结果**（或从 `MODEL_DISPLAY/ORDER` 删除）
  - 目前 compile_sota_results.py 里有该模型但无结果文件，compile 会自动跳过（n=0）
  - 若不跑，需从 `compile_sota_results.py` 的 `MODEL_DISPLAY` / `MODEL_ORDER` 删掉
- [ ] **deepseek-v3.1 few-shot 结果**（shot1/shot3 缺失，shot0 存在）
  - 不影响论文（tab:shotablation 未包含该模型），但结果目录不完整

### P2（后续工作）

- [ ] 开源 LLM 基线（Llama/Qwen）——需另找推理端点，目前内部 API 无法访问
- [ ] Demo 系统（`system/nicegui_app.py`）——port 7861，现状不明

---

## 九、快速操作参考

```bash
# 查看当前结果摘要
cd "C:\Users\mengzehong\Desktop\展览馆llm"
python results/compile_sota_results.py

# 查看论文 git log
cd "C:\Users\mengzehong\Desktop\ExhibBench-repo"
git log --oneline -10

# 跑新模型（三任务全跑）
python baselines/sota_eval.py --models NEW_MODEL --tasks meip tes ecd

# 生成 LaTeX 表格
python results/compile_sota_results.py --latex
```

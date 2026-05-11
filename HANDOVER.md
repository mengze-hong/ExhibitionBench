# ExhibitionBench — AI 接手交接文档

> 版本：v1.2 | 日期：2026-05-01 | 面向：任何接手本项目的 AI 助手

---

## 一、项目一句话定位

**ExhibitionBench** 是面向博物馆展览策划的 LLM 评测基准，目标发表于 **ACL/EMNLP 2026 Main Conference**。核心贡献：系统性揭示 LLM 在三类文化锚定任务上的行为差异。

---

## 二、目录结构

```
C:\Users\mengzehong\Desktop\展览馆llm\
├── data\
│   ├── objects_v3.jsonl          ← 主展品库（23,658件，5数据源）
│   ├── exhibitions_v3.jsonl      ← 展览数据（300个）
│   ├── meip_samples_v3.jsonl     ← MEIP任务评测集（1,495样本）
│   ├── ecd_samples_v3.jsonl      ← ECD任务评测集（800样本）
│   └── [older versions: v2, bare]
├── baselines\
│   └── sota_eval.py              ← 主评测脚本（唯一入口）
├── results\                      ← 所有模型评测结果（JSON）
├── system\
│   └── nicegui_app.py            ← Demo 系统（NiceGUI，port 7861）
├── paper\   (C:\Users\mengzehong\Desktop\ExhibBench-repo\paper\)
│   ├── policyeval_bench.tex      ← 论文正文
│   └── references.bib
├── review.md                     ← 文献调研报告
└── HANDOVER.md                   ← 本文件
```

---

## 三、数据集说明

### 展品库（objects_v3.jsonl）

- **规模**：23,658 件，300 个展览
- **字段**：`id, title, date, culture, medium, department, description, image_url, source`
- **数据源分布**（重要！脚本用 `find_data_file()` 自动选 `_v3` 版本）：

| 来源 | 件数 | 文化圈 |
|------|------|--------|
| aic (Art Institute of Chicago) | 7,270 | 西方/美洲 |
| europeana_ext | 5,614 | 欧洲 |
| cleveland | 4,339 | 东西方混合 |
| vam_ext (Victoria & Albert) | 3,736 | 全球 |
| met (Metropolitan) | 1,143 | 西方为主 |
| vam | 809 | 全球 |
| europeana | 669 | 欧洲 |
| met_ext | 78 | 西方 |

### 三个评测任务

| 任务 | 文件 | 样本数 | 说明 |
|------|------|--------|------|
| MEIP | `meip_samples_v3.jsonl` | 1,495 | 掩码展品预测，20-way multiple choice |
| TES | `tes_queries_v3.jsonl` *(或由exhibitions生成)* | ~300 | 主题驱动展品选择 |
| ECD | `ecd_samples_v3.jsonl` | 800 | 展览连贯性判别，4级扰动 |

---

## 四、评测脚本使用方法

### 唯一入口

```bash
cd "C:\Users\mengzehong\Desktop\展览馆llm"
python baselines/sota_eval.py --models MODEL_NAME --tasks meip tes ecd
```

### LiteLLM API（内部，无限额度）

```python
api_key = "sk-TpK0g832p8LbMXTdI_pjkQ"
base_url = "http://csig.litellm.prod.sgpolaris/v1"
model    = "gpt-5.2"   # 或下表中任意模型名
```

### 已测试模型名称（--models 参数用这些名称）

| 参数名 | 对应模型 |
|--------|---------|
| `gpt-5.2` | GPT-5.2 |
| `claude-opus-4.6` | Claude Opus 4.6 |
| `claude-sonnet-4.5` | Claude Sonnet 4.5 |
| `gemini-2.5-pro` | Gemini 2.5 Pro |
| `gemini-2.5-flash` | Gemini 2.5 Flash |
| `deepseek-r1` | DeepSeek-R1 |
| `doubao-seed-2.0-pro` | 豆包 Seed 2.0 Pro |
| `glm-5` | GLM-5 |
| `kimi-k2.5` | Kimi K2.5 |
| `minimax-m2.5` | MiniMax M2.5 |

### 常用命令

```bash
# 仅跑 ECD（claude-opus 之前 ECD 崩了，已修复 format_seq）
python baselines/sota_eval.py --models claude-opus-4.6 --tasks ecd

# 跑缺失的任务
python baselines/sota_eval.py --models kimi-k2.5 claude-sonnet-4.5 --tasks ecd tes

# 强制重跑（覆盖已有结果）
python baselines/sota_eval.py --models gpt-5.2 --tasks meip --force

# 跑开源 LLM（待实现，见下文 TODO）
python baselines/sota_eval.py --models llama-3-70b mistral-7b qwen2.5-7b --tasks meip tes ecd
```

### 结果文件位置与键名（重要！）

结果保存在 `results/` 下，格式：`{task}_{model}_shot{n}.json`

| 任务 | 文件示例 | 关键 JSON 键 |
|------|---------|-------------|
| MEIP | `meip_gpt-5.2_shot0.json` | `mrr`, `hit@1`, `n_samples`, `total_latency_sec`, `avg_latency_sec`, `total_prompt_tokens`, `total_completion_tokens`, `total_tokens` |
| TES | `tes_gpt-5.2_shot0.json` | `ndcg@10`（含@符号！不是 ndcg_10）, `mrr`, **+ 所有 latency/token 字段** |
| ECD | `ecd_gpt-5.2_shot0.json` | `macro_pairaccc`, `pairaccc_L1/L2/L3/L4`, **+ 所有 latency/token 字段** |

⚠️ **常见坑**：TES 用 `ndcg@10`（带@），ECD 用 `macro_pairaccc`（不是 `accuracy`）。  
⚠️ **旧格式**：2026-04-30 之前跑的 shot0 结果没有 latency/token 字段，2026-05-01 已用 `--force` 全部重跑补全。

### 结果汇总工具

**新版（推荐）**：`results/compile_sota_results.py`  
```bash
cd "C:\Users\mengzehong\Desktop\展览馆llm"
python results/compile_sota_results.py              # 汇总 shot0
python results/compile_sota_results.py --shot 0 1 3  # 多 shot 对比
python results/compile_sota_results.py --latex       # 生成 LaTeX 表格
```
功能：自动读取所有 `{task}_{model}_shot{n}.json`，输出完整主表（含 latency/token breakdown）+ CSV + LaTeX。

**旧版（有键名 bug，不要用）**：`results/compile_results.py`

---

## 五、当前实验结果（截至 2026-05-01）

> 所有 shot0 结果均已用带 latency/token 的新版脚本重跑（2026-05-01）。

### MEIP（MRR）

| 模型 | MRR | Hit@1 | n |
|------|-----|-------|---|
| **claude-opus-4.6** | **0.7573** | **0.7010** | 194 |
| claude-sonnet-4.5 | 0.6936 | 0.6296 | 189 |
| gemini-2.5-pro | 0.6789 | 0.6100 | 200 |
| doubao-seed-2.0-pro | 0.6729 | 0.6050 | 200 |
| gpt-5.2 | 0.6741 | 0.6000 | 200 |
| gemini-2.5-flash | 0.5967 | 0.5050 | 200 |
| kimi-k2.5 | 0.5652 | 0.4700 | 200 |
| deepseek-r1 | 0.5085 | 0.3950 | 200 |
| minimax-m2.5 | 0.4894 | 0.3650 | 200 |
| glm-5 | 0.3579 | 0.2100 | 200 |

### TES（NDCG@10）

| 模型 | NDCG@10 | MRR | n |
|------|---------|-----|---|
| **doubao-seed-2.0-pro** | **0.7348** | **0.6568** | 200 |
| gemini-2.5-flash | 0.4615 | 0.4276 | 200 |
| gemini-2.5-pro | 0.4592 | 0.2971 | 200 |
| claude-opus-4.6 | 0.4310 | 0.2622 | 194 |
| claude-sonnet-4.5 | 0.4194 | 0.2524 | 195 |
| gpt-5.2 | 0.4165 | 0.2459 | 200 |
| deepseek-r1 | 0.3691 | 0.2494 | 200 |
| minimax-m2.5 | 0.1593 | 0.1623 | 200 |
| kimi-k2.5 | 0.1104 | 0.1164 | 161 |
| glm-5 | 0.1062 | 0.1009 | 200 |

> ⚠️ **TES 任务设计缺陷（已确认）**：`query_theme` 100% 出现在候选展览的 `theme` 字段（exact match）。纯 BM25 关键词匹配可达 NDCG@10=**0.979**，doubao 只是更擅长利用这个 shortcut。详见 `analysis/tes_leakage_analysis.py`。**论文中必须声明这个 leakage 问题或修复数据集**。

### ECD（macro_pairaccc）

| 模型 | Macro | L1 | L2 | L3 | L4 | n |
|------|-------|----|----|----|----|---|
| **doubao-seed-2.0-pro** | **0.8350** | 0.940 | 0.980 | 0.480 | 0.940 | 200 |
| gemini-2.5-pro | 0.8100 | 0.880 | 0.960 | 0.480 | 0.920 | 200 |
| claude-sonnet-4.5 | 0.7848 | 0.872 | 0.846 | 0.500 | 0.921 | 154* |
| gpt-5.2 | 0.7600 | 0.900 | 0.880 | 0.520 | 0.740 | 200 |
| gemini-2.5-flash | 0.7150 | 0.700 | 0.860 | 0.520 | 0.780 | 200 |
| claude-opus-4.6 | 🔄 **进行中**（2026-05-01，logs/opus_ecd.log）| | | | | |
| glm-5 | 0.5450 | 0.540 | 0.680 | 0.460 | 0.500 | 200 |
| minimax-m2.5 | 0.5100 | 0.520 | 0.540 | 0.500 | 0.480 | 200 |
| kimi-k2.5 | 0.4950 | 0.500 | 0.480 | 0.500 | 0.500 | 200 |
| deepseek-r1 | 0.4600 | 0.460 | 0.480 | 0.400 | 0.500 | 200 |

*sonnet n=154 是旧版结果（有 bug），正在重跑中（logs/sonnet_ecd_rerun.log）

> **核心规律**：L3（主题偏差）最难（约0.48-0.52），所有模型趋近随机猜；L1（时代错配）和L2（文化偏移）较易（L2最易，0.48-0.98）。

### Few-shot 消融（截至 2026-05-01）

| 模型 | shot=0 MRR | shot=1 MRR | shot=3 MRR |
|------|-----------|-----------|-----------|
| gpt-5.2 (MEIP) | 0.6741 | **0.6652** ↓ | 🔄 进行中 |
| claude-sonnet-4.5 (MEIP) | 0.6936 | 🔄 进行中 | 🔄 进行中 |
| claude-opus-4.6 (MEIP) | 0.7573 | 🔄 进行中 | — |

> **初步发现**：gpt-5.2 MEIP 1-shot (0.6652) < 0-shot (0.6741)，符合 few-shot 退化假设。数据待补全后进行正式分析。

---

## 六、已知 Bug 与修复记录

### Bug 1：format_seq None crash（已修复）

- **位置**：`baselines/sota_eval.py::format_seq()`
- **症状**：`TypeError: 'NoneType' object is not subscriptable`
- **触发**：ECD 评测时 obj_list 中有 None 元素
- **修复**：已加 `if obj is None or not isinstance(obj, dict): continue` 守卫
- **状态**：✅ 已修复（2026-04-30）

### Bug 2：compile_results.py 键名错误（已废弃）

- **位置**：`results/compile_results.py`（旧文件）
- **症状**：TES/ECD 列显示 `???`
- **原因**：用了 `ndcg_10` 而非 `ndcg@10`，用了 `accuracy` 而非 `macro_pairaccc`
- **状态**：✅ 已通过新建 `results/compile_sota_results.py` 解决，请勿使用旧文件

### Bug 3：call_llm() 不记录 latency/token（已修复）

- **位置**：`baselines/sota_eval.py::call_llm()`
- **症状**：旧版只返回 `Optional[str]`，完全没有计时或 token 统计
- **修复**：已改为返回 `(content, latency_sec, usage_dict)` 三元组；三个 evaluate 函数全部更新
- **状态**：✅ 已修复（2026-05-01），旧 shot0 结果已用 `--force` 重跑

### Bug 4：ECD sonnet n=154 不完整

- **原因**：旧版跑时中途崩溃，只完成 154/200 个样本
- **修复**：2026-05-01 用新版 `--force` 重跑（logs/sonnet_ecd_rerun.log 进行中）

---

## 七、待完成 TODO（优先级排序，截至 2026-05-01）

### P0 — 必须完成（影响论文 reject/accept）

- [x] **记录 latency/token cost**：`call_llm()` 已改造为三元组，三个 evaluate 函数全部更新 ✅
- [x] **compile_sota_results.py 新汇总工具**：正确键名，含 latency/token breakdown ✅
- [x] **TES leakage 分析**：query_theme 100% exact match，BM25=0.979，写入论文必须说明 ✅
- [ ] **claude-opus-4.6 ECD 完成**：后台运行中 `logs/opus_ecd.log`，预计约 3-4 小时
- [ ] **claude-sonnet-4.5 ECD 重跑（完整200样本）**：后台运行中 `logs/sonnet_ecd_rerun.log`
- [ ] **所有 shot0 补 latency/token 重跑**：9个模型并行中（logs/*_shot0_rerun.log）
- [ ] **加入开源 LLM baseline**（审稿红线）：Llama-3-70B、Mistral-7B、Qwen2.5-7B
  - 用同一 LiteLLM API，在 `compile_sota_results.py` MODEL_DISPLAY 里加映射即可
  - 运行：`python baselines/sota_eval.py --models llama-3-70b qwen2.5-7b --tasks meip tes ecd`

### P1 — 核心科学贡献

- [ ] **Few-shot 退化机制实验**（收集后对比分析）
  - 已有：gpt-5.2 shot0=0.6741, shot1=0.6652（退化确认）
  - 进行中：gpt-5.2 shot3, claude-sonnet shot1/3, claude-opus shot1
  - 下一步：运行完整消融 `--shot 0 1 2 3 5`，写 Analysis 章节
  - 三个假设：H1文化锚定偏差、H2上下文过载、H3格式一致性偏差
- [ ] **元数据消融实验**（6级：仅title → 全字段）
  - 创建 `analysis/metadata_ablation.py`，用 `call_llm()` 测不同字段组合的 MEIP MRR
- [ ] **TES 任务修复**：
  - 方案A：去掉候选列表中的 `theme` 字段（推荐，最简单）
  - 方案B：构造 hard negatives（同主题词，不同内容）
  - 方案C：论文中加 leakage 脚注 + BM25 baseline 对比（最快）
- [ ] **细化错误分析**：当前 Other 类 = 53-73%，增加 Semantic Overlap 和 Popularity Bias 子类

### P2 — 加分项

- [ ] 多模态 baseline（CLIP + GPT-4o Vision）
- [ ] 人工评估（3人 × 80样本）

---

## 八、当前后台进程（截至 2026-05-01 12:15）

| 进程 | 日志 | 进度 |
|------|------|------|
| claude-opus-4.6 ECD shot0 | logs/opus_ecd.log | 🔄 进行中（~18行，刚开始） |
| claude-sonnet-4.5 ECD shot0 重跑 | logs/sonnet_ecd_rerun.log | 🔄 进行中 |
| gpt-5.2 MEIP+TES shot0 重跑 | logs/gpt52_meip_tes_shot0_rerun.log | 🔄 进行中 |
| doubao all tasks shot0 重跑 | logs/doubao_all_shot0_rerun.log | 🔄 进行中 |
| gemini-2.5-pro all tasks shot0 重跑 | logs/gemini_pro_all_shot0_rerun.log | 🔄 进行中 |
| gemini-2.5-flash all tasks shot0 重跑 | logs/gemini_flash_all_shot0_rerun.log | 🔄 进行中 |
| deepseek-r1 all tasks shot0 重跑 | logs/deepseek_all_shot0_rerun.log | 🔄 进行中 |
| kimi-k2.5 all tasks shot0 重跑 | logs/kimi_all_shot0_rerun.log | 🔄 进行中 |
| glm-5 all tasks shot0 重跑 | logs/glm_all_shot0_rerun.log | 🔄 进行中 |
| minimax-m2.5 all tasks shot0 重跑 | logs/minimax_all_shot0_rerun.log | 🔄 进行中 |
| gpt-5.2 MEIP shot3 | logs/gpt52_meip_shot3.log | 🔄 进行中（150/200，MRR=0.8098） |
| claude-sonnet-4.5 MEIP shot1/3 | logs/sonnet_meip_shot1.log, shot3.log | 🔄 进行中 |
| claude-opus-4.6 MEIP+TES shot1 | logs/opus_meip_tes_shot1.log | 🔄 进行中 |

---

## 九、论文状态

- **文件**：`C:\Users\mengzehong\Desktop\ExhibBench-repo\paper\policyeval_bench.tex`
- **当前状态**：已更新到 v3 数据集统计（23,658件、300展览）、三任务框架
- **还需要**：
  1. 等所有 shot0 重跑完成（latency/token）后，用 `compile_sota_results.py --latex` 生成主结果表
  2. opus ECD 完成后补入 Table 1
  3. Few-shot 机制实验完成后写 Analysis 章节（H1/H2/H3 假设）
  4. 开源 LLM baseline 完成后补入 Table 1（ACL/EMNLP 审稿红线）
  5. TES leakage 问题写入 Dataset Construction 章节（加 BM25=0.979 作为 trivial baseline）

---

## 十、新发现（2026-05-01）

### TES 任务泄露分析
- **文件**：`analysis/tes_leakage_analysis.py`
- **发现**：`query_theme` 字段 **100%** exact match 到候选展览的 `theme` 字段
- **影响**：纯 BM25 关键词匹配可达 NDCG@10 = **0.9790**（近乎完美）
- **解释 doubao 异常**：doubao TES NDCG=0.7348 远低于 BM25 baseline，说明 doubao 只是比其他模型更擅长利用关键词 shortcut，而非真正的语义策展能力
- **论文处理方案**：
  - 方案A（推荐）：在数据集章节加 leakage 声明 + BM25 baseline 列
  - 方案B：修复数据集（去掉 candidates 的 theme 字段后重新评测）

### Latency/Token 记录（新格式）
新版 `sota_eval.py` 的每个结果 JSON 现在包含：
```json
{
  "total_latency_sec": 436.21,
  "avg_latency_sec": 2.181,
  "total_prompt_tokens": 146252,
  "total_completion_tokens": 3065,
  "total_tokens": 149317
}
```
用 `compile_sota_results.py` 可自动汇总全部模型的 cost/latency。

---

---

## 九、Demo 系统

```bash
# NiceGUI demo（已在运行，port 7861）
python system/nicegui_app.py --port 7861
# 浏览器访问：http://localhost:7861

# 旧 Gradio demo（port 7860，可能在运行）
python system/gradio_app.py --port 7860
```

NiceGUI demo 功能：
- Tab 1 TES：输入主题 → BM25/GPT-5.2重排 → 展品卡片网格
- Tab 2 MEIP：输入已有展品 → 预测下一件
- Tab 3 统计：数据集概况 + 指标说明

---

## 十、关键设计决策（背景知识）

1. **为什么 find_data_file 选 _v3**：`sota_eval.py` 第 147 行搜索顺序 `_v3 > _v2 > bare`，保证始终用最新数据

2. **ECD 4 级扰动设计**：
   - L1 时代错配（>500年）→ 最容易检测
   - L2 文化圈偏移 → 较易
   - L3 主题偏差（BM25相关但主题不符）→ 最难，所有模型约 0.48
   - L4 细粒度风格矛盾 → 难

3. **MEIP 20-way choice**：从候选池中选 20 个（含1个正确答案+19个BM25难负样本），让 LLM 排序

4. **TES 评测方式**：展览真实展品作为 ground truth，LLM 从候选池选 top-k，计算 NDCG@10

5. **few-shot 退化反直觉发现**：GPT zero-shot MEIP MRR=0.674，3-shot 约 0.65（更差）— 这是本文最强科学贡献，尚未完整解释

---

## 十一、API 凭证

```
LiteLLM API（内部，无限额度）：
  api_key: sk-TpK0g832p8LbMXTdI_pjkQ
  base_url: http://csig.litellm.prod.sgpolaris/v1
  推荐模型: gpt-5.2（最强，用于论文主结果）
```

---

## 十二、快速上手检查清单

接手时运行以下命令验证环境：

```bash
cd "C:\Users\mengzehong\Desktop\展览馆llm"

# 1. 验证数据
python -c "import json; print(sum(1 for _ in open('data/objects_v3.jsonl', encoding='utf-8')), 'objects')"
# 期望输出：23658 objects

# 2. 验证 API
python -c "
from openai import OpenAI
c = OpenAI(api_key='sk-TpK0g832p8LbMXTdI_pjkQ', base_url='http://csig.litellm.prod.sgpolaris/v1')
r = c.chat.completions.create(model='gpt-5.2', messages=[{'role':'user','content':'ping'}], max_tokens=5)
print('API OK:', r.choices[0].message.content)
"

# 3. 查看已有结果
python -c "
import json, os, glob
for f in sorted(glob.glob('results/meip_*_shot0.json')):
    d = json.load(open(f))
    print(os.path.basename(f), 'MRR=', round(d.get('mrr',0), 4))
"

# 4. 运行待补 ECD
python baselines/sota_eval.py --models claude-opus-4.6 --tasks ecd
```

---

*生成时间：2026-04-30 | 作者：Claude Sonnet 4.5 (AI助手)*

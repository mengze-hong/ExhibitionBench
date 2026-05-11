# 展览馆LLM 项目 - 前端界面文件搜索报告

## 📁 目录结构（顶层 + 1级）
```
展览馆llm/
├── analysis/              # 数据分析脚本
├── baselines/             # 模型baseline实现
├── benchmark/             # 评测框架
├── data/                  # 数据文件夹
├── logs/                  # 日志文件夹
├── paper/                 # 论文相关
├── results/               # 结果输出
├── system/                # 前端系统（核心）
├── *.py                   # 顶层脚本
└── requirements.txt
```

---

## 🎯 前端界面相关文件清单

### 1️⃣ **system/app.py** （🔴 主要）
- **行数**: 469 行
- **框架**: Gradio（基于 Python 的快速 Web UI 库）
- **功能**:
  - **TES 模式**（主题策展）: 输入展览主题 → 从 1500+ 件藏品池用 BM25 检索 → 可选 GPT-5.2 LLM 重排 → 网格展示推荐展品
  - **MEIP 模式**（展品补全）: 输入已有展品描述 → 预测最佳补充展品
  - **人在回路反馈**: 用户可点赞/踩推荐结果 → 反馈记录到 `logs/feedback.jsonl`
  - **数据统计面板**: 展示数据来源、展品数量等

- **主要 UI 组件**:
  - 3 个 Tab: TES、MEIP、数据集概览
  - 搜索输入框、滑块、勾选框
  - HTML 网格卡片展示（每张卡片含图片、标题、文化、年代、媒介）
  - 投票按钮（👍👎）
  - 示例按钮组

- **运行方式**: 
  ```bash
  python system/app.py --port 7860 [--share]
  ```

---

### 2️⃣ **system/nicegui_app.py** （🟠 备选方案）
- **行数**: 453 行
- **框架**: NiceGUI（基于 FastAPI + Vue.js WebSocket，比 Gradio 轻量）
- **功能**: 与 `app.py` 功能相同（TES + MEIP + 统计），但更轻量，不会卡死
- **特点**:
  - 使用自定义 CSS（渐变色背景、卡片悬停效果）
  - WebSocket 实时更新，更流畅
  - Markdown 表格展示数据统计

- **主要 UI 组件**:
  - 3 个 Tab: TES、MEIP、数据统计
  - 输入框、数字滑块、勾选框
  - 卡片网格（对象卡片），带图片、标题、元数据
  - 反馈选择、快速示例按钮
  - 统计卡片网格（6 个 KPI）

- **运行方式**:
  ```bash
  python system/nicegui_app.py --port 7861
  ```

---

### 3️⃣ **benchmark/annotation_ui.html** （🔴 标注工具）
- **行数**: 5735 行（包含大量内联 JavaScript）
- **框式**: 纯 HTML + CSS + JavaScript（浏览器前端）
- **功能**: Human A/B 对比标注工具
  - 展示两个展品 A/B 方案的对比
  - 用户选择更优方案或平局
  - 记录标注者的理由
  - 进度条实时显示完成进度
  - 批量导出标注结果

- **主要结构**:
  - Header（进度条、导出按钮）
  - Card（展品元数据、题目、上下文）
  - 双面板（Panel A / Panel B），各显示一件展品的详细信息
  - 投票区（A/B/TIE 按钮、理由文本框）
  - 快捷键支持

- **关键特性**:
  - 本地存储 (localStorage) 自动保存进度
  - 支持键盘快捷键 (1=A, 2=B, 3=Tie)
  - 实时计算完成百分比
  - 导出为 JSON/CSV

---

## 📊 Python 文件中的关键词匹配

### ✅ 找到的 Python 文件（含 gradio/streamlit/flask/demo/app.py）
1. `system/app.py` → **gradio** ✓
2. `system/nicegui_app.py` → nicegui（相关框架）
3. `run_week2_demo.py` → demo ✓
4. `collect_exhibitions.py` → demo（collect 场景演示）
5. `collect_expand_v3.py` → demo
6. `collect_v3_fixed.py` → demo

---

## 📈 统计摘要

| 指标 | 值 |
|------|-----|
| **前端相关文件总数** | 3 个 |
| **总行数（核心前端）** | 6,657 行 |
| **Gradio 应用** | system/app.py |
| **NiceGUI 应用** | system/nicegui_app.py |
| **HTML 标注工具** | benchmark/annotation_ui.html |
| **无 HTML/JS/TS 文件** | ✓（HTML 是标注工具，不是静态网页） |

---

## 🔧 数据流向

```
用户输入 (主题/描述)
    ↓
[BM25 检索] (from rank_bm25)
    ↓
[可选 LLM 重排] (GPT-5.2 via internal API)
    ↓
[Gradio/NiceGUI] 渲染网格卡片
    ↓
用户反馈 (点赞/踩)
    ↓
logs/feedback.jsonl 记录
```

---

## 💡 重点发现

1. **双应用架构**:
   - `app.py` (Gradio) - 官方主方案
   - `nicegui_app.py` - 备选轻量方案（更流畅）

2. **核心功能**:
   - TES（主题策展）: 批量检索推荐
   - MEIP（展品补全）: 单项预测
   - 人在回路: 用户反馈记录

3. **后端依赖**:
   - `data/objects.jsonl` - 1500+ 件展品库
   - `data/exhibitions.jsonl` - 展览元数据
   - 可选 LLM 重排（需 API Key）

4. **HTML 标注工具**:
   - 用于人工评测，非用户端应用
   - 支持 A/B 对比标注
   - 本地存储 + 导出功能


---

## 🗂️ 文件详细对比表

| 文件 | 类型 | 行数 | 框架 | 主要功能 | 用途 |
|------|------|------|------|--------|------|
| `system/app.py` | Python | 469 | **Gradio** | TES 主题策展、MEIP 展品补全、反馈记录 | 用户端 Web UI（官方） |
| `system/nicegui_app.py` | Python | 453 | **NiceGUI** | TES 主题策展、MEIP 展品补全、数据统计 | 用户端 Web UI（备选，轻量） |
| `benchmark/annotation_ui.html` | HTML+JS | 5735 | 原生 JS | Human A/B 对比标注、进度跟踪、导出 | 评测工具（标注员用） |

---

## 📝 快速使用指南

### 启动 Gradio 演示
```bash
cd C:\Users\mengzehong\Desktop\展览馆llm
python system/app.py --port 7860
# 访问: http://localhost:7860
```

### 启动 NiceGUI 演示（推荐，更流畅）
```bash
python system/nicegui_app.py --port 7861
# 访问: http://localhost:7861
```

### 打开标注工具
```bash
# 使用浏览器直接打开
benchmark/annotation_ui.html
```

---

## 🔍 关键配置项

### system/app.py
- **Port 默认**: 7860
- **API Key**: 内置 `sk-TpK0g832p8LbMXTdI_pjkQ`（LiteLLM）
- **模型**: `gpt-5.2`
- **数据目录**: `data/objects.jsonl`, `data/exhibitions.jsonl`
- **反馈日志**: `logs/feedback.jsonl`

### system/nicegui_app.py
- **Port 默认**: 7861
- **API Key**: 内置 `sk-TpK0g832p8LbMXTdI_pjkQ`（LiteLLM）
- **模型**: `gpt-5.2`
- **反馈日志**: `logs/feedback_nicegui.jsonl`
- **主题**: Soft（Gradio 主题库）

### benchmark/annotation_ui.html
- **存储**: LocalStorage（浏览器本地）
- **导出格式**: JSON + CSV
- **快捷键**: 1=选A、2=选B、3=平局

---

## 🎯 核心业务流程

### TES（Theme-based Exhibition Selection）
```
用户输入主题
  ↓
BM25 检索前 50 个候选
  ↓
[可选] GPT-5.2 重排前 K 个
  ↓
HTML 卡片网格展示
  ↓
用户反馈（点赞/踩）→ feedback.jsonl
```

### MEIP（Masked Exhibition Item Prediction）
```
用户输入已有展品描述
  ↓
组合为查询，BM25 检索前 50 个
  ↓
GPT-5.2 预测最佳补充展品
  ↓
展示预测理由 + 推荐展品详情
```

---

## 📦 依赖库

- **Gradio**: `gr`
- **NiceGUI**: `nicegui`, `ui`
- **检索**: `rank_bm25`
- **LLM**: `openai` (custom base_url)
- **数据**: `json`, `pathlib`


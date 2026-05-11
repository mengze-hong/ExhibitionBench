#!/usr/bin/env python3
"""
ExhibitionBench Human A/B Preference Annotation Generator
从 MEIP 预测结果中抽取 A/B 对比样本，生成标注 JSONL 和自包含 HTML 界面。

用法：
    python benchmark/human_ab_generator.py

输出：
    benchmark/annotation_tasks.jsonl   — 标注任务（JSONL）
    benchmark/annotation_ui.html       — 标注员界面（离线可用）
"""

import json
import random
from pathlib import Path
from typing import Optional

# ── 路径常量 ────────────────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\mengzehong\Desktop\展览馆llm")
BENCHMARK   = BASE / "benchmark"
DATA        = BASE / "data"
RESULTS_DIR = BASE / "results" / "cultural_bias"

MEIP_SAMPLES = DATA / "meip_samples.jsonl"
OBJECTS_FILE  = DATA / "objects.jsonl"

SAMPLE_SIZE  = 60
RANDOM_SEED  = 42

# 模型名 → 结果文件名（不含路径）
MODEL_FILES = {
    "gpt-5.2":        "meip_cultural_gpt-5.2_n200.jsonl",
    "sbert":          "meip_cultural_sbert_n200.jsonl",
    "gemini-2.5-pro": "meip_cultural_gemini-2.5-pro_n200.jsonl",
}

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    """加载 JSONL，文件不存在时返回空列表。"""
    if not path.exists():
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSON 解析失败 ({path.name}): {e}")
    return items


def load_model_preds(model_name: str) -> dict:
    """加载单个模型的预测，返回 {meip_id: record}。"""
    filename = MODEL_FILES.get(model_name)
    if filename is None:
        print(f"[WARN] 未知模型: {model_name}")
        return {}
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"[WARN] 预测文件不存在，跳过: {path}")
        return {}
    preds = load_jsonl(path)
    result = {p["id"]: p for p in preds}
    print(f"[OK]  加载 {model_name}: {len(result)} 条预测")
    return result


def build_objects_index(path: Path) -> dict:
    """从 objects.jsonl 建立 {object_id: object_record} 索引。"""
    items = load_jsonl(path)
    return {obj["id"]: obj for obj in items}


def get_object_info(obj_id: str, objects: dict) -> Optional[dict]:
    """返回展品的标注所需字段；找不到则返回 None。"""
    obj = objects.get(obj_id)
    if obj is None:
        return None
    desc = obj.get("description", "") or ""
    return {
        "pred_id":            obj_id,
        "title":              obj.get("title", "Unknown"),
        "culture":            obj.get("culture", ""),
        "date":               obj.get("date", ""),
        "medium":             obj.get("medium", ""),
        "description_snippet": desc[:150],
        "image_url":          obj.get("image_url", ""),
    }


def build_context_summary(context_items: list, max_items: int = 3) -> str:
    """取前 max_items 件上下文展品，拼成一句中文摘要。"""
    parts = []
    for item in context_items[:max_items]:
        title   = item.get("title", "Unknown")
        culture = item.get("culture", "")
        date    = item.get("date", "")
        suffix  = f"（{culture}, {date}）" if (culture or date) else ""
        parts.append(f"{title}{suffix}")
    return "；".join(parts) if parts else "（无上下文）"


# ── 构建对比组 ────────────────────────────────────────────────────────────────

def build_comparison_group(
    group_name: str,
    model_a:    str,
    model_b:    str,          # 当 is_gold_b=True 时本参数仅作标签
    preds_a:    dict,
    preds_b:    dict,          # 当 is_gold_b=True 时忽略（用 gold_id）
    meip_samples: dict,
    objects:    dict,
    rng:        random.Random,
    is_gold_b:  bool = False,
) -> list:
    """
    构建一个对比组的标注任务列表。
    is_gold_b=True 时 system_b 使用人类策展的 gold_id。
    """
    if is_gold_b:
        common_ids = sorted(set(preds_a.keys()) & set(meip_samples.keys()))
    else:
        if not preds_b:
            print(f"[SKIP] {group_name}：模型 {model_b} 无预测，跳过此对比组")
            return []
        common_ids = sorted(
            set(preds_a.keys()) & set(preds_b.keys()) & set(meip_samples.keys())
        )

    if not common_ids:
        print(f"[SKIP] {group_name}：没有公共查询，跳过")
        return []

    sampled = rng.sample(common_ids, min(SAMPLE_SIZE, len(common_ids)))
    print(f"[INFO] {group_name}: 公共查询 {len(common_ids)} 条，抽样 {len(sampled)} 条")

    tasks = []
    for meip_id in sampled:
        sample = meip_samples[meip_id]
        pred_a_rec = preds_a[meip_id]

        info_a = get_object_info(pred_a_rec["pred_id"], objects)
        if info_a is None:
            continue

        if is_gold_b:
            gold_id = sample["gold_id"]
            info_b  = get_object_info(gold_id, objects)
        else:
            pred_b_rec = preds_b[meip_id]
            info_b     = get_object_info(pred_b_rec["pred_id"], objects)

        if info_b is None:
            continue

        info_a["model"] = model_a
        info_b["model"] = "human-curator" if is_gold_b else model_b

        # 随机打乱 A/B 顺序，防止位置偏见
        if rng.random() < 0.5:
            sys_a, sys_b = info_a, info_b
        else:
            sys_a, sys_b = info_b, info_a

        tasks.append({
            "task_id":          "",   # 稍后统一编号
            "comparison":       group_name,
            "meip_id":          meip_id,
            "exhibition_theme": sample.get("exhibition_theme", ""),
            "context_summary":  build_context_summary(sample.get("context", [])),
            "system_a":         sys_a,
            "system_b":         sys_b,
            "gold_id":          sample["gold_id"],
            "annotator_choice": None,
            "annotator_reason": None,
        })

    return tasks


# ── HTML 生成 ─────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ExhibitionBench — Human A/B Annotation</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', 'PingFang SC', system-ui, sans-serif;
       background: #f0f2f8; color: #222; line-height: 1.5; }

/* ── Header ── */
#header {
  background: #1a1a2e; color: #fff; padding: 12px 24px;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 100;
  box-shadow: 0 2px 10px rgba(0,0,0,0.4);
}
#header h1 { font-size: 17px; font-weight: 700; white-space: nowrap; }
#pb-wrap { flex: 1; margin: 0 20px; }
#pb { height: 8px; background: #3a3a5e; border-radius: 4px; overflow: hidden; }
#pb-fill { height: 100%; background: #4ecca3; border-radius: 4px; transition: width .3s; width: 0%; }
#pb-text { font-size: 12px; color: #aaa; margin-top: 3px; text-align: center; }
#export-hdr-btn {
  padding: 8px 18px; background: #4ecca3; color: #1a1a2e;
  border: none; border-radius: 6px; cursor: pointer; font-weight: 700; font-size: 13px;
  white-space: nowrap;
}
#export-hdr-btn:hover { background: #3ab88d; }

/* ── Main ── */
#main { max-width: 1100px; margin: 20px auto; padding: 0 14px; }

/* ── Card ── */
.card {
  background: #fff; border-radius: 12px;
  box-shadow: 0 2px 14px rgba(0,0,0,0.09); overflow: hidden;
}

.card-meta {
  background: #f7f8ff; padding: 14px 22px; border-bottom: 1px solid #e6e8f5;
}
.tag {
  display: inline-block; background: #e3efff; color: #1565c0;
  padding: 2px 10px; border-radius: 10px; font-size: 12px; font-weight: 600;
  margin-bottom: 6px;
}
.task-num { float: right; font-size: 12px; color: #aaa; }
.theme { font-size: 19px; font-weight: 800; color: #1a1a2e; margin-bottom: 6px; }
.ctx-lbl { font-size: 11px; color: #999; font-weight: 700; text-transform: uppercase;
           letter-spacing: .6px; }
.ctx-txt { font-size: 13px; color: #555; margin-top: 2px; }

/* ── Two panels ── */
.panels { display: grid; grid-template-columns: 1fr 1fr; }
.panel { padding: 18px 22px; }
.panel:first-child { border-right: 1px solid #f0f0f5; }

.panel-hdr {
  display: inline-block; font-size: 11px; font-weight: 800;
  text-transform: uppercase; letter-spacing: 1px; color: #fff;
  padding: 3px 12px; border-radius: 10px; margin-bottom: 10px;
}
.panel-a .panel-hdr { background: #1565c0; }
.panel-b .panel-hdr { background: #b71c1c; }

.obj-title { font-size: 15px; font-weight: 700; color: #1a1a2e; margin-bottom: 5px; }
.obj-meta { font-size: 12px; margin-bottom: 8px; }
.obj-meta .culture { color: #1565c0; margin-right: 10px; }
.obj-meta .date    { color: #666;    margin-right: 10px; }
.obj-meta .medium  { color: #888; font-style: italic; }
.obj-desc {
  font-size: 13px; color: #555; background: #fafafa;
  padding: 8px 12px; border-radius: 6px; border-left: 3px solid #ddd;
}

/* ── Vote ── */
.vote {
  padding: 14px 22px; background: #f7f8ff; border-top: 1px solid #e6e8f5;
}
.vote-lbl { font-size: 13px; font-weight: 600; color: #444; margin-bottom: 8px; }
.vote-btns { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.vbtn {
  padding: 8px 22px; border: 2px solid #d0d4e8; background: #fff;
  border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600;
  transition: all .12s;
}
.vbtn:hover { border-color: #aaa; background: #f5f5f5; }
.vbtn.sel-a   { border-color: #1565c0; background: #e3efff; color: #1565c0; }
.vbtn.sel-tie { border-color: #2e7d32; background: #e8f5e9; color: #2e7d32; }
.vbtn.sel-b   { border-color: #b71c1c; background: #ffebee; color: #b71c1c; }

.reason {
  width: 100%; padding: 8px 12px; border: 1px solid #d0d4e8; border-radius: 6px;
  font-size: 13px; font-family: inherit; resize: vertical; min-height: 56px;
  background: #fff;
}
.reason:focus { outline: none; border-color: #1565c0;
                box-shadow: 0 0 0 2px rgba(21,101,192,.12); }
.hint { font-size: 11px; color: #bbb; margin-top: 4px; }

/* ── Nav ── */
#nav { display: flex; align-items: center; justify-content: space-between;
       padding: 14px 0; }
.nav-btn {
  padding: 9px 26px; background: #1a1a2e; color: #fff;
  border: none; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600;
  transition: background .12s;
}
.nav-btn:hover  { background: #2d2d4e; }
.nav-btn:disabled { background: #ccc; cursor: not-allowed; }
#nav-ctr { font-size: 14px; color: #666; }

/* ── Summary ── */
#summary {
  background: #fff; border-radius: 12px; padding: 18px 22px; margin-top: 4px;
  box-shadow: 0 2px 14px rgba(0,0,0,0.09);
}
#summary h3 { font-size: 15px; font-weight: 700; margin-bottom: 12px; }
.sum-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px,1fr)); gap: 10px; }
.sum-item {
  background: #f7f8ff; padding: 12px; border-radius: 8px; text-align: center;
}
.sum-count { font-size: 26px; font-weight: 800; color: #1565c0; }
.sum-lbl   { font-size: 11px; color: #888; margin-top: 2px; }
.sum-sub   { font-size: 11px; color: #999; margin-top: 4px; }

@media (max-width: 680px) {
  .panels { grid-template-columns: 1fr; }
  .panel:first-child { border-right: none; border-bottom: 1px solid #f0f0f5; }
}
</style>
</head>
<body>

<div id="header">
  <h1>&#127981; ExhibitionBench A/B Annotation</h1>
  <div id="pb-wrap">
    <div id="pb"><div id="pb-fill"></div></div>
    <div id="pb-text">0 / 0 已标注</div>
  </div>
  <button id="export-hdr-btn" onclick="exportResults()">&#8595; 导出结果</button>
</div>

<div id="main">
  <div class="card" id="task-card"></div>

  <div id="nav">
    <button class="nav-btn" id="prev-btn" onclick="go(-1)" disabled>&#8592; 上一条</button>
    <span id="nav-ctr">— / —</span>
    <button class="nav-btn" id="next-btn" onclick="go(1)">下一条 &#8594;</button>
  </div>

  <div id="summary">
    <h3>&#128202; 标注进度</h3>
    <div class="sum-grid" id="sum-grid"></div>
  </div>
</div>

<script>
// ── 数据 ──────────────────────────────────────────────────────────────────
const TASKS = %%TASKS_JSON%%;

const ann   = {};     // { task_id: { choice, reason } }
let   cur   = 0;

// ── 初始化 ────────────────────────────────────────────────────────────────
function init() {
  try {
    const saved = localStorage.getItem('exhbench_ann');
    if (saved) Object.assign(ann, JSON.parse(saved));
  } catch(e) {}
  render();
  updateSummary();
}

// ── 渲染当前任务 ──────────────────────────────────────────────────────────
function render() {
  const t = TASKS[cur];
  if (!t) return;
  const a = ann[t.task_id] || {};
  const ch = a.choice || null;

  document.getElementById('task-card').innerHTML = `
    <div class="card-meta">
      <span class="task-num">${cur+1} / ${TASKS.length}</span>
      <div class="tag">${esc(t.comparison)}</div>
      <div class="theme">&#127912; ${esc(t.exhibition_theme)}</div>
      <div class="ctx-lbl">上下文展品</div>
      <div class="ctx-txt">${esc(t.context_summary)}</div>
    </div>

    <div class="panels">
      <div class="panel panel-a">
        <span class="panel-hdr">推荐 A</span>
        ${renderObj(t.system_a)}
      </div>
      <div class="panel panel-b">
        <span class="panel-hdr">推荐 B</span>
        ${renderObj(t.system_b)}
      </div>
    </div>

    <div class="vote">
      <div class="vote-lbl">哪件展品更适合当前展览主题？</div>
      <div class="vote-btns">
        <button class="vbtn ${ch==='A'?'sel-a':''}"   onclick="vote('A')">&#10003; A 更好</button>
        <button class="vbtn ${ch==='Tie'?'sel-tie':''}" onclick="vote('Tie')">&#8776; 差不多 (Tie)</button>
        <button class="vbtn ${ch==='B'?'sel-b':''}"   onclick="vote('B')">&#10003; B 更好</button>
      </div>
      <textarea class="reason" id="reason-ta"
        placeholder="（可选）简短说明理由，如：A 的年代与主题更吻合..."
        onchange="saveReason(this.value)">${esc(a.reason||'')}</textarea>
      <div class="hint">快捷键：A / B / T（Tie）&nbsp;|&nbsp;&#8592;&#8594; 或 P / N 切换题目</div>
    </div>
  `;

  document.getElementById('prev-btn').disabled = (cur === 0);
  document.getElementById('next-btn').disabled = (cur === TASKS.length - 1);
  document.getElementById('nav-ctr').textContent = `${cur+1} / ${TASKS.length}`;
  updateProgress();
}

function renderObj(obj) {
  const culture = obj.culture ? `<span class="culture">${esc(obj.culture)}</span>` : '';
  const date    = obj.date    ? `<span class="date">${esc(obj.date)}</span>` : '';
  const medium  = obj.medium  ? `<span class="medium">${esc(obj.medium)}</span>` : '';
  const desc    = obj.description_snippet
    ? `<div class="obj-desc">${esc(obj.description_snippet)}${obj.description_snippet.length>=150?'…':''}</div>`
    : `<div class="obj-desc" style="color:#bbb">（无描述）</div>`;
  return `
    <div class="obj-title">${esc(obj.title||'—')}</div>
    <div class="obj-meta">${culture}${date}${medium}</div>
    ${desc}
  `;
}

function esc(s) {
  return String(s||'')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── 操作 ──────────────────────────────────────────────────────────────────
function vote(choice) {
  const t = TASKS[cur];
  if (!ann[t.task_id]) ann[t.task_id] = {};
  ann[t.task_id].choice = choice;
  save();
  render();
  updateSummary();
  if (cur < TASKS.length - 1) setTimeout(() => go(1), 280);
}

function saveReason(val) {
  const t = TASKS[cur];
  if (!ann[t.task_id]) ann[t.task_id] = {};
  ann[t.task_id].reason = val;
  save();
}

function go(d) {
  cur = Math.max(0, Math.min(TASKS.length - 1, cur + d));
  render();
}

function save() {
  try { localStorage.setItem('exhbench_ann', JSON.stringify(ann)); } catch(e) {}
}

function updateProgress() {
  const done = TASKS.filter(t => ann[t.task_id] && ann[t.task_id].choice).length;
  const pct  = TASKS.length ? (done / TASKS.length * 100).toFixed(1) : 0;
  document.getElementById('pb-fill').style.width = pct + '%';
  document.getElementById('pb-text').textContent = `${done} / ${TASKS.length} 已标注`;
}

function updateSummary() {
  const groups = {};
  TASKS.forEach(t => {
    if (!groups[t.comparison]) groups[t.comparison] = {total:0,done:0,a:0,b:0,tie:0};
    const g = groups[t.comparison];
    g.total++;
    const a = ann[t.task_id];
    if (a && a.choice) {
      g.done++;
      if (a.choice==='A')   g.a++;
      else if (a.choice==='B') g.b++;
      else g.tie++;
    }
  });
  document.getElementById('sum-grid').innerHTML =
    Object.entries(groups).map(([name, g]) => `
      <div class="sum-item">
        <div class="sum-count">${g.done}/${g.total}</div>
        <div class="sum-lbl">${esc(name)}</div>
        <div class="sum-sub">A:${g.a} &nbsp; Tie:${g.tie} &nbsp; B:${g.b}</div>
      </div>
    `).join('');
}

function exportResults() {
  const lines = TASKS.map(t => {
    const a = ann[t.task_id] || {};
    return JSON.stringify({
      ...t,
      annotator_choice: a.choice || null,
      annotator_reason: a.reason || null,
    });
  });
  const blob = new Blob([lines.join('\n')], {type:'application/json'});
  const url  = URL.createObjectURL(blob);
  const el   = document.createElement('a');
  el.href = url;
  el.download = 'annotation_tasks_filled.jsonl';
  el.click();
  URL.revokeObjectURL(url);
}

// ── 快捷键 ────────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') return;
  const k = e.key.toLowerCase();
  if      (k === 'a')           vote('A');
  else if (k === 'b')           vote('B');
  else if (k === 't')           vote('Tie');
  else if (k === 'arrowright' || k === 'n') go(1);
  else if (k === 'arrowleft'  || k === 'p') go(-1);
});

init();
</script>
</body>
</html>
"""


def generate_html(tasks: list, output_path: Path) -> None:
    """将任务嵌入 HTML 模板，写出自包含的标注界面。"""
    tasks_json = json.dumps(tasks, ensure_ascii=False, indent=2)
    html = _HTML_TEMPLATE.replace("%%TASKS_JSON%%", tasks_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK]  HTML 标注界面 → {output_path}")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(RANDOM_SEED)

    # 1. 加载基础数据
    print("[INFO] 加载 MEIP 样本...")
    meip_list    = load_jsonl(MEIP_SAMPLES)
    meip_samples = {s["id"]: s for s in meip_list}
    print(f"[OK]  meip_samples: {len(meip_samples)} 条")

    print("[INFO] 加载展品库...")
    objects = build_objects_index(OBJECTS_FILE)
    print(f"[OK]  objects: {len(objects)} 件")

    # 2. 加载模型预测
    preds_gpt   = load_model_preds("gpt-5.2")
    preds_sbert = load_model_preds("sbert")
    preds_gem   = load_model_preds("gemini-2.5-pro")

    # 3. 确定"最优 LLM"
    def hit_at_1(preds: dict) -> float:
        if not preds:
            return 0.0
        return sum(1 for p in preds.values() if p.get("hit", 0) == 1) / len(preds)

    candidates_best = {
        "gpt-5.2":        (preds_gpt,  hit_at_1(preds_gpt)),
        "gemini-2.5-pro": (preds_gem,  hit_at_1(preds_gem)),
    }
    best_model, (best_preds, best_hit) = max(
        candidates_best.items(), key=lambda kv: kv[1][1]
    )
    print(f"[INFO] 最优 LLM: {best_model}  Hit@1={best_hit:.3f}")

    # 4. 构建三个对比组
    all_tasks: list = []

    # 组 1: GPT-5.2 vs SBERT（LLM vs 检索）
    g1 = build_comparison_group(
        group_name="GPT-5.2 vs SBERT",
        model_a="gpt-5.2",
        model_b="sbert",
        preds_a=preds_gpt,
        preds_b=preds_sbert,
        meip_samples=meip_samples,
        objects=objects,
        rng=rng,
    )
    all_tasks.extend(g1)

    # 组 2: GPT-5.2 vs Gemini-2.5-Pro（LLM vs LLM）
    g2 = build_comparison_group(
        group_name="GPT-5.2 vs Gemini-2.5-Pro",
        model_a="gpt-5.2",
        model_b="gemini-2.5-pro",
        preds_a=preds_gpt,
        preds_b=preds_gem,
        meip_samples=meip_samples,
        objects=objects,
        rng=rng,
    )
    all_tasks.extend(g2)

    # 组 3: 最优 LLM vs Gold（AI vs 人类策展）
    g3 = build_comparison_group(
        group_name=f"{best_model} vs Gold",
        model_a=best_model,
        model_b="human-curator",
        preds_a=best_preds,
        preds_b={},
        meip_samples=meip_samples,
        objects=objects,
        rng=rng,
        is_gold_b=True,
    )
    all_tasks.extend(g3)

    if not all_tasks:
        print("[ERROR] 没有生成任何标注任务，请检查预测文件是否存在。")
        return

    # 5. 统一编号
    for i, task in enumerate(all_tasks):
        task["task_id"] = f"ab_{i+1:04d}"

    # 6. 写出 JSONL
    BENCHMARK.mkdir(parents=True, exist_ok=True)
    out_jsonl = BENCHMARK / "annotation_tasks.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for task in all_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"[OK]  标注任务 JSONL → {out_jsonl}  ({len(all_tasks)} 条)")

    # 7. 写出 HTML
    out_html = BENCHMARK / "annotation_ui.html"
    generate_html(all_tasks, out_html)

    # 8. 打印摘要
    print("\n===== 生成摘要 =====")
    for group in ["GPT-5.2 vs SBERT", "GPT-5.2 vs Gemini-2.5-Pro", f"{best_model} vs Gold"]:
        n = sum(1 for t in all_tasks if t["comparison"] == group)
        print(f"  {group}: {n} 条")
    print(f"  合计: {len(all_tasks)} 条")
    print("=====================")


if __name__ == "__main__":
    main()

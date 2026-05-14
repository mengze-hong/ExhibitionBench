"""
collect_multi_source.py
=======================
多源博物馆数据采集脚本 — ExhibitionBench 数据扩充 (Week 1-2)

数据源:
  1. Art Institute of Chicago (AIC)  — api.artic.edu
  2. Cleveland Museum of Art         — openaccess-api.clevelandart.org
  3. Victoria & Albert Museum (V&A)  — api.vam.ac.uk

目标:
  - ≥ 200 展览（含现有 Met + Europeana）
  - ≥ 4000 件展品，覆盖 ≥ 3 个文化圈
  - 统一 schema: id/source/title/date/culture/medium/description/image_url/department/classification/period

输出:
  - data/aic_exhibitions.jsonl   + data/aic_objects.jsonl
  - data/cleveland_exhibitions.jsonl + data/cleveland_objects.jsonl
  - data/vam_exhibitions.jsonl   + data/vam_objects.jsonl
  - data/exhibitions_v2.jsonl    (合并后, 含原有数据)
  - data/objects_v2.jsonl        (合并后)

用法:
  python collect_multi_source.py [--sources aic cleveland vam] [--max-exhibitions 80]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import re
import sys
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ExhibitionBench-Research/1.0 (academic, non-commercial)"})

# ─────────────────────────────────────────────────────────────────────────────
# 通用工具
# ─────────────────────────────────────────────────────────────────────────────

def get_json(url: str, params: dict | None = None, retries: int = 3, delay: float = 1.0) -> dict | None:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5))
                log.warning(f"Rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                log.warning(f"Failed {url}: {e}")
    return None


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records)} records -> {path}")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 文化分类辅助（用于 V&A 展览主题构造）
# ─────────────────────────────────────────────────────────────────────────────

CULTURE_KEYWORDS = {
    "East Asian": ["japan", "china", "korea", "chinese", "japanese", "korean", "east asia", "ming", "qing", "tang", "song", "edo", "meiji"],
    "South Asian": ["india", "indian", "mughal", "hindu", "buddhist", "south asia", "sri lanka"],
    "Islamic": ["islamic", "ottoman", "persian", "arab", "iran", "turkey", "mughal", "mamluk"],
    "African": ["africa", "african", "egypt", "egyptian", "mali", "yoruba", "akan", "benin"],
    "Western European": ["france", "french", "italy", "italian", "dutch", "flemish", "german", "british", "english", "spain", "spanish", "renaissance", "baroque", "impressionism", "modernism"],
    "American": ["america", "american", "usa", "united states", "colonial america", "native american"],
    "Ancient": ["greek", "roman", "egypt", "mesopotamia", "ancient", "classical", "byzantine", "celtic"],
}


def infer_culture_from_text(*texts: str) -> str:
    combined = " ".join(t.lower() for t in texts if t)
    for culture, kws in CULTURE_KEYWORDS.items():
        if any(kw in combined for kw in kws):
            return culture
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Art Institute of Chicago
# ─────────────────────────────────────────────────────────────────────────────

AIC_BASE = "https://api.artic.edu/api/v1"
AIC_EXH_FIELDS = "id,title,short_description,aic_start_at,aic_end_at,artwork_ids,status"
AIC_ART_FIELDS = "id,title,date_display,place_of_origin,medium_display,department_title,classification_titles,description,image_id,artist_display,style_titles,subject_titles"
AIC_IMAGE_BASE = "https://www.artic.edu/iiif/2/{image_id}/full/400,/0/default.jpg"


def fetch_aic_exhibitions(max_exhibitions: int = 100) -> list[dict]:
    """抓取含有足够展品（>=5件）的AIC展览列表。"""
    exhibitions = []
    page = 1
    pbar = tqdm(desc="AIC exhibitions", unit="page")

    while len(exhibitions) < max_exhibitions:
        data = get_json(f"{AIC_BASE}/exhibitions", params={
            "fields": AIC_EXH_FIELDS,
            "limit": 100,
            "page": page,
        })
        if not data or not data.get("data"):
            break

        for exh in data["data"]:
            artwork_ids = exh.get("artwork_ids") or []
            if len(artwork_ids) >= 5:  # 过滤条件：至少5件展品
                exhibitions.append(exh)
                if len(exhibitions) >= max_exhibitions:
                    break

        pbar.update(1)
        pbar.set_postfix(found=len(exhibitions), page=page)

        if page >= data["pagination"]["total_pages"]:
            break
        page += 1
        time.sleep(0.3)  # 礼貌爬取

    pbar.close()
    log.info(f"AIC: found {len(exhibitions)} valid exhibitions")
    return exhibitions


def fetch_aic_artworks(artwork_ids: list[int]) -> list[dict]:
    """批量抓取 AIC 展品元数据（每批 100 件）。"""
    artworks = []
    batch_size = 100

    for i in range(0, len(artwork_ids), batch_size):
        batch = artwork_ids[i:i + batch_size]
        ids_str = ",".join(str(x) for x in batch)
        data = get_json(f"{AIC_BASE}/artworks", params={
            "ids": ids_str,
            "fields": AIC_ART_FIELDS,
            "limit": batch_size,
        })
        if data and data.get("data"):
            artworks.extend(data["data"])
        time.sleep(0.3)

    return artworks


def normalize_aic_artwork(art: dict) -> dict:
    """将 AIC artwork 转换为统一 schema。"""
    image_id = art.get("image_id")
    image_url = AIC_IMAGE_BASE.format(image_id=image_id) if image_id else ""

    classifications = art.get("classification_titles") or []
    styles = art.get("style_titles") or []
    artist = art.get("artist_display") or ""
    culture = art.get("place_of_origin") or ""
    if not culture:
        culture = infer_culture_from_text(
            art.get("department_title", ""),
            " ".join(classifications),
            " ".join(styles),
        )

    desc_parts = []
    if art.get("description"):
        # 去掉 HTML 标签
        desc_parts.append(re.sub(r"<[^>]+>", "", art["description"]))
    if artist:
        desc_parts.append(f"Artist: {artist.split(chr(10))[0]}")

    return {
        "id": f"aic_{art['id']}",
        "source": "aic",
        "title": art.get("title", ""),
        "date": art.get("date_display", ""),
        "culture": culture,
        "medium": art.get("medium_display", ""),
        "description": " | ".join(desc_parts),
        "image_url": image_url,
        "department": art.get("department_title", ""),
        "classification": ", ".join(classifications[:3]) if classifications else "",
        "period": ", ".join(styles[:2]) if styles else "",
    }


def collect_aic(max_exhibitions: int = 100, max_total_artworks: int = 2000) -> tuple[list[dict], list[dict]]:
    """AIC 完整采集流程。返回 (exhibitions, objects)。"""
    log.info("=== Collecting AIC data ===")
    raw_exhibitions = fetch_aic_exhibitions(max_exhibitions)

    all_artwork_ids: set[int] = set()
    for exh in raw_exhibitions:
        all_artwork_ids.update(exh.get("artwork_ids") or [])
        if len(all_artwork_ids) >= max_total_artworks:
            break

    artwork_ids = list(all_artwork_ids)[:max_total_artworks]
    log.info(f"AIC: fetching {len(artwork_ids)} unique artworks")
    raw_artworks = fetch_aic_artworks(artwork_ids)

    # ID 映射
    art_lookup = {str(a["id"]): normalize_aic_artwork(a) for a in raw_artworks}

    exhibitions = []
    for exh in raw_exhibitions:
        artwork_ids_in_exh = [aid for aid in (exh.get("artwork_ids") or []) if str(aid) in art_lookup]
        if len(artwork_ids_in_exh) < 5:
            continue

        title = exh.get("title") or ""
        desc = exh.get("short_description") or ""
        # 推断展览主题（去掉年份括号）
        theme = re.sub(r"\s*\(\d{4}.*?\)\s*$", "", title).strip()

        exhibitions.append({
            "id": f"aic_exh_{exh['id']}",
            "source": "aic",
            "title": f"AIC: {title}",
            "theme": theme,
            "description": desc,
            "start_date": exh.get("aic_start_at", ""),
            "end_date": exh.get("aic_end_at", ""),
            "object_ids": [f"aic_{aid}" for aid in artwork_ids_in_exh],
        })

    objects = list(art_lookup.values())
    log.info(f"AIC: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cleveland Museum of Art
# ─────────────────────────────────────────────────────────────────────────────

CMA_BASE = "https://openaccess-api.clevelandart.org/api"
CMA_DEPARTMENTS = [
    "American Painting and Sculpture",
    "Asian Art",
    "African Art",
    "Ancient Near Eastern Art",
    "Egyptian and Ancient Near Eastern Art",
    "Greek and Roman Art",
    "European Paintings",
    "Medieval Art",
    "Prints",
    "Photographs",
    "Textiles",
]

# 为 CMA 构造虚拟展览（按 department + culture 分组）
CMA_VIRTUAL_EXHIBITIONS = [
    {"theme": "American Painting and Sculpture", "departments": ["American Painting and Sculpture"], "cultures": ["America"]},
    {"theme": "East Asian Art", "departments": ["Asian Art"], "cultures": ["China", "Japan", "Korea"]},
    {"theme": "African Art", "departments": ["African Art", "Egyptian and Ancient Near Eastern Art"]},
    {"theme": "Greek and Roman Antiquities", "departments": ["Greek and Roman Art"]},
    {"theme": "European Masters", "departments": ["European Paintings"]},
    {"theme": "Medieval and Byzantine Art", "departments": ["Medieval Art"]},
    {"theme": "Photography and Prints", "departments": ["Prints", "Photographs"]},
    {"theme": "Ancient Near Eastern Art", "departments": ["Ancient Near Eastern Art", "Egyptian and Ancient Near Eastern Art"]},
    {"theme": "Textile Arts", "departments": ["Textiles"]},
    {"theme": "Modern and Contemporary Art", "departments": ["Contemporary Art", "Modern European Painting and Drawing"]},
]


def fetch_cma_artworks_by_department(department: str, limit: int = 300) -> list[dict]:
    """抓取 CMA 某部门的展品列表。"""
    artworks = []
    skip = 0
    batch = 100

    while len(artworks) < limit:
        data = get_json(f"{CMA_BASE}/artworks/", params={
            "department": department,
            "limit": batch,
            "skip": skip,
            "has_image": 1,
            "cc0": 1,  # 只取 CC0 授权展品
        })
        if not data or not data.get("data"):
            break

        artworks.extend(data["data"])
        if len(data["data"]) < batch:
            break
        skip += batch
        time.sleep(0.3)

    return artworks[:limit]


def normalize_cma_artwork(art: dict) -> dict:
    """将 CMA artwork 转换为统一 schema。"""
    images = art.get("images") or {}
    image_url = ""
    if "web" in images and images["web"].get("url"):
        image_url = images["web"]["url"]
    elif "print" in images and images["print"].get("url"):
        image_url = images["print"]["url"]

    culture_raw = art.get("culture") or []
    if isinstance(culture_raw, list):
        culture = ", ".join(culture_raw[:2])
    else:
        culture = str(culture_raw)

    creators = art.get("creators") or []
    creator_str = ""
    if creators:
        c0 = creators[0]
        creator_str = c0.get("description", "") or c0.get("name", "")

    desc = art.get("description") or art.get("did_you_know") or ""

    return {
        "id": f"cma_{art['id']}",
        "source": "cleveland",
        "title": art.get("title", ""),
        "date": art.get("creation_date", ""),
        "culture": culture,
        "medium": art.get("technique", ""),
        "description": f"{desc} | Creator: {creator_str}".strip(" |"),
        "image_url": image_url,
        "department": art.get("department", ""),
        "classification": art.get("type", ""),
        "period": "",
    }


def collect_cleveland(max_per_dept: int = 200) -> tuple[list[dict], list[dict]]:
    """Cleveland 完整采集流程。返回 (exhibitions, objects)。"""
    log.info("=== Collecting Cleveland Museum data ===")

    all_objects: dict[str, dict] = {}
    dept_objects: dict[str, list[str]] = {}

    departments_to_fetch = list({dept for exh in CMA_VIRTUAL_EXHIBITIONS for dept in exh["departments"]})

    for dept in tqdm(departments_to_fetch, desc="CMA departments"):
        raw_artworks = fetch_cma_artworks_by_department(dept, limit=max_per_dept)
        ids_for_dept = []
        for art in raw_artworks:
            obj = normalize_cma_artwork(art)
            if obj["title"]:
                all_objects[obj["id"]] = obj
                ids_for_dept.append(obj["id"])
        dept_objects[dept] = ids_for_dept
        log.debug(f"  CMA {dept}: {len(ids_for_dept)} objects")

    exhibitions = []
    for vexh in CMA_VIRTUAL_EXHIBITIONS:
        obj_ids = []
        for dept in vexh["departments"]:
            obj_ids.extend(dept_objects.get(dept, []))

        if len(obj_ids) < 5:
            continue

        theme = vexh["theme"]
        exhibitions.append({
            "id": f"cma_exh_{theme.lower().replace(' ', '_')}",
            "source": "cleveland",
            "title": f"Cleveland: {theme}",
            "theme": theme,
            "description": f"Works from the Cleveland Museum of Art collection: {theme}",
            "start_date": "",
            "end_date": "",
            "object_ids": obj_ids,
        })

    objects = list(all_objects.values())
    log.info(f"Cleveland: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ─────────────────────────────────────────────────────────────────────────────
# 3. Victoria & Albert Museum
# ─────────────────────────────────────────────────────────────────────────────

VAM_BASE = "https://api.vam.ac.uk/v2"

# V&A 按主题构造虚拟展览
VAM_VIRTUAL_EXHIBITIONS = [
    {"theme": "Japanese Art and Design", "q": "japanese art", "object_type": ""},
    {"theme": "Islamic Art and Architecture", "q": "islamic art", "object_type": ""},
    {"theme": "Indian Textiles and Decorative Arts", "q": "india textile", "object_type": ""},
    {"theme": "European Furniture and Decorative Arts", "q": "european furniture", "object_type": ""},
    {"theme": "Photography", "q": "photography", "object_type": "Photograph"},
    {"theme": "Ceramics from East Asia", "q": "chinese porcelain", "object_type": "Ceramic"},
    {"theme": "Fashion and Textiles", "q": "fashion dress", "object_type": "Textile"},
    {"theme": "Medieval European Art", "q": "medieval european", "object_type": ""},
    {"theme": "South Asian Sculpture", "q": "india sculpture buddhist", "object_type": "Sculpture"},
    {"theme": "Renaissance and Baroque", "q": "renaissance baroque painting", "object_type": ""},
]

VAM_OBJ_FIELDS = "systemNumber,titles,objectType,materialsAndTechniques,productionDates,placesOfOrigin,objectDescriptionNote,images,physicalDescription,artistMakerPerson"


def fetch_vam_objects_by_query(q: str, object_type: str = "", limit: int = 200) -> list[dict]:
    """抓取 V&A 按主题查询的展品。"""
    objects = []
    page = 1
    page_size = 50

    while len(objects) < limit:
        params = {
            "q": q,
            "page": page,
            "page_size": page_size,
            "fields": VAM_OBJ_FIELDS,
            "images_exist": 1,
        }
        if object_type:
            params["object_type"] = object_type

        data = get_json(f"{VAM_BASE}/objects/search", params=params)
        if not data or not data.get("records"):
            break

        objects.extend(data["records"])
        info = data.get("info", {})
        total_pages = (info.get("record_count", 0) + page_size - 1) // page_size
        if page >= min(total_pages, 5):  # 最多5页
            break
        page += 1
        time.sleep(0.4)

    return objects[:limit]


def normalize_vam_artwork(art: dict) -> dict:
    """将 V&A artwork 转换为统一 schema。"""
    sys_num = art.get("systemNumber", "")

    # title
    titles = art.get("titles") or []
    title = titles[0].get("title", "") if titles else art.get("_primaryTitle", "")

    # date
    prod_dates = art.get("productionDates") or []
    date = prod_dates[0].get("date", {}).get("text", "") if prod_dates else ""

    # culture / place of origin
    places = art.get("placesOfOrigin") or []
    culture = places[0].get("place", {}).get("text", "") if places else ""

    # medium
    medium = art.get("materialsAndTechniques", "") or ""
    if isinstance(medium, list):
        medium = ", ".join(medium[:2])

    # description
    desc_parts = []
    if art.get("objectDescriptionNote"):
        desc_parts.append(art["objectDescriptionNote"])
    if art.get("physicalDescription"):
        desc_parts.append(art["physicalDescription"])

    # image
    images = art.get("images") or {}
    image_url = ""
    if "_primary_thumbnail" in images:
        image_url = images["_primary_thumbnail"]
    elif art.get("_primaryImageId"):
        image_url = f"https://framemark.vam.ac.uk/collections/{art['_primaryImageId']}/full/400,/0/default.jpg"

    # artist
    artists = art.get("artistMakerPerson") or []
    artist = artists[0].get("name", {}).get("text", "") if artists else ""

    if not culture:
        culture = infer_culture_from_text(title, " ".join(desc_parts), artist)

    return {
        "id": f"vam_{sys_num}",
        "source": "vam",
        "title": title,
        "date": date,
        "culture": culture,
        "medium": medium,
        "description": " | ".join(p for p in desc_parts if p)[:500],
        "image_url": image_url,
        "department": art.get("objectType", ""),
        "classification": art.get("objectType", ""),
        "period": "",
    }


def collect_vam(max_per_theme: int = 150) -> tuple[list[dict], list[dict]]:
    """V&A 完整采集流程。返回 (exhibitions, objects)。"""
    log.info("=== Collecting V&A data ===")

    all_objects: dict[str, dict] = {}
    theme_objects: dict[str, list[str]] = {}

    for vexh in tqdm(VAM_VIRTUAL_EXHIBITIONS, desc="V&A themes"):
        raw_objects = fetch_vam_objects_by_query(
            vexh["q"], vexh["object_type"], limit=max_per_theme
        )
        ids_for_theme = []
        for art in raw_objects:
            obj = normalize_vam_artwork(art)
            if obj["title"] and obj["id"] != "vam_":
                all_objects[obj["id"]] = obj
                ids_for_theme.append(obj["id"])
        theme_objects[vexh["theme"]] = ids_for_theme
        log.debug(f"  V&A '{vexh['theme']}': {len(ids_for_theme)} objects")

    exhibitions = []
    for vexh in VAM_VIRTUAL_EXHIBITIONS:
        obj_ids = theme_objects.get(vexh["theme"], [])
        if len(obj_ids) < 5:
            continue

        exhibitions.append({
            "id": f"vam_exh_{vexh['theme'].lower().replace(' ', '_').replace('&', 'and')}",
            "source": "vam",
            "title": f"V&A: {vexh['theme']}",
            "theme": vexh["theme"],
            "description": f"Victoria & Albert Museum collection: {vexh['theme']}",
            "start_date": "",
            "end_date": "",
            "object_ids": obj_ids,
        })

    objects = list(all_objects.values())
    log.info(f"V&A: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ─────────────────────────────────────────────────────────────────────────────
# 合并 + 去重 + 统计
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_deduplicate(
    existing_exh: list[dict],
    existing_obj: list[dict],
    new_exh_list: list[list[dict]],
    new_obj_list: list[list[dict]],
) -> tuple[list[dict], list[dict]]:
    """合并新旧数据，按 id 去重。"""
    all_exh = {e["id"]: e for e in existing_exh}
    all_obj = {o["id"]: o for o in existing_obj}

    for exhs in new_exh_list:
        for e in exhs:
            if e["id"] not in all_exh:
                all_exh[e["id"]] = e

    for objs in new_obj_list:
        for o in objs:
            if o["id"] not in all_obj:
                all_obj[o["id"]] = o

    return list(all_exh.values()), list(all_obj.values())


def print_statistics(exhibitions: list[dict], objects: list[dict]) -> None:
    from collections import Counter

    sources = Counter(e.get("source", "?") for e in exhibitions)
    obj_sources = Counter(o.get("source", "?") for o in objects)

    print("\n" + "=" * 60)
    print("ExhibitionBench Dataset Statistics (v2)")
    print("=" * 60)
    print(f"Total exhibitions : {len(exhibitions)}")
    for src, cnt in sources.most_common():
        print(f"  {src:20s}: {cnt}")
    print(f"\nTotal objects     : {len(objects)}")
    for src, cnt in obj_sources.most_common():
        print(f"  {src:20s}: {cnt}")

    # Culture coverage
    cultures = Counter()
    for o in objects:
        c = o.get("culture", "") or ""
        cultures[infer_culture_from_text(c)] += 1
    print("\nCultural coverage (inferred):")
    for cg, cnt in cultures.most_common():
        pct = cnt / len(objects) * 100
        print(f"  {cg:25s}: {cnt:4d} ({pct:.1f}%)")

    # Exhibitions with enough objects for TES
    tes_eligible = sum(1 for e in exhibitions if len(e.get("object_ids", [])) >= 5)
    print(f"\nTES-eligible exhibitions (>=5 objects): {tes_eligible}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Multi-source museum data collection for ExhibitionBench")
    parser.add_argument("--sources", nargs="+", default=["aic", "cleveland", "vam"],
                        choices=["aic", "cleveland", "vam"],
                        help="Which sources to collect")
    parser.add_argument("--max-aic-exhibitions", type=int, default=100,
                        help="Max AIC exhibitions to collect")
    parser.add_argument("--max-aic-artworks", type=int, default=2000,
                        help="Max AIC artworks total")
    parser.add_argument("--max-cma-per-dept", type=int, default=200,
                        help="Max Cleveland artworks per department")
    parser.add_argument("--max-vam-per-theme", type=int, default=150,
                        help="Max V&A objects per theme")
    parser.add_argument("--output-prefix", default="v2",
                        help="Output file suffix (e.g., v2 -> exhibitions_v2.jsonl)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Do not merge with existing data")
    args = parser.parse_args()

    new_exhibitions: list[list[dict]] = []
    new_objects: list[list[dict]] = []

    # --- Collect ---
    if "aic" in args.sources:
        aic_exh, aic_obj = collect_aic(
            max_exhibitions=args.max_aic_exhibitions,
            max_total_artworks=args.max_aic_artworks,
        )
        write_jsonl(DATA / "aic_exhibitions.jsonl", aic_exh)
        write_jsonl(DATA / "aic_objects.jsonl", aic_obj)
        new_exhibitions.append(aic_exh)
        new_objects.append(aic_obj)

    if "cleveland" in args.sources:
        cma_exh, cma_obj = collect_cleveland(max_per_dept=args.max_cma_per_dept)
        write_jsonl(DATA / "cleveland_exhibitions.jsonl", cma_exh)
        write_jsonl(DATA / "cleveland_objects.jsonl", cma_obj)
        new_exhibitions.append(cma_exh)
        new_objects.append(cma_obj)

    if "vam" in args.sources:
        vam_exh, vam_obj = collect_vam(max_per_theme=args.max_vam_per_theme)
        write_jsonl(DATA / "vam_exhibitions.jsonl", vam_exh)
        write_jsonl(DATA / "vam_objects.jsonl", vam_obj)
        new_exhibitions.append(vam_exh)
        new_objects.append(vam_obj)

    # --- Merge ---
    if not args.skip_merge:
        existing_exh = load_jsonl(DATA / "exhibitions.jsonl")
        existing_obj = load_jsonl(DATA / "objects.jsonl")
        log.info(f"Existing: {len(existing_exh)} exhibitions, {len(existing_obj)} objects")

        merged_exh, merged_obj = merge_and_deduplicate(
            existing_exh, existing_obj, new_exhibitions, new_objects
        )

        out_exh = DATA / f"exhibitions_{args.output_prefix}.jsonl"
        out_obj = DATA / f"objects_{args.output_prefix}.jsonl"
        write_jsonl(out_exh, merged_exh)
        write_jsonl(out_obj, merged_obj)

        print_statistics(merged_exh, merged_obj)
    else:
        # 仅统计新数据
        flat_exh = [e for exhs in new_exhibitions for e in exhs]
        flat_obj = [o for objs in new_objects for o in objs]
        print_statistics(flat_exh, flat_obj)


if __name__ == "__main__":
    main()

"""
collect_exhibitions.py  (v2)
============================
Week 1 数据采集主脚本。

数据来源：
1. Met Museum Open Access API   — 按 departmentId 过滤，保证主题一致性
2. Europeana Record Search API  — 替换已失效的 Gallery API，按主题关键词检索

目标：30+ 展览，600+ 件展品（去重后）

输出：
  data/exhibitions.jsonl  — {"id", "source", "title", "theme", "description", "object_ids": [...]}
  data/objects.jsonl      — {"id", "source", "title", "date", "culture", "medium", "description", "image_url"}

使用方法：
  python collect_exhibitions.py              # 全部来源
  python collect_exhibitions.py --source met
  python collect_exhibitions.py --source europeana
"""

import argparse
import json
import random
import time
import logging
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

EXHIBITIONS_PATH = DATA_DIR / "exhibitions.jsonl"
OBJECTS_PATH     = DATA_DIR / "objects.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# 通用工具
# ─────────────────────────────────────────────────────────────────────────────

def get_json(url: str, params: dict = None, retries: int = 3, delay: float = 1.5) -> dict | list | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=25)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.warning(f"[attempt {attempt+1}/{retries}] {url} → {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return None


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_ids(path: Path) -> set[str]:
    """读取已有 JSONL 的 id 集合（断点续采用）。"""
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                pass
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# 来源 1：Met Museum Open Access（按 departmentId + 关键词）
# ─────────────────────────────────────────────────────────────────────────────

MET_SEARCH_API = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT_API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

# (exhibition_name, search_query, department_id)
# department_id=None 表示不限 department
MET_THEMES = [
    # European Paintings (dept 11)
    ("Impressionism",          "impressionism",          11),
    ("Dutch Masters",          "dutch golden age",       11),
    ("Italian Renaissance",    "renaissance",            11),
    ("Baroque Painting",       "baroque",                11),
    ("Rococo Art",             "rococo",                 11),
    ("Romanticism",            "romanticism",            11),
    ("Realism Painting",       "realism landscape",      11),
    ("Portrait Painting",      "portrait nobleman",      11),
    # Egyptian Art (dept 10)
    ("Ancient Egypt",          "egypt ritual tomb",      None),
    ("Egyptian Jewelry",       "egypt jewelry amulet",   None),
    # Asian Art (dept 6)
    ("Japanese Art",           "japan",                   6),
    ("Chinese Art",            "china",                   6),
    ("Korean Art",             "korea",                  None),
    ("Indian Art",             "india",                  None),
    ("Southeast Asian Art",    "thailand cambodia",      None),
    ("Japanese Ceramics",      "japan ceramic porcelain", 6),
    ("Chinese Landscape",      "chinese landscape scroll", 6),
    # Islamic Art
    ("Islamic Art",            "islamic mosque",         None),
    ("Islamic Manuscript",     "islamic manuscript quran", None),
    ("Persian Art",            "persian iran",           None),
    # Greek and Roman Art (dept 13)
    ("Ancient Greece",         "greek pottery",          None),
    ("Roman Art",              "roman emperor",          None),
    ("Greek Sculpture",        "greek marble sculpture", None),
    ("Roman Portrait",         "roman portrait bust",    None),
    # Arts of Africa, Oceania, Americas (dept 5)
    ("African Art",            "africa",                 None),
    ("Oceanic Art",            "pacific oceanic",        None),
    ("Pre-Columbian Art",      "mesoamerica maya aztec", None),
    # Medieval Art (dept 17)
    ("Medieval Art",           "medieval",               17),
    ("Medieval Tapestry",      "tapestry medieval",      17),
    # American Wing (dept 1)
    ("American Art",           "american landscape",     None),
    ("American Portrait",      "american colonial portrait", None),
    # Arms and Armor (dept 4)
    ("Arms and Armor",         "armor",                  None),
    ("Ceremonial Weapons",     "ceremonial sword dagger", None),
    # Photographs (dept 19)
    ("Photography",            "photograph portrait",    None),
    ("Documentary Photography","street photography documentary", None),
    # Modern Art (dept 21)
    ("Modern Art",             "abstract modern",        21),
    ("Expressionism",          "expressionism",          21),
    # Drawings and Prints (dept 9)
    ("Drawings and Prints",    "drawing etching",         9),
    ("Woodblock Prints",       "woodblock print",         9),
    # Musical Instruments (dept 18)
    ("Musical Instruments",    "instrument lute violin", 18),
    # Costume Institute (dept 8)
    ("Fashion and Costume",    "dress embroidery",       None),
    ("Historical Textiles",    "textile weaving loom",   None),
    # Decorative Arts
    ("European Decorative Arts","furniture silver enamel", None),
    ("Ceramics and Glass",     "glass vessel ceramic",   None),
]


def collect_met(themes=None, max_per_theme: int = 40) -> tuple[int, int]:
    """
    通过 Met 搜索 API（按 departmentId）采集展品，每个主题独立成一个展览。

    关键修复：
    - 使用 departmentId 过滤确保主题一致性，避免不同主题搜索结果交叉污染
    - existing_obj 仅用于跳过重复写入，不影响 exhibition 的 object_ids 组成
    - 每个主题的对象独立决定，与其他主题互不干扰
    """
    if themes is None:
        themes = MET_THEMES
    log.info("=== Met Museum: 开始采集 ===")

    # 仅用于跳过已有展览（断点续采）
    existing_exh = load_existing_ids(EXHIBITIONS_PATH)
    # 仅用于跳过已写入展品（避免重复写入 objects.jsonl）
    existing_obj = load_existing_ids(OBJECTS_PATH)

    n_exh = 0
    n_obj_new = 0

    for theme_name, query, dept_id in tqdm(themes, desc="Met themes"):
        exh_id = f"met_{theme_name.lower().replace(' ', '_')}"
        if exh_id in existing_exh:
            log.info(f"  跳过已有展览: {exh_id}")
            continue

        # 使用 departmentId 限制搜索范围
        params = {
            "q": query,
            "hasImages": "true",
            "isPublicDomain": "true",
        }
        if dept_id is not None:
            params["departmentId"] = dept_id

        search_data = get_json(MET_SEARCH_API, params=params)
        if not search_data or "objectIDs" not in search_data:
            log.warning(f"  {theme_name}: 搜索无结果")
            continue

        # 随机采样以增加多样性（而非只取前 N 个）
        all_ids = search_data.get("objectIDs") or []
        if not all_ids:
            log.warning(f"  {theme_name}: objectIDs 为空")
            continue
        if len(all_ids) > max_per_theme * 4:
            candidate_pool = random.sample(all_ids, max_per_theme * 4)
        else:
            candidate_pool = all_ids

        collected_ids = []
        for oid in tqdm(candidate_pool, desc=f"  {theme_name} objects", leave=False):
            if len(collected_ids) >= max_per_theme:
                break

            obj_id = f"met_{oid}"

            # 若展品未在 objects.jsonl，则获取并写入
            if obj_id not in existing_obj:
                obj_data = get_json(MET_OBJECT_API.format(oid))
                if not obj_data or not obj_data.get("title"):
                    time.sleep(0.1)
                    continue
                obj_record = {
                    "id": obj_id,
                    "source": "met",
                    "title": obj_data.get("title", ""),
                    "date": obj_data.get("objectDate", ""),
                    "culture": (
                        obj_data.get("culture", "")
                        or obj_data.get("artistNationality", "")
                        or obj_data.get("country", "")
                    ),
                    "medium": obj_data.get("medium", ""),
                    "description": " | ".join(filter(None, [
                        obj_data.get("creditLine", ""),
                        obj_data.get("artistDisplayName", ""),
                        obj_data.get("objectName", ""),
                    ])),
                    "image_url": (
                        obj_data.get("primaryImageSmall", "")
                        or obj_data.get("primaryImage", "")
                    ),
                    "department": obj_data.get("department", ""),
                    "classification": obj_data.get("classification", ""),
                    "period": obj_data.get("period", ""),
                }
                append_jsonl(OBJECTS_PATH, obj_record)
                existing_obj.add(obj_id)
                n_obj_new += 1
                time.sleep(0.15)

            # 无论是否已有，都加入本展览的 object_ids
            collected_ids.append(obj_id)

        if collected_ids:
            exh_record = {
                "id": exh_id,
                "source": "met",
                "title": f"Met: {theme_name}",
                "theme": theme_name,
                "description": f"Met Museum collection: {theme_name}",
                "object_ids": collected_ids,
            }
            append_jsonl(EXHIBITIONS_PATH, exh_record)
            existing_exh.add(exh_id)
            n_exh += 1
            log.info(f"  ✓ {theme_name}: {len(collected_ids)} 件展品 ({n_obj_new} 件新写入)")

    log.info(f"Met: 采集 {n_exh} 个展览, 新写入 {n_obj_new} 件展品")
    return n_exh, n_obj_new


# ─────────────────────────────────────────────────────────────────────────────
# 来源 2：Europeana Record Search API（替换已失效的 Gallery API）
# ─────────────────────────────────────────────────────────────────────────────

EUROPEANA_SEARCH_API = "https://api.europeana.eu/record/v2/search.json"

# 每个主题对应一个 Europeana "虚拟展览"
EUROPEANA_THEMES = [
    ("Impressionism",        "impressionism monet renoir"),
    ("Baroque Art",          "baroque OR caravaggio OR rembrandt"),
    ("Italian Renaissance",  "renaissance OR raphael"),
    ("Ancient Greece",       "ancient OR greek OR pottery OR amphora"),
    ("Islamic Calligraphy",  "islamic OR calligraphy OR mosque"),
    ("Medieval Manuscript",  "medieval OR illuminated OR manuscript"),
    ("Chinese Porcelain",    "chinese OR porcelain OR celadon"),
    ("Japanese Prints",      "japanese OR woodblock OR ukiyo-e"),
    ("African Sculpture",    "african OR sculpture OR mask"),
    ("Photography Portrait", "photography OR portrait OR daguerreotype"),
    ("Dutch Golden Age",     "dutch OR golden OR vermeer"),
    ("Egyptian Antiquities", "egypt OR hieroglyph OR mummy"),
    ("Roman Sculpture",      "roman OR sculpture OR emperor"),
    ("Victorian Painting",   "victorian OR painting OR england"),
    ("Symbolism Art",        "symbolism OR klimt OR moreau"),
    # New themes
    ("Flemish Masters",      "flemish OR rubens OR eyck"),
    ("Spanish Painting",     "spanish OR velazquez OR goya"),
    ("Nordic Art",           "nordic OR scandinavian OR landscape"),
    ("Art Nouveau",          "nouveau OR alphonse OR mucha"),
    ("Neoclassicism",        "neoclassicism OR neoclassical OR david"),
    ("Landscape Painting",   "landscape OR european OR countryside"),
    ("Still Life Painting",  "still OR flowers OR fruit"),
    ("Religious Art",        "religious OR altar OR madonna"),
    ("Byzantine Art",        "byzantine OR icon OR mosaic"),
    ("Celtic Art",           "celtic OR medieval OR ireland"),
    ("Ottoman Art",          "ottoman OR turkish OR miniature"),
    ("Persian Miniature",    "persian OR miniature OR iran"),
    ("Indian Miniature",     "mughal OR indian OR miniature"),
    ("Ancient Roman",        "roman OR fresco OR mosaic"),
    ("Etruscan Art",         "etruscan OR ancient OR italy"),
]


def _extract_europeana_obj(item: dict) -> dict | None:
    """从 Europeana 搜索结果条目提取展品记录。"""
    record_id = item.get("id", "")  # e.g. /2024900/photograph_1234
    if not record_id:
        return None

    obj_id = "europeana" + record_id.replace("/", "_")

    # 提取 title
    title = ""
    raw_title = item.get("title", [])
    if isinstance(raw_title, list) and raw_title:
        title = str(raw_title[0])
    elif isinstance(raw_title, str):
        title = raw_title

    # 提取 date
    date = ""
    for key in ("year", "dctermsCreated", "dcDate"):
        raw = item.get(key, [])
        if raw:
            date = str(raw[0]) if isinstance(raw, list) else str(raw)
            break

    # 提取 culture / country
    culture = ""
    for key in ("edmCountry", "dctermsProvenance", "country"):
        raw = item.get(key, [])
        if raw:
            culture = str(raw[0]) if isinstance(raw, list) else str(raw)
            break

    # 提取 medium
    medium = ""
    for key in ("dcFormat", "edmHasView"):
        raw = item.get(key, [])
        if raw:
            medium = str(raw[0]) if isinstance(raw, list) else str(raw)
            break

    # 提取 description
    description = ""
    for key in ("dcDescription", "description"):
        raw = item.get(key, [])
        if raw:
            description = str(raw[0]) if isinstance(raw, list) else str(raw)
            if len(description) > 300:
                description = description[:300]
            break
    # 补充 creator
    creator = ""
    raw_creator = item.get("dcCreator", [])
    if raw_creator:
        creator = str(raw_creator[0]) if isinstance(raw_creator, list) else str(raw_creator)
    if creator:
        description = (creator + " | " + description).strip(" |")

    # 提取 image_url
    image_url = ""
    for key in ("edmPreview", "edmIsShownBy", "thumbnail"):
        raw = item.get(key, [])
        if raw:
            image_url = str(raw[0]) if isinstance(raw, list) else str(raw)
            break

    if not title:
        return None

    return {
        "id": obj_id,
        "source": "europeana",
        "title": title,
        "date": date,
        "culture": culture,
        "medium": medium,
        "description": description,
        "image_url": image_url,
        "europeana_id": record_id,
    }


def collect_europeana(themes=None, max_per_theme: int = 40, api_key: str = "apidemo") -> tuple[int, int]:
    """
    通过 Europeana Record Search API 按主题采集展品。
    每个主题关键词 → 一个虚拟展览。

    注意：使用 Record Search API（非已失效的 Gallery API）。
    """
    if themes is None:
        themes = EUROPEANA_THEMES
    log.info("=== Europeana Record Search: 开始采集 ===")

    existing_exh = load_existing_ids(EXHIBITIONS_PATH)
    existing_obj = load_existing_ids(OBJECTS_PATH)

    n_exh = 0
    n_obj_new = 0

    for theme_name, query in tqdm(themes, desc="Europeana themes"):
        exh_id = f"europeana_{theme_name.lower().replace(' ', '_')}"
        if exh_id in existing_exh:
            log.info(f"  跳过已有展览: {exh_id}")
            continue

        # 搜索结果
        params = {
            "query": query,
            "rows": min(max_per_theme * 2, 100),  # 多取一些以防空 title 的条目
            "wskey": api_key,
            "profile": "rich",
            "media": "true",
            "type": "IMAGE",
            "start": 1,
        }
        search_data = get_json(EUROPEANA_SEARCH_API, params=params)
        if not search_data or "items" not in search_data:
            log.warning(f"  {theme_name}: 搜索无结果 (resp={str(search_data)[:100]})")
            continue

        items = search_data.get("items", [])
        collected_ids = []

        for item in items:
            if len(collected_ids) >= max_per_theme:
                break
            obj_record = _extract_europeana_obj(item)
            if obj_record is None:
                continue
            obj_id = obj_record["id"]
            if obj_id not in existing_obj:
                append_jsonl(OBJECTS_PATH, obj_record)
                existing_obj.add(obj_id)
                n_obj_new += 1
            collected_ids.append(obj_id)

        if collected_ids:
            exh_record = {
                "id": exh_id,
                "source": "europeana",
                "title": f"Europeana: {theme_name}",
                "theme": theme_name,
                "description": f"Europeana curated selection: {theme_name}",
                "object_ids": collected_ids,
            }
            append_jsonl(EXHIBITIONS_PATH, exh_record)
            existing_exh.add(exh_id)
            n_exh += 1
            log.info(f"  ✓ {theme_name}: {len(collected_ids)} 件展品 ({n_obj_new} 件新写入)")

        time.sleep(0.5)  # Europeana rate limit

    log.info(f"Europeana: 采集 {n_exh} 个展览, 新写入 {n_obj_new} 件展品")
    return n_exh, n_obj_new


# ─────────────────────────────────────────────────────────────────────────────
# 统计报告
# ─────────────────────────────────────────────────────────────────────────────

def print_stats():
    exh_count = sum(1 for _ in open(EXHIBITIONS_PATH, encoding="utf-8")) if EXHIBITIONS_PATH.exists() else 0
    obj_count = sum(1 for _ in open(OBJECTS_PATH, encoding="utf-8")) if OBJECTS_PATH.exists() else 0
    log.info("=" * 60)
    log.info(f"总计: {exh_count} 个展览, {obj_count} 件展品")
    log.info(f"exhibitions.jsonl → {EXHIBITIONS_PATH}")
    log.info(f"objects.jsonl     → {OBJECTS_PATH}")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ExhibitionBench 数据采集脚本 v2")
    parser.add_argument("--source", choices=["europeana", "met", "all"], default="all")
    parser.add_argument("--europeana-key", default="apidemo")
    parser.add_argument("--max-per-theme", type=int, default=40)
    args = parser.parse_args()

    if args.source in ("met", "all"):
        collect_met(max_per_theme=args.max_per_theme)

    if args.source in ("europeana", "all"):
        collect_europeana(max_per_theme=args.max_per_theme, api_key=args.europeana_key)

    print_stats()


if __name__ == "__main__":
    main()

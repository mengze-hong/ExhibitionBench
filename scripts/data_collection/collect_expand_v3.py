"""
collect_expand_v3.py
====================
大规模数据扩充 — ExhibitionBench v3

新增数据源：
  1. Smithsonian Institution (collections.si.edu)
  2. Harvard Art Museums API
  3. National Gallery of Art (NGA) Open Data
  4. Brooklyn Museum Open API
  5. Cooper Hewitt Smithsonian Design Museum
  6. Rijksmuseum (re-attempt with public API)
  7. Finnish National Gallery (Ateneum / Finna API)
  8. LACMA (Los Angeles County Museum of Art)
  9. National Museum of Korea (OpenAPI)
 10. Museum of Modern Art (MoMA) GitHub data

目标: ≥ 300 展览, ≥ 12,000 展品, 覆盖 8+ 文化圈

用法:
  python collect_expand_v3.py
  python collect_expand_v3.py --source smithsonian
  python collect_expand_v3.py --all --merge-v2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.parse
import urllib.error

log = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def http_get(url: str, headers: dict = None, timeout: int = 20, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {
                "User-Agent": "ExhibitionBench/3.0 (academic research; github.com/exhibitionbench)"
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(2 ** attempt)
            elif e.code == 404:
                return None
            else:
                log.debug(f"HTTP {e.code} for {url}")
                return None
        except Exception as e:
            log.debug(f"Error fetching {url}: {e}")
            time.sleep(1)
    return None


def safe_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


CULTURE_KEYWORDS = {
    "East Asian":       ["chinese","china","japanese","japan","korean","korea","tang","song","ming","qing","edo","meiji","han","zhou","yuan","dynasties","dynasty"],
    "South Asian":      ["indian","india","mughal","hindu","buddhist","bengal","rajput","nepali","tibetan","southeast asian","thai","cambodian","khmer","vietnamese"],
    "Islamic":          ["islamic","muslim","ottoman","persian","iran","arab","mamluk","safavid","timurid","calligraphy","quran"],
    "African":          ["african","africa","yoruba","akan","benin","zulu","ashanti","egyptian","nubian","ethiopian","sub-saharan"],
    "Ancient":          ["greek","roman","roman empire","byzantine","ancient","egypt","mesopotamia","sumerian","assyrian","babylonian","etruscan","celtic","viking","prehistoric","bronze age","iron age"],
    "Western European": ["french","french","german","dutch","flemish","italian","spanish","british","english","scandinavian","european","renaissance","baroque","rococo","impressionism","cubism","surrealism"],
    "American":         ["american","united states","native american","pre-columbian","aztec","mayan","inca","latin american","mexican","colonial american"],
    "South American":   ["south american","brazilian","peruvian","colombian","andean","amazon"],
}

def infer_culture(record: dict) -> str:
    text = " ".join([
        safe_str(record.get("culture", "")),
        safe_str(record.get("department", "")),
        safe_str(record.get("classification", "")),
        safe_str(record.get("period", "")),
        safe_str(record.get("title", ""))
    ]).lower()
    for group, kws in CULTURE_KEYWORDS.items():
        if any(k in text for k in kws):
            return group
    return "Other"


def make_obj(source, oid, title, date="", culture="", medium="", dept="",
             desc="", img_url="", classification="", period="") -> dict:
    obj = {
        "id": f"{source}_{oid}",
        "source": source,
        "title": safe_str(title),
        "date": safe_str(date),
        "culture": safe_str(culture),
        "medium": safe_str(medium),
        "department": safe_str(dept),
        "description": safe_str(desc),
        "image_url": safe_str(img_url),
        "classification": safe_str(classification),
        "period": safe_str(period),
    }
    if not obj["culture"]:
        obj["culture"] = infer_culture(obj)
    return obj


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def write_jsonl(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records)} -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Smithsonian Institution Open Access
# ──────────────────────────────────────────────────────────────────────────────

def collect_smithsonian(max_per_unit: int = 200, max_units: int = 20) -> tuple[list, list]:
    """
    Smithsonian Open Access API — multiple units (NMAH, NASM, NMNH, Freer, SAAM etc.)
    Endpoint: https://api.si.edu/openaccess/api/v1.0/search
    """
    log.info("=== Collecting Smithsonian data ===")
    API = "https://api.si.edu/openaccess/api/v1.0/search"

    # Virtual exhibitions by topic query per museum unit
    SI_THEMES = [
        {"theme": "American History and Culture", "q": "american history flag democracy", "unit": "NMAH"},
        {"theme": "Space Exploration and Aviation", "q": "spacecraft satellite astronaut aviation", "unit": "NASM"},
        {"theme": "Natural History: Ancient Life", "q": "fossil dinosaur prehistoric mammal skeleton", "unit": "NMNH"},
        {"theme": "Freer Gallery: Asian Art", "q": "chinese jade bronze japanese ink painting", "unit": "FSG"},
        {"theme": "American Art: Landscapes and Portraits", "q": "landscape portrait oil painting american artist", "unit": "SAAM"},
        {"theme": "African and Pacific Cultures", "q": "african mask sculpture ritual pacific textile", "unit": "NMNH"},
        {"theme": "Native American Heritage", "q": "native american indigenous beadwork pottery weaving", "unit": "NMAI"},
        {"theme": "Industrial Design and Technology", "q": "industrial design machine tool manufacturing", "unit": "NMAH"},
        {"theme": "World Cultures: Asia and Middle East", "q": "persian tile ottoman ceramics india mughal", "unit": "FSG"},
        {"theme": "Modern and Contemporary Art", "q": "abstract expressionism modern sculpture installation", "unit": "SAAM"},
        {"theme": "Science and Medicine in History", "q": "medical instrument scientific apparatus laboratory", "unit": "NMAH"},
        {"theme": "Global Textiles and Fashion", "q": "textile weaving embroidery silk loom garment", "unit": "NMNH"},
        {"theme": "Ancient Civilizations", "q": "ancient egypt roman greek mesopotamia bronze", "unit": "NMNH"},
        {"theme": "Photography and Visual Media", "q": "photograph daguerreotype camera photographic print", "unit": "SAAM"},
        {"theme": "Musical Instruments Across Cultures", "q": "musical instrument violin drum flute guitar", "unit": "NMAH"},
    ]

    exhibitions, objects = [], []
    seen_ids = set()

    for theme_conf in SI_THEMES[:max_units]:
        params = {
            "q": theme_conf["q"],
            "unit_code": theme_conf["unit"],
            "type": "edanmdm",
            "rows": max_per_unit,
            "online_only": "true",
        }
        url = API + "?" + urllib.parse.urlencode(params)
        data = http_get(url)
        if not data:
            log.warning(f"Smithsonian: no data for theme {theme_conf['theme']}")
            continue

        rows = data.get("response", {}).get("rows", [])
        log.info(f"  SI {theme_conf['unit']} '{theme_conf['theme']}': {len(rows)} results")

        obj_ids = []
        for row in rows:
            content = row.get("content", {})
            desc_no = content.get("descriptiveNonRepeating", {})
            freetext = content.get("freetext", {})
            indexed = content.get("indexedStructured", {})

            oid = row.get("id", "")
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)

            title = desc_no.get("title", {}).get("content", "") or safe_str(row.get("title", ""))
            if not title:
                continue

            # Date
            dates = freetext.get("date", [])
            date_str = dates[0].get("content", "") if dates else ""

            # Culture/place
            places = indexed.get("place", [])
            culture_str = places[0] if places else ""

            # Medium
            media = freetext.get("physicalDescription", [])
            medium_str = media[0].get("content", "") if media else ""

            # Dept
            data_src = desc_no.get("data_source", "")
            dept_str = theme_conf["unit"]

            # Image
            img = desc_no.get("online_media", {})
            img_list = img.get("media", []) if isinstance(img, dict) else []
            img_url = img_list[0].get("content", "") if img_list else ""

            obj = make_obj(
                "smithsonian", oid, title, date=date_str, culture=culture_str,
                medium=medium_str, dept=dept_str, img_url=img_url
            )
            objects.append(obj)
            obj_ids.append(f"smithsonian_{oid}")

        if len(obj_ids) >= 5:
            exh_id = f"si_{theme_conf['unit'].lower()}_{theme_conf['theme'].lower().replace(' ','_')[:30]}"
            exhibitions.append({
                "id": exh_id,
                "source": "smithsonian",
                "title": theme_conf["theme"],
                "theme": theme_conf["theme"],
                "description": f"Smithsonian {theme_conf['unit']} collection focused on {theme_conf['theme'].lower()}.",
                "object_ids": obj_ids,
            })

    log.info(f"Smithsonian: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ──────────────────────────────────────────────────────────────────────────────
# 2. Harvard Art Museums
# ──────────────────────────────────────────────────────────────────────────────

def collect_harvard(max_per_exh: int = 150) -> tuple[list, list]:
    """Harvard Art Museums API — real exhibitions endpoint."""
    log.info("=== Collecting Harvard Art Museums data ===")
    # Public key (demo) — works for academic use
    HAM_API = "https://api.harvardartmuseums.org"
    # They allow unauthenticated with limited rate
    # Use their public gallery endpoint

    HAM_GALLERIES = [
        {"theme": "Ancient Mediterranean Art", "classification": "Ancient Art"},
        {"theme": "European Paintings 15th-19th Century", "classification": "Paintings"},
        {"theme": "Works on Paper and Photography", "classification": "Works on Paper"},
        {"theme": "Decorative Arts and Design", "classification": "Decorative Arts"},
        {"theme": "Asian and Middle Eastern Art", "classification": "Asian Art"},
        {"theme": "African and Oceanic Art", "classification": "African Art"},
        {"theme": "Modern and Contemporary Art", "classification": "Contemporary Art"},
        {"theme": "Prints and Drawings", "classification": "Drawings"},
    ]

    exhibitions, objects = [], []
    seen_ids = set()

    for gal in HAM_GALLERIES:
        params = {
            "classification": gal["classification"],
            "hasimage": 1,
            "size": max_per_exh,
            "fields": "id,title,dated,culture,medium,department,classification,primaryimageurl,description",
            "sort": "random",
        }
        url = f"{HAM_API}/object?" + urllib.parse.urlencode(params)
        data = http_get(url, timeout=25)
        if not data:
            log.warning(f"Harvard: no data for {gal['theme']}")
            continue

        records = data.get("records", [])
        log.info(f"  HAM '{gal['theme']}': {len(records)} results")

        obj_ids = []
        for rec in records:
            oid = str(rec.get("id", ""))
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)

            title = safe_str(rec.get("title", ""))
            if not title:
                continue

            obj = make_obj(
                "harvard", oid, title,
                date=safe_str(rec.get("dated", "")),
                culture=safe_str(rec.get("culture", "")),
                medium=safe_str(rec.get("medium", "")),
                dept=safe_str(rec.get("department", "")),
                desc=safe_str(rec.get("description", "")),
                img_url=safe_str(rec.get("primaryimageurl", "")),
                classification=safe_str(rec.get("classification", "")),
            )
            objects.append(obj)
            obj_ids.append(f"harvard_{oid}")

        if len(obj_ids) >= 5:
            exh_id = f"harvard_{gal['classification'].lower().replace(' ','_')[:25]}"
            exhibitions.append({
                "id": exh_id,
                "source": "harvard",
                "title": gal["theme"],
                "theme": gal["theme"],
                "description": f"Harvard Art Museums collection: {gal['theme'].lower()}.",
                "object_ids": obj_ids,
            })
        time.sleep(0.5)

    log.info(f"Harvard: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ──────────────────────────────────────────────────────────────────────────────
# 3. National Gallery of Art (Open Data on GitHub)
# ──────────────────────────────────────────────────────────────────────────────

def collect_nga(max_per_dept: int = 200) -> tuple[list, list]:
    """NGA Open Data CSV via their public API."""
    log.info("=== Collecting National Gallery of Art data ===")
    NGA_API = "https://api.nga.gov/art/tms/objects"

    NGA_THEMES = [
        {"theme": "American Art: Colonial to Modern", "school": "American"},
        {"theme": "French Impressionism and Post-Impressionism", "school": "French"},
        {"theme": "Italian Renaissance Masterworks", "school": "Italian"},
        {"theme": "Dutch and Flemish Golden Age", "school": "Dutch"},
        {"theme": "Spanish Baroque and Modern", "school": "Spanish"},
        {"theme": "British Art and Portraiture", "school": "British"},
        {"theme": "German and Austrian Art", "school": "German"},
        {"theme": "Asian Decorative Arts", "school": "East Asian"},
    ]

    exhibitions, objects = [], []
    seen_ids = set()

    for theme_conf in NGA_THEMES:
        params = {
            "q": theme_conf["school"],
            "limit": max_per_dept,
            "offset": 0,
        }
        url = f"{NGA_API}?" + urllib.parse.urlencode(params)
        data = http_get(url, timeout=20)
        if not data:
            # Fallback: use NGA's IIIF/JSON feed
            log.warning(f"NGA: no data for {theme_conf['theme']}, trying alternate")
            alt_url = f"https://api.nga.gov/art/tms/objects?q={urllib.parse.quote(theme_conf['school'])}&limit={max_per_dept}"
            data = http_get(alt_url, timeout=20)

        if not data:
            log.warning(f"NGA: skipping {theme_conf['theme']}")
            continue

        items = []
        if isinstance(data, dict):
            items = data.get("data", data.get("objects", data.get("items", [])))
        elif isinstance(data, list):
            items = data

        log.info(f"  NGA '{theme_conf['theme']}': {len(items)} results")

        obj_ids = []
        for item in items[:max_per_dept]:
            oid = str(item.get("objectID", item.get("id", item.get("tmsid", ""))))
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)

            title = safe_str(item.get("title", item.get("displayTitle", "")))
            if not title:
                continue

            obj = make_obj(
                "nga", oid, title,
                date=safe_str(item.get("displayDate", item.get("beginYear", ""))),
                culture=safe_str(item.get("attributionInverted", item.get("school", ""))),
                medium=safe_str(item.get("medium", item.get("technique", ""))),
                dept=safe_str(item.get("classification", item.get("department", ""))),
                desc=safe_str(item.get("description", "")),
                img_url=safe_str(item.get("iiifThumbUrl", item.get("primaryImageSmall", ""))),
                classification=safe_str(item.get("classification", "")),
                period=safe_str(item.get("subclassification", "")),
            )
            objects.append(obj)
            obj_ids.append(f"nga_{oid}")

        if len(obj_ids) >= 5:
            exh_id = f"nga_{theme_conf['school'].lower().replace(' ','_')}"
            exhibitions.append({
                "id": exh_id,
                "source": "nga",
                "title": theme_conf["theme"],
                "theme": theme_conf["theme"],
                "description": f"National Gallery of Art: {theme_conf['theme'].lower()}.",
                "object_ids": obj_ids,
            })
        time.sleep(0.3)

    log.info(f"NGA: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ──────────────────────────────────────────────────────────────────────────────
# 4. Rijksmuseum (public key approach)
# ──────────────────────────────────────────────────────────────────────────────

def collect_rijksmuseum(max_per_theme: int = 150) -> tuple[list, list]:
    """Rijksmuseum API with public demo key."""
    log.info("=== Collecting Rijksmuseum data ===")
    # The Rijksmuseum API requires a key but provides a demo key in documentation
    RIJ_API = "https://www.rijksmuseum.nl/api/en/collection"
    API_KEY = "0fiuZFh4"  # public demo key from docs

    THEMES = [
        {"theme": "Dutch Golden Age Painting", "q": "Rembrandt Vermeer Dutch painting"},
        {"theme": "Delftware and Ceramics", "q": "Delft ceramic porcelain pottery"},
        {"theme": "Dutch Still Life", "q": "still life flowers fruit Dutch"},
        {"theme": "Portraiture and Identity", "q": "portrait Dutch nobleman lady"},
        {"theme": "Landscapes and Seascapes", "q": "landscape seascape Dutch countryside"},
        {"theme": "Prints and Drawings: Rembrandt", "q": "Rembrandt etching drawing print"},
        {"theme": "Asian Art in Dutch Collections", "q": "Japan China Japanese Chinese export"},
        {"theme": "Silver and Decorative Objects", "q": "silver gold ornament decorative Dutch"},
    ]

    exhibitions, objects = [], []
    seen_ids = set()

    for theme_conf in THEMES:
        params = {
            "key": API_KEY,
            "q": theme_conf["q"],
            "ps": max_per_theme,
            "imgonly": True,
            "format": "json",
        }
        url = f"{RIJ_API}?" + urllib.parse.urlencode(params)
        data = http_get(url, timeout=20)
        if not data or "artObjects" not in data:
            log.warning(f"Rijksmuseum: failed for '{theme_conf['theme']}'")
            continue

        items = data["artObjects"]
        log.info(f"  Rijks '{theme_conf['theme']}': {len(items)} items")

        obj_ids = []
        for item in items:
            oid = safe_str(item.get("objectNumber", ""))
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)

            title = safe_str(item.get("title", item.get("longTitle", "")))
            if not title:
                continue

            img_url = ""
            web_img = item.get("webImage", {})
            if isinstance(web_img, dict):
                img_url = safe_str(web_img.get("url", ""))

            obj = make_obj(
                "rijksmuseum", oid, title,
                date=safe_str(item.get("longTitle", "").split(",")[-1].strip() if "," in item.get("longTitle","") else ""),
                culture="Dutch",
                medium=safe_str(item.get("scLabelLine", "")),
                dept="Dutch and Flemish Art",
                img_url=img_url,
                classification=safe_str(item.get("objectTypes", [""])[0] if item.get("objectTypes") else ""),
            )
            objects.append(obj)
            obj_ids.append(f"rijksmuseum_{oid}")

        if len(obj_ids) >= 5:
            exh_id = f"rijks_{theme_conf['theme'].lower().replace(' ','_')[:30].replace(':','')}"
            exhibitions.append({
                "id": exh_id,
                "source": "rijksmuseum",
                "title": theme_conf["theme"],
                "theme": theme_conf["theme"],
                "description": f"Rijksmuseum collection: {theme_conf['theme'].lower()}.",
                "object_ids": obj_ids,
            })
        time.sleep(0.5)

    log.info(f"Rijksmuseum: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ──────────────────────────────────────────────────────────────────────────────
# 5. Cooper Hewitt Smithsonian Design Museum
# ──────────────────────────────────────────────────────────────────────────────

def collect_cooperhewitt(max_per_theme: int = 150) -> tuple[list, list]:
    """Cooper Hewitt open collection API."""
    log.info("=== Collecting Cooper Hewitt data ===")
    CH_API = "https://api.collection.cooperhewitt.org/rest"
    # Public access key for open data
    CH_KEY = "3f2cf6a2bf6a0cef7e640a6ef8c2a4a2"

    THEMES = [
        {"theme": "Graphic Design and Posters", "dept": "Drawings, Prints, and Graphic Design"},
        {"theme": "Textiles and Surface Design", "dept": "Textiles"},
        {"theme": "Product and Industrial Design", "dept": "Product Design and Decorative Arts"},
        {"theme": "Wallcoverings and Architectural Design", "dept": "Wallcoverings"},
        {"theme": "Digital Design and Media", "dept": "Digital and Interaction Design"},
    ]

    exhibitions, objects = [], []
    seen_ids = set()

    for theme_conf in THEMES:
        params = {
            "method": "cooperhewitt.objects.search",
            "access_token": CH_KEY,
            "department_name": theme_conf["dept"],
            "per_page": max_per_theme,
            "page": 1,
            "has_images": 1,
        }
        url = f"{CH_API}/?" + urllib.parse.urlencode(params)
        data = http_get(url, timeout=20)
        if not data or data.get("stat") == "fail":
            # Try alternate endpoint
            alt_url = f"https://api.collection.cooperhewitt.org/rest/?method=cooperhewitt.objects.search&access_token={CH_KEY}&q={urllib.parse.quote(theme_conf['theme'])}&per_page={max_per_theme}"
            data = http_get(alt_url, timeout=20)

        if not data or not isinstance(data, dict):
            log.warning(f"Cooper Hewitt: failed for '{theme_conf['theme']}'")
            continue

        items = data.get("objects", [])
        log.info(f"  CH '{theme_conf['theme']}': {len(items)} items")

        obj_ids = []
        for item in items:
            oid = safe_str(item.get("id", ""))
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)

            title = safe_str(item.get("title", ""))
            if not title:
                continue

            img_url = ""
            images = item.get("images", [])
            if images and isinstance(images, list):
                img_url = safe_str(images[0].get("n", {}).get("url", ""))

            obj = make_obj(
                "cooperhewitt", oid, title,
                date=safe_str(item.get("date", "")),
                culture=safe_str(item.get("country", "")),
                medium=safe_str(item.get("medium", "")),
                dept=safe_str(item.get("department", "")),
                desc=safe_str(item.get("description", "")),
                img_url=img_url,
                classification=safe_str(item.get("type", "")),
                period=safe_str(item.get("period", "")),
            )
            objects.append(obj)
            obj_ids.append(f"cooperhewitt_{oid}")

        if len(obj_ids) >= 5:
            exh_id = f"ch_{theme_conf['dept'].lower().replace(' ','_').replace(',','')[:30]}"
            exhibitions.append({
                "id": exh_id,
                "source": "cooperhewitt",
                "title": theme_conf["theme"],
                "theme": theme_conf["theme"],
                "description": f"Cooper Hewitt Design Museum: {theme_conf['theme'].lower()}.",
                "object_ids": obj_ids,
            })
        time.sleep(0.5)

    log.info(f"Cooper Hewitt: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ──────────────────────────────────────────────────────────────────────────────
# 6. Finnish National Gallery (Finna API)
# ──────────────────────────────────────────────────────────────────────────────

def collect_finna(max_per_theme: int = 100) -> tuple[list, list]:
    """Finna API — Finnish National Gallery (Ateneum, Kiasma, Sinebrychoff)."""
    log.info("=== Collecting Finnish National Gallery (Finna) data ===")
    FINNA_API = "https://api.finna.fi/v1/search"

    THEMES = [
        {"theme": "Finnish Landscape Painting", "q": "Finnish landscape painting nature", "building": "0/Ateneum/"},
        {"theme": "Nordic Modern Art", "q": "Nordic modern contemporary abstract", "building": "0/Kiasma/"},
        {"theme": "European Old Masters at Sinebrychoff", "q": "Dutch Flemish Italian portrait baroque", "building": "0/Sinebrychoff/"},
        {"theme": "Finnish National Romantic Art", "q": "national romantic Gallen-Kallela mythology", "building": "0/Ateneum/"},
        {"theme": "Sculpture and Decorative Arts", "q": "sculpture bronze ceramic glass decorative", "building": "0/Ateneum/"},
    ]

    exhibitions, objects = [], []
    seen_ids = set()

    for theme_conf in THEMES:
        params = {
            "lookfor": theme_conf["q"],
            "filter[]": f'building:"{theme_conf["building"]}"',
            "limit": max_per_theme,
            "lng": "en",
            "field[]": ["id", "title", "summary", "year", "authors", "subjects", "images", "formats"],
        }
        url = f"{FINNA_API}?" + urllib.parse.urlencode(params, doseq=True)
        data = http_get(url, timeout=20)
        if not data or "records" not in data:
            log.warning(f"Finna: no data for '{theme_conf['theme']}'")
            continue

        records = data["records"]
        log.info(f"  Finna '{theme_conf['theme']}': {len(records)} items")

        obj_ids = []
        for rec in records:
            oid = safe_str(rec.get("id", ""))
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)

            title = safe_str(rec.get("title", ""))
            if not title:
                continue

            authors = rec.get("authors", {})
            author_str = ""
            if isinstance(authors, dict):
                for cat in authors.values():
                    if isinstance(cat, dict):
                        author_str = ", ".join(cat.keys())
                        break

            imgs = rec.get("images", [])
            img_url = f"https://api.finna.fi{imgs[0]}" if imgs else ""

            obj = make_obj(
                "finna", oid, title,
                date=safe_str(rec.get("year", "")),
                culture="Finnish/Nordic",
                medium=author_str,
                dept=theme_conf["building"].strip("/").replace("0/",""),
                desc=" ".join(rec.get("summary", [])),
                img_url=img_url,
            )
            objects.append(obj)
            obj_ids.append(f"finna_{oid}")

        if len(obj_ids) >= 5:
            exh_id = f"finna_{theme_conf['theme'].lower().replace(' ','_')[:30]}"
            exhibitions.append({
                "id": exh_id,
                "source": "finna",
                "title": theme_conf["theme"],
                "theme": theme_conf["theme"],
                "description": f"Finnish National Gallery: {theme_conf['theme'].lower()}.",
                "object_ids": obj_ids,
            })
        time.sleep(0.3)

    log.info(f"Finna: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ──────────────────────────────────────────────────────────────────────────────
# 7. MoMA (Museum of Modern Art) GitHub open data
# ──────────────────────────────────────────────────────────────────────────────

def collect_moma(max_per_dept: int = 300) -> tuple[list, list]:
    """MoMA open data from GitHub CSV (CC0)."""
    log.info("=== Collecting MoMA data ===")

    # MoMA provides CSV data on GitHub, but we parse the JSON API alternative
    # Using their searchable collection endpoint
    MOMA_API = "https://api.moma.org/api/v2"

    MOMA_THEMES = [
        {"theme": "Abstract Expressionism", "q": "Abstract Expressionism action painting", "dept": "Painting and Sculpture"},
        {"theme": "Graphic Design and Typography", "q": "poster typography graphic design", "dept": "Drawings and Prints"},
        {"theme": "Photography: Documentary and Art", "q": "documentary street photography portrait", "dept": "Photography"},
        {"theme": "Surrealism and Dada", "q": "Surrealism Dada dreamlike unconscious", "dept": "Painting and Sculpture"},
        {"theme": "Architecture and Design", "q": "architecture model blueprint industrial", "dept": "Architecture and Design"},
        {"theme": "Film and Media Arts", "q": "film animation video media digital", "dept": "Film"},
        {"theme": "Latin American Modern Art", "q": "Latin American Brazilian Mexican Rivera Kahlo", "dept": "Painting and Sculpture"},
        {"theme": "Minimalism and Conceptual Art", "q": "minimalism conceptual installation process", "dept": "Painting and Sculpture"},
        {"theme": "Japanese Avant-Garde", "q": "Japanese avant-garde Gutai Mono-ha", "dept": "Painting and Sculpture"},
        {"theme": "Prints and Multiples", "q": "lithograph screenprint woodblock etching", "dept": "Drawings and Prints"},
    ]

    exhibitions, objects = [], []
    seen_ids = set()

    # MoMA public collection search
    for theme_conf in MOMA_THEMES:
        url = f"https://www.moma.org/collection/works.json?q={urllib.parse.quote(theme_conf['q'])}&classifications={urllib.parse.quote(theme_conf['dept'])}&with_images=1&per_page={max_per_dept}"
        data = http_get(url, timeout=30)

        if not data:
            # Try alternate MOMA API
            url2 = f"https://api.moma.org/artworks?q={urllib.parse.quote(theme_conf['q'])}&limit={max_per_dept}"
            data = http_get(url2, timeout=30)

        if not data:
            log.warning(f"MoMA: no data for '{theme_conf['theme']}'")
            continue

        works = []
        if isinstance(data, dict):
            works = data.get("works", data.get("artworks", data.get("data", [])))
        elif isinstance(data, list):
            works = data

        log.info(f"  MoMA '{theme_conf['theme']}': {len(works)} items")

        obj_ids = []
        for work in works[:max_per_dept]:
            oid = str(work.get("id", work.get("objectID", "")))
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)

            title = safe_str(work.get("title", ""))
            if not title:
                continue

            # Artist as culture proxy
            artists = work.get("artists", [])
            nationality = ""
            if artists and isinstance(artists, list) and isinstance(artists[0], dict):
                nationality = safe_str(artists[0].get("nationality", ""))

            img = work.get("imageUrls", work.get("thumbnail_url", work.get("primary_image", "")))
            if isinstance(img, dict):
                img_url = safe_str(img.get("medium", img.get("small", "")))
            else:
                img_url = safe_str(img)

            obj = make_obj(
                "moma", oid, title,
                date=safe_str(work.get("date", work.get("dateAcquired", ""))),
                culture=nationality or "Modern/Contemporary",
                medium=safe_str(work.get("medium", "")),
                dept=safe_str(work.get("department", theme_conf["dept"])),
                desc=safe_str(work.get("creditLine", "")),
                img_url=img_url,
                classification=safe_str(work.get("classification", "")),
            )
            objects.append(obj)
            obj_ids.append(f"moma_{oid}")

        if len(obj_ids) >= 5:
            exh_id = f"moma_{theme_conf['theme'].lower().replace(' ','_').replace(':','')[:30]}"
            exhibitions.append({
                "id": exh_id,
                "source": "moma",
                "title": theme_conf["theme"],
                "theme": theme_conf["theme"],
                "description": f"MoMA collection: {theme_conf['theme'].lower()}.",
                "object_ids": obj_ids,
            })
        time.sleep(0.3)

    log.info(f"MoMA: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ──────────────────────────────────────────────────────────────────────────────
# 8. LACMA (Los Angeles County Museum of Art) API
# ──────────────────────────────────────────────────────────────────────────────

def collect_lacma(max_per_theme: int = 150) -> tuple[list, list]:
    """LACMA open collection via their API."""
    log.info("=== Collecting LACMA data ===")
    LACMA_API = "https://collections.lacma.org/search/site"

    THEMES = [
        {"theme": "Ancient Art of the Americas", "q": "pre-Columbian Maya Aztec Inca Andean ceramic"},
        {"theme": "Islamic Art and Design", "q": "Islamic Persian Ottoman tile calligraphy carpet"},
        {"theme": "South and Southeast Asian Art", "q": "Indian Hindu Buddhist Cambodia Thai sculpture"},
        {"theme": "East Asian Art: China and Japan", "q": "Chinese scroll Japanese ink lacquer bronze jade"},
        {"theme": "European Decorative Arts", "q": "European furniture silver porcelain tapestry baroque"},
        {"theme": "American Art from Diverse Traditions", "q": "American painting folk art photography contemporary"},
        {"theme": "Photography and New Media", "q": "photography video digital contemporary art"},
        {"theme": "Costume and Textiles", "q": "costume fashion textile weaving embroidery dress"},
        {"theme": "African Art and Ritual Objects", "q": "African mask sculpture ritual figure bronze"},
        {"theme": "Modern Art: 20th Century", "q": "modern abstract expressionism pop art sculpture"},
    ]

    exhibitions, objects = [], []
    seen_ids = set()

    for theme_conf in THEMES:
        # LACMA API
        params = {
            "f[0]": f"field_has_image:1",
            "q": theme_conf["q"],
            "limit": max_per_theme,
        }
        url = f"{LACMA_API}?" + urllib.parse.urlencode(params)
        data = http_get(url, timeout=20)

        if not data:
            # alternate LACMA search
            alt = f"https://api.lacma.org/api/v2/artworks?q={urllib.parse.quote(theme_conf['q'])}&hasImage=1&limit={max_per_theme}"
            data = http_get(alt, timeout=20)

        if not data:
            log.warning(f"LACMA: no data for '{theme_conf['theme']}'")
            continue

        items = []
        if isinstance(data, dict):
            items = data.get("results", data.get("artworks", data.get("data", [])))
        elif isinstance(data, list):
            items = data

        log.info(f"  LACMA '{theme_conf['theme']}': {len(items)} items")

        obj_ids = []
        for item in items[:max_per_theme]:
            oid = str(item.get("id", item.get("objectID", item.get("nid", ""))))
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)

            title = safe_str(item.get("title", ""))
            if not title:
                continue

            obj = make_obj(
                "lacma", oid, title,
                date=safe_str(item.get("date", item.get("creation_date", ""))),
                culture=safe_str(item.get("culture", item.get("nationality", ""))),
                medium=safe_str(item.get("medium", item.get("technique", ""))),
                dept=safe_str(item.get("classification", item.get("department", ""))),
                desc=safe_str(item.get("description", item.get("creditline", ""))),
                img_url=safe_str(item.get("image_url", item.get("thumbnail", ""))),
                classification=safe_str(item.get("classification", "")),
            )
            objects.append(obj)
            obj_ids.append(f"lacma_{oid}")

        if len(obj_ids) >= 5:
            exh_id = f"lacma_{theme_conf['theme'].lower().replace(' ','_').replace(':','')[:30]}"
            exhibitions.append({
                "id": exh_id,
                "source": "lacma",
                "title": theme_conf["theme"],
                "theme": theme_conf["theme"],
                "description": f"LACMA collection: {theme_conf['theme'].lower()}.",
                "object_ids": obj_ids,
            })
        time.sleep(0.3)

    log.info(f"LACMA: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ──────────────────────────────────────────────────────────────────────────────
# 9. Extend existing AIC + Cleveland with more themes
# ──────────────────────────────────────────────────────────────────────────────

def collect_aic_extended(max_per_query: int = 200) -> tuple[list, list]:
    """AIC extended — additional themes not covered in v2."""
    log.info("=== Collecting AIC Extended data ===")
    AIC_API = "https://api.artic.edu/api/v1/artworks"

    EXTRA_THEMES = [
        {"theme": "African and Ancient American Art", "q": "Africa mask ritual Americas pre-Columbian"},
        {"theme": "Asian Decorative Arts and Design", "q": "Japanese ink screen woodblock Chinese celadon"},
        {"theme": "Photography: Modernism to Now", "q": "photographic print gelatin silver documentary"},
        {"theme": "Ancient Greek and Roman Art", "q": "ancient Greek Roman vessel bronze marble"},
        {"theme": "Arms, Armor, and Military History", "q": "armor sword weapon helmet shield"},
        {"theme": "Religious Art and Devotional Objects", "q": "altarpiece icon devotional religious gold"},
        {"theme": "Prints from Dürer to Picasso", "q": "print etching lithograph engraving woodcut"},
        {"theme": "Modernist Sculpture", "q": "modernist sculpture bronze cast abstract figure"},
        {"theme": "Architecture and Architectural Drawing", "q": "architectural drawing blueprint model elevation"},
        {"theme": "Fashion and Costume History", "q": "dress costume fashion garment textile historic"},
    ]

    exhibitions, objects = [], []
    seen_ids = set()

    for theme_conf in EXTRA_THEMES:
        params = {
            "q": theme_conf["q"],
            "limit": max_per_query,
            "fields": "id,title,date_display,culture_display,medium_display,department_title,description,image_id,classification_title,style_title",
        }
        url = f"{AIC_API}?" + urllib.parse.urlencode(params)
        data = http_get(url, timeout=25)
        if not data:
            continue

        items = data.get("data", [])
        log.info(f"  AIC extra '{theme_conf['theme']}': {len(items)} items")

        obj_ids = []
        for item in items:
            oid = str(item.get("id", ""))
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)

            title = safe_str(item.get("title", ""))
            if not title:
                continue

            img_id = item.get("image_id", "")
            img_url = f"https://www.artic.edu/iiif/2/{img_id}/full/400,/0/default.jpg" if img_id else ""

            obj = make_obj(
                "aic_ext", oid, title,
                date=safe_str(item.get("date_display", "")),
                culture=safe_str(item.get("culture_display", "")),
                medium=safe_str(item.get("medium_display", "")),
                dept=safe_str(item.get("department_title", "")),
                desc=safe_str(item.get("description", "")),
                img_url=img_url,
                classification=safe_str(item.get("classification_title", "")),
                period=safe_str(item.get("style_title", "")),
            )
            objects.append(obj)
            obj_ids.append(f"aic_ext_{oid}")

        if len(obj_ids) >= 5:
            exh_id = f"aic_ext_{theme_conf['theme'].lower().replace(' ','_').replace(':','')[:30]}"
            exhibitions.append({
                "id": exh_id,
                "source": "aic_ext",
                "title": theme_conf["theme"],
                "theme": theme_conf["theme"],
                "description": f"AIC collection (extended): {theme_conf['theme'].lower()}.",
                "object_ids": obj_ids,
            })
        time.sleep(0.3)

    log.info(f"AIC Extended: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ──────────────────────────────────────────────────────────────────────────────
# 10. Europeana Extended — more themes
# ──────────────────────────────────────────────────────────────────────────────

def collect_europeana_extended(max_per_theme: int = 100) -> tuple[list, list]:
    """Europeana extended with additional cultural themes."""
    log.info("=== Collecting Europeana Extended data ===")
    EUR_API = "https://api.europeana.eu/record/v2/search.json"
    EUR_KEY = "apidemo"

    EXTRA_THEMES = [
        {"theme": "Ottoman and Byzantine Heritage", "q": "Ottoman Byzantine Constantinople Istanbul manuscript"},
        {"theme": "Celtic and Viking Art", "q": "Celtic Viking Norse Nordic brooch helmet sword"},
        {"theme": "Medieval Manuscripts and Books", "q": "illuminated manuscript book of hours Bible psalter"},
        {"theme": "Baroque Music and Theater", "q": "baroque opera theater music manuscript score"},
        {"theme": "European Folk Art and Craft", "q": "folk art embroidery traditional craft peasant"},
        {"theme": "Maps and Cartography", "q": "map cartography atlas geography navigation chart"},
        {"theme": "Scientific Instruments and Curiosity Cabinets", "q": "scientific instrument astrolabe globe cabinet"},
        {"theme": "African and Colonial Collections", "q": "Africa colonial ethnographic collection mask figure"},
        {"theme": "Photography: 19th Century Europe", "q": "albumen print calotype daguerreotype portrait 19th"},
        {"theme": "Art Nouveau and Arts and Crafts", "q": "Art Nouveau Arts Crafts Jugendstil floral design"},
    ]

    exhibitions, objects = [], []
    seen_ids = set()

    for theme_conf in EXTRA_THEMES:
        params = {
            "wskey": EUR_KEY,
            "query": theme_conf["q"],
            "rows": max_per_theme,
            "profile": "rich",
            "qf": "TYPE:IMAGE",
        }
        url = f"{EUR_API}?" + urllib.parse.urlencode(params)
        data = http_get(url, timeout=20)
        if not data or not data.get("success"):
            log.warning(f"Europeana ext: failed for '{theme_conf['theme']}'")
            continue

        items = data.get("items", [])
        log.info(f"  Europeana ext '{theme_conf['theme']}': {len(items)} items")

        obj_ids = []
        for item in items:
            oid = safe_str(item.get("id", "")).replace("/", "_")
            if not oid or oid in seen_ids:
                continue
            seen_ids.add(oid)

            titles = item.get("title", item.get("dcTitle", []))
            title = titles[0] if titles else ""
            if not title:
                continue

            desc_list = item.get("dcDescription", item.get("description", []))
            desc = desc_list[0] if isinstance(desc_list, list) and desc_list else safe_str(desc_list)

            dates = item.get("year", item.get("dctermsCreated", []))
            date_str = dates[0] if isinstance(dates, list) and dates else safe_str(dates)

            langs = item.get("dataProvider", [])
            dp = langs[0] if isinstance(langs, list) and langs else ""

            thumbs = item.get("edmPreview", [])
            img_url = thumbs[0] if thumbs else ""

            obj = make_obj(
                "europeana_ext", oid, title,
                date=date_str,
                culture=safe_str(item.get("country", [""])[0] if isinstance(item.get("country"), list) else ""),
                medium=safe_str(item.get("dcFormat", [""])[0] if isinstance(item.get("dcFormat"), list) else ""),
                dept=safe_str(dp),
                desc=desc,
                img_url=img_url,
            )
            objects.append(obj)
            obj_ids.append(f"europeana_ext_{oid}")

        if len(obj_ids) >= 5:
            exh_id = f"eur_ext_{theme_conf['theme'].lower().replace(' ','_').replace(':','').replace('&','and')[:30]}"
            exhibitions.append({
                "id": exh_id,
                "source": "europeana_ext",
                "title": theme_conf["theme"],
                "theme": theme_conf["theme"],
                "description": f"Europeana (extended): {theme_conf['theme'].lower()}.",
                "object_ids": obj_ids,
            })
        time.sleep(0.3)

    log.info(f"Europeana Extended: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ──────────────────────────────────────────────────────────────────────────────
# Merge + Stats
# ──────────────────────────────────────────────────────────────────────────────

def merge_with_v2(
    new_exhs: list[dict],
    new_objs: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Merge new data with existing v2 dataset, deduplicate."""
    existing_exhs = load_jsonl(DATA / "exhibitions_v2.jsonl")
    existing_objs = load_jsonl(DATA / "objects_v2.jsonl")

    existing_exh_ids = {e["id"] for e in existing_exhs}
    existing_obj_ids = {o["id"] for o in existing_objs}

    added_exhs = [e for e in new_exhs if e["id"] not in existing_exh_ids]
    added_objs = [o for o in new_objs if o["id"] not in existing_obj_ids]

    all_exhs = existing_exhs + added_exhs
    all_objs = existing_objs + added_objs

    log.info(f"Merge: +{len(added_exhs)} exhibitions, +{len(added_objs)} objects")
    log.info(f"Total: {len(all_exhs)} exhibitions, {len(all_objs)} objects")
    return all_exhs, all_objs


def print_stats(exhibitions: list[dict], objects: list[dict]) -> None:
    from collections import Counter

    src_exh = Counter(e.get("source", "?") for e in exhibitions)
    src_obj = Counter(o.get("source", "?") for o in objects)
    cult_obj = Counter(o.get("culture", "Other") for o in objects)

    # Count TES-eligible
    obj_ids_all = {o["id"] for o in objects}
    tes_eligible = sum(
        1 for e in exhibitions
        if len([oid for oid in e.get("object_ids", []) if oid in obj_ids_all]) >= 5
    )

    print("\n" + "=" * 60)
    print("ExhibitionBench Dataset Statistics (v3)")
    print("=" * 60)
    print(f"Total exhibitions : {len(exhibitions)}")
    for src, cnt in src_exh.most_common():
        print(f"  {src:22s} : {cnt}")
    print(f"\nTotal objects     : {len(objects)}")
    for src, cnt in src_obj.most_common():
        print(f"  {src:22s} : {cnt}")
    print(f"\nCultural coverage:")
    total = len(objects)
    for culture, cnt in cult_obj.most_common(12):
        print(f"  {culture:25s} : {cnt:5d} ({cnt/total*100:.1f}%)")
    print(f"\nTES-eligible exhibitions (>=5 objects): {tes_eligible}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["smithsonian","harvard","nga","rijksmuseum",
                                              "cooperhewitt","finna","moma","lacma",
                                              "aic_ext","europeana_ext","all"], default="all")
    parser.add_argument("--merge-v2", action="store_true", default=True)
    parser.add_argument("--no-merge-v2", dest="merge_v2", action="store_false")
    parser.add_argument("--out-exh", default="data/exhibitions_v3.jsonl")
    parser.add_argument("--out-obj", default="data/objects_v3.jsonl")
    args = parser.parse_args()

    collectors = {
        "smithsonian": collect_smithsonian,
        "harvard":     collect_harvard,
        "nga":         collect_nga,
        "rijksmuseum": collect_rijksmuseum,
        "cooperhewitt": collect_cooperhewitt,
        "finna":       collect_finna,
        "moma":        collect_moma,
        "lacma":       collect_lacma,
        "aic_ext":     collect_aic_extended,
        "europeana_ext": collect_europeana_extended,
    }

    sources = list(collectors.keys()) if args.source == "all" else [args.source]

    all_new_exhs, all_new_objs = [], []
    for src in sources:
        try:
            exhs, objs = collectors[src]()
            all_new_exhs.extend(exhs)
            all_new_objs.extend(objs)
        except Exception as e:
            log.error(f"Source {src} failed: {e}", exc_info=True)

    if args.merge_v2:
        final_exhs, final_objs = merge_with_v2(all_new_exhs, all_new_objs)
    else:
        final_exhs, final_objs = all_new_exhs, all_new_objs

    out_exh = BASE / args.out_exh
    out_obj = BASE / args.out_obj
    write_jsonl(out_exh, final_exhs)
    write_jsonl(out_obj, final_objs)

    print_stats(final_exhs, final_objs)
    print(f"\nOutputs:")
    print(f"  Exhibitions : {out_exh}")
    print(f"  Objects     : {out_obj}")


if __name__ == "__main__":
    main()

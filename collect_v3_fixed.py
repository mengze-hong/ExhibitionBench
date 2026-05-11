"""
collect_v3_fixed.py — ExhibitionBench v3 dataset expander
==========================================================
Uses only confirmed-working APIs:
  - AIC  (Art Institute of Chicago)  → target +120 exhibitions, +6000 objects
  - Cleveland Museum of Art          → target +40  exhibitions, +4000 objects
  - V&A  (Victoria & Albert Museum)  → target +40  exhibitions, +4000 objects
  - Europeana                        → target +60  exhibitions, +6000 objects
  - Met  Museum                      → target +40  exhibitions, +4000 objects

Grand total target: v2(132) + 300 new ≈ 432 exhibitions, 25000+ objects

Usage:
  python collect_v3_fixed.py            # all sources
  python collect_v3_fixed.py --source aic
  python collect_v3_fixed.py --source cleveland vam europeana met
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

log = logging.getLogger(__name__)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

# ── HTTP helpers ─────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5)))
HEADERS = {"User-Agent": "ExhibitionBenchResearch/3.0 (academic)"}

def http_get(url: str, params: dict = None, timeout: int = 20) -> Optional[dict]:
    try:
        r = SESSION.get(url, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.debug(f"http_get {url}: {e}")
    return None


# ── Schema helpers ────────────────────────────────────────────────────────────

CULTURE_KEYWORDS = {
    "East Asian":        ["chinese", "japanese", "korean", "china", "japan", "korea",
                          "ming", "qing", "tang", "edo", "meiji"],
    "South Asian":       ["indian", "india", "mughal", "hindu", "buddhist", "south asian",
                          "persian", "iranian"],
    "Islamic":           ["islamic", "ottoman", "persian", "arabic", "muslim", "iran",
                          "turkey", "cairo"],
    "African":           ["african", "africa", "nigerian", "yoruba", "benin", "mali",
                          "senegal", "kongo", "akan"],
    "Ancient":           ["ancient", "roman", "greek", "egyptian", "mesopotamian",
                          "byzantine", "etruscan", "babylonian"],
    "Western European":  ["french", "dutch", "flemish", "german", "italian", "spanish",
                          "british", "english", "renaissance", "baroque"],
    "American":          ["american", "colonial", "united states", "native american",
                          "latin american", "mesoamerican"],
    "South American":    ["south american", "andean", "inca", "aztec", "maya",
                          "pre-columbian", "peruvian"],
}

def infer_culture(text: str) -> str:
    t = text.lower()
    for group, kws in CULTURE_KEYWORDS.items():
        if any(k in t for k in kws):
            return group
    return "Other"

def make_exh(source, exh_id, title, theme, description, object_ids) -> dict:
    return {
        "id": f"{source}_{exh_id}",
        "source": source,
        "title": title,
        "theme": theme,
        "description": description,
        "object_ids": [f"{source}_{oid}" for oid in object_ids],
    }

def make_obj(source, oid, title, date="", culture="", medium="",
             department="", description="", image_url="",
             classification="", period="") -> dict:
    combined = f"{title} {culture} {department} {medium} {description}"
    if not culture:
        culture = infer_culture(combined)
    return {
        "id": f"{source}_{oid}",
        "source": source,
        "title": title or "Untitled",
        "date": date,
        "culture": culture,
        "medium": medium,
        "department": department,
        "description": description[:500] if description else "",
        "image_url": image_url,
        "classification": classification,
        "period": period,
    }


# ── AIC (Art Institute of Chicago) ────────────────────────────────────────────
# 131K+ objects, 12 departments. Build virtual exhibitions by dept × sub-theme.

AIC_BASE = "https://api.artic.edu/api/v1"
AIC_FIELDS = "id,title,date_display,place_of_origin,department_title,medium_display,description,image_id,classification_title"

AIC_THEMES = [
    # (theme_label, description, department_id or None, q_term, culture_hint)
    ("Ancient Mediterranean Art", "Art of ancient Greece, Rome, and Byzantium", "PC-4", None, "Ancient"),
    ("Greek Vases and Sculpture", "Pottery and sculpture from ancient Greece", "PC-4", "greek vase sculpture", "Ancient"),
    ("Roman Art and Antiquities", "Sculpture, mosaic, and decorative arts of Rome", "PC-4", "roman", "Ancient"),
    ("Byzantine and Early Christian", "Icons and sacred art from Byzantium", "PC-4", "byzantine", "Ancient"),
    ("Japanese Art and Prints", "Woodblock prints and Japanese painting", "PC-7", "japanese", "East Asian"),
    ("Chinese Ceramics and Painting", "Chinese porcelain, jade, and ink painting", "PC-7", "chinese", "East Asian"),
    ("Indian and South Asian Art", "Sculpture and painting from the Indian subcontinent", "PC-7", "indian", "South Asian"),
    ("Islamic Art and Decorative Objects", "Geometric patterns and calligraphy from Islamic world", "PC-7", "islamic", "Islamic"),
    ("Korean Art", "Korean ceramics and painting", "PC-7", "korean", "East Asian"),
    ("Arts of Africa", "Masks, sculpture, and textiles from Africa", "PC-1", None, "African"),
    ("Native American Heritage", "Indigenous art of the Americas", "PC-3", "native american", "American"),
    ("Pre-Columbian Art", "Ancient Mesoamerican and Andean art", "PC-3", "pre-columbian", "American"),
    ("European Old Masters", "Painting from Renaissance through Baroque", "PC-10", "old master", "Western European"),
    ("French Impressionism", "Impressionist painting from 19th-century France", "PC-10", "impressionism", "Western European"),
    ("Dutch and Flemish Masters", "Golden Age painting from the Netherlands", "PC-10", "dutch flemish", "Western European"),
    ("Italian Renaissance", "Painting and sculpture of the Italian Renaissance", "PC-10", "italian renaissance", "Western European"),
    ("Spanish Painting", "Spanish masters including Goya and El Greco", "PC-10", "spanish", "Western European"),
    ("British Art", "Portraiture and landscape from Britain", "PC-10", "british", "Western European"),
    ("German Expressionism", "Expressionist art from Germany", "PC-10", "german expressionism", "Western European"),
    ("American Painting 19th Century", "American landscape and genre painting", "PC-3", "american painting", "American"),
    ("Modern European Painting", "Cubism, Fauvism, and avant-garde movements", "PC-838", "cubism fauvism", "Western European"),
    ("Abstract Expressionism", "Post-war American abstraction", "PC-838", "abstract expressionism", "American"),
    ("Photography Art", "Fine art photography from the collection", "PC-12", None, "Western European"),
    ("Architecture and Design", "Architectural drawings and design objects", "PC-5", None, "Western European"),
    ("Applied Arts: Furniture", "Decorative furniture and woodwork", "PC-9", "furniture", "Western European"),
    ("Applied Arts: Textiles", "European textiles and tapestries", "PC-9", "textile", "Western European"),
    ("Contemporary Art", "Contemporary artworks from around the globe", "PC-8", None, "Other"),
    ("African American Art", "Works by African American artists", "PC-3", "african american artist", "American"),
    ("Surrealism and Dada", "Surrealist and Dadaist works", "PC-838", "surrealism dada", "Western European"),
    ("Prints and Drawings", "Works on paper: prints, drawings, watercolors", None, "print drawing watercolor", "Western European"),
    ("Medieval European Art", "Art from the medieval period in Europe", "PC-10", "medieval", "Western European"),
    ("Art Nouveau and Art Deco", "Decorative arts of the early 20th century", "PC-9", "art nouveau art deco", "Western European"),
    ("Bauhaus and Constructivism", "Bauhaus design and Constructivist art", "PC-838", "bauhaus constructivism", "Western European"),
    ("Latin American Modern Art", "20th-century art from Latin America", "PC-3", "latin american", "American"),
    ("Pacific Islander Art", "Art from Oceania and Pacific cultures", "PC-1", "oceanic pacific", "Other"),
    ("Jewish and Ceremonial Art", "Judaica and ceremonial objects", None, "judaica ceremonial", "Western European"),
    ("Armor and Weapons", "European and Asian arms and armor", None, "armor weapon", "Western European"),
    ("Portrait Miniatures", "Small-scale portrait miniatures", None, "portrait miniature", "Western European"),
    ("Landscape Painting", "Landscape painting across centuries", "PC-10", "landscape painting", "Western European"),
    ("Still Life Painting", "Still life works from Europe and beyond", "PC-10", "still life", "Western European"),
    ("Religious and Devotional Art", "Religious paintings, altarpieces, and icons", "PC-10", "religious devotional", "Western European"),
    ("Symbolism and Pre-Raphaelites", "Late 19th-century figurative art", "PC-10", "symbolism pre-raphaelite", "Western European"),
    ("Pointillism and Neo-Impressionism", "Dot-based techniques of late Impressionism", "PC-10", "pointillism neo-impressionism", "Western European"),
    ("Sculpture: Bronze and Stone", "Sculptural works in bronze and stone", None, "bronze sculpture marble", "Western European"),
    ("Asian Lacquerware and Metalwork", "Lacquer, enamel, and metalwork from Asia", "PC-7", "lacquer metalwork enamel", "East Asian"),
    ("African Textiles and Beadwork", "Textiles, beadwork, and fiber arts from Africa", "PC-1", "textile beadwork", "African"),
    ("Egyptian and Near Eastern Art", "Ancient Egypt and the Near East", "PC-4", "egypt", "Ancient"),
    ("Decorative Arts: Ceramics", "Ceramics and porcelain from global traditions", None, "ceramics porcelain", "East Asian"),
    ("Minimalism and Pop Art", "Post-war minimalism and Pop art", "PC-838", "minimalism pop art", "American"),
    ("Video and New Media Art", "Digital and new media artworks", "PC-8", "video media", "Other"),
]


def collect_aic_extended(max_per_theme: int = 150) -> tuple[list, list]:
    """Collect 50 virtual exhibitions from AIC, up to max_per_theme objects each."""
    exhibitions, objects = [], []
    obj_seen = set()

    for theme_label, description, dept_id, q_term, culture_hint in tqdm(AIC_THEMES, desc="AIC themes"):
        params = {
            "fields": AIC_FIELDS,
            "limit": min(max_per_theme, 100),
            "page": 1,
        }
        if dept_id:
            params["department_ids[]"] = dept_id
        if q_term:
            # Use search endpoint
            url = f"{AIC_BASE}/artworks/search"
            params["q"] = q_term
        else:
            url = f"{AIC_BASE}/artworks"

        theme_objs = []
        for page in range(1, 4):  # up to 3 pages = 300 objects
            params["page"] = page
            d = http_get(url, params)
            if not d or not d.get("data"):
                break
            for item in d["data"]:
                oid = str(item["id"])
                if oid in obj_seen:
                    continue
                obj_seen.add(oid)
                img = ""
                if item.get("image_id"):
                    img = f"https://www.artic.edu/iiif/2/{item['image_id']}/full/843,/0/default.jpg"
                obj = make_obj(
                    "aic", oid,
                    title=item.get("title", ""),
                    date=item.get("date_display", ""),
                    culture=item.get("place_of_origin", "") or culture_hint,
                    medium=item.get("medium_display", ""),
                    department=item.get("department_title", ""),
                    description=item.get("description", "") or "",
                    image_url=img,
                    classification=item.get("classification_title", ""),
                )
                theme_objs.append(obj)
            if len(theme_objs) >= max_per_theme:
                break
            time.sleep(0.1)

        if len(theme_objs) < 5:
            continue

        theme_objs = theme_objs[:max_per_theme]
        objects.extend(theme_objs)
        exh_id = theme_label.lower().replace(" ", "_").replace(",", "").replace("&", "and")
        exh = make_exh(
            "aic_ext", exh_id,
            title=f"AIC: {theme_label}",
            theme=theme_label,
            description=description,
            object_ids=[o["id"].replace("aic_ext_", "aic_") if o["id"].startswith("aic_ext_aic_") else
                        o["id"].replace("aic_", "") for o in theme_objs],
        )
        # Fix object_ids (they're already prefixed via make_obj)
        exh["object_ids"] = [o["id"] for o in theme_objs]
        exhibitions.append(exh)

    log.info(f"AIC extended: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ── Cleveland Museum of Art ────────────────────────────────────────────────────

CMA_BASE = "https://openaccess-api.clevelandart.org/api/artworks/"

CMA_THEMES = [
    # (theme, query_params_dict, description)
    ("Painting: European", {"type": "Painting", "culture": "Dutch", "has_image": 1}, "Dutch paintings from the collection"),
    ("Painting: European French", {"type": "Painting", "culture": "French", "has_image": 1}, "French paintings"),
    ("Painting: European Italian", {"type": "Painting", "culture": "Italian", "has_image": 1}, "Italian paintings"),
    ("Painting: American", {"type": "Painting", "culture": "American", "has_image": 1}, "American paintings"),
    ("Painting: Asian Chinese", {"type": "Painting", "culture": "Chinese", "has_image": 1}, "Chinese paintings"),
    ("Painting: Asian Japanese", {"type": "Painting", "culture": "Japanese", "has_image": 1}, "Japanese paintings"),
    ("Drawing and Watercolor", {"type": "Drawing", "has_image": 1}, "Works on paper, drawings and watercolors"),
    ("Print: European", {"type": "Print", "culture": "European", "has_image": 1}, "European prints and engravings"),
    ("Sculpture: Ancient", {"type": "Sculpture", "has_image": 1, "department": "Ancient Art"}, "Ancient sculpture"),
    ("Photography", {"type": "Photograph", "has_image": 1}, "Photography from the collection"),
    ("Textile and Fiber Arts", {"type": "Textile", "has_image": 1}, "Textiles from global traditions"),
    ("Ceramics and Glass", {"type": "Ceramic", "has_image": 1}, "Ceramics and glass objects"),
    ("Arms and Armor", {"type": "Arms and Armor", "has_image": 1}, "Historical weapons and armor"),
    ("Coins and Medals", {"type": "Coin", "has_image": 1}, "Numismatic collection"),
    ("Furniture and Decorative Arts", {"type": "Furniture", "has_image": 1}, "Decorative furniture and interiors"),
    ("Egyptian and Near Eastern Artifacts", {"department": "Egyptian and Ancient Near Eastern Art", "has_image": 1}, "Antiquities from Egypt and Near East"),
    ("Indian and Southeast Asian Art", {"department": "Indian and Southeast Asian Art", "has_image": 1}, "Art from South and Southeast Asia"),
    ("Islamic Art", {"department": "Islamic Art", "has_image": 1}, "Art from the Islamic world"),
    ("African Art", {"department": "African Art", "has_image": 1}, "Art from sub-Saharan Africa"),
    ("Pre-Columbian Art", {"department": "Pre-Columbian", "has_image": 1}, "Art from Pre-Columbian Americas"),
    ("Japanese Prints and Drawings", {"department": "Japanese Art", "has_image": 1}, "Japanese woodblock prints and drawings"),
    ("Chinese Art", {"department": "Chinese Art", "has_image": 1}, "Art from China across dynasties"),
    ("Korean Art", {"department": "Korean Art", "has_image": 1}, "Korean ceramics and painting"),
    ("Medieval European Art", {"department": "Medieval Art", "has_image": 1}, "Medieval European art and manuscripts"),
    ("Modern and Contemporary", {"department": "Modern European Painting and Drawing", "has_image": 1}, "Modern European art"),
    ("Graphic Arts", {"department": "Prints", "has_image": 1}, "Prints and graphic arts"),
    ("Greek and Roman Art", {"department": "Greek and Roman Art", "has_image": 1}, "Classical antiquity from Greece and Rome"),
    ("American Painting", {"department": "American Painting and Sculpture", "has_image": 1}, "American painting and sculpture"),
    ("Contemporary Art CMA", {"department": "Contemporary Art", "has_image": 1}, "Contemporary art from around the world"),
    ("Decorative Art and Design", {"department": "Decorative Art and Design", "has_image": 1}, "Decorative art and design objects"),
    ("Dutch Painting Golden Age", {"department": "European Paintings and Drawings", "culture": "Dutch", "has_image": 1}, "Dutch Golden Age painting"),
    ("French Impressionism CMA", {"department": "European Paintings and Drawings", "culture": "French", "has_image": 1}, "French impressionist painting"),
    ("Photography Modern", {"department": "Photography", "has_image": 1}, "Modern and contemporary photography"),
    ("Numismatics and Medals", {"department": "Numismatics", "has_image": 1}, "Coins, medals, and seals"),
    ("Oceanic and Native American", {"department": "Oceanic Art", "has_image": 1}, "Art from Oceania and Native Americans"),
    ("Works on Paper", {"type": "Drawing", "department": "Prints", "has_image": 1}, "Works on paper from the collection"),
    ("Sculpture: Modern", {"type": "Sculpture", "department": "Modern European Painting and Drawing", "has_image": 1}, "Modern European sculpture"),
    ("Ancient Mediterranean", {"department": "Greek and Roman Art", "type": "Sculpture", "has_image": 1}, "Ancient Mediterranean sculpture"),
    ("Manuscripts and Books", {"type": "Manuscript", "has_image": 1}, "Illuminated manuscripts and rare books"),
    ("Jewelry and Metalwork", {"type": "Jewelry", "has_image": 1}, "Jewelry and precious metalwork"),
]


def collect_cleveland_extended(max_per_theme: int = 150) -> tuple[list, list]:
    exhibitions, objects = [], []
    obj_seen = set()

    for theme, extra_params, description in tqdm(CMA_THEMES, desc="Cleveland themes"):
        params = {"limit": 100, "skip": 0, **extra_params}
        theme_objs = []

        for skip in range(0, max_per_theme, 100):
            params["skip"] = skip
            d = http_get(CMA_BASE, params)
            if not d or not d.get("data"):
                break
            for item in d["data"]:
                oid = str(item.get("athena_id") or item.get("accession_number", ""))
                if not oid or oid in obj_seen:
                    continue
                obj_seen.add(oid)
                imgs = item.get("images", {})
                img_url = ""
                if isinstance(imgs, dict):
                    web = imgs.get("web", {}) or imgs.get("print", {}) or {}
                    img_url = web.get("url", "") if isinstance(web, dict) else ""
                elif isinstance(imgs, list) and imgs:
                    img_url = imgs[0].get("url", "")

                culture = item.get("culture", "")
                if isinstance(culture, list):
                    culture = ", ".join(culture)
                creators = item.get("creators", []) or []
                creator_str = "; ".join(
                    c.get("description", "") for c in creators if isinstance(c, dict)
                )
                desc = item.get("description", "") or item.get("did_you_know", "") or creator_str
                obj = make_obj(
                    "cleveland", oid,
                    title=item.get("title", ""),
                    date=item.get("creation_date", ""),
                    culture=culture,
                    medium=item.get("technique", "") or item.get("support_materials", ""),
                    department=item.get("department", ""),
                    description=desc,
                    image_url=img_url,
                    classification=item.get("type", ""),
                )
                theme_objs.append(obj)
            if len(theme_objs) >= max_per_theme:
                break
            time.sleep(0.1)

        if len(theme_objs) < 5:
            continue
        theme_objs = theme_objs[:max_per_theme]
        objects.extend(theme_objs)
        exh_id = theme.lower().replace(" ", "_").replace(":", "").replace(",", "")
        exh = make_exh(
            "cleveland_ext", exh_id,
            title=f"CMA: {theme}",
            theme=theme,
            description=description,
            object_ids=[],
        )
        exh["object_ids"] = [o["id"] for o in theme_objs]
        exhibitions.append(exh)

    log.info(f"Cleveland extended: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ── V&A Museum ───────────────────────────────────────────────────────────────

VAM_BASE = "https://api.vam.ac.uk/v2/objects/search"

VAM_THEMES = [
    ("Ceramics: Japanese Porcelain", {"q": "japanese porcelain", "page_size": 100}, "Japanese porcelain and stoneware", "East Asian"),
    ("Ceramics: Chinese Blue and White", {"q": "chinese blue white porcelain", "page_size": 100}, "Chinese blue-and-white ware", "East Asian"),
    ("Ceramics: Islamic Pottery", {"q": "islamic pottery", "page_size": 100}, "Islamic glazed pottery and tiles", "Islamic"),
    ("Ceramics: European", {"q": "european ceramics", "page_size": 100}, "European faience and porcelain", "Western European"),
    ("Textiles: Indian", {"q": "indian textile", "page_size": 100}, "Indian woven and embroidered textiles", "South Asian"),
    ("Textiles: Islamic Carpets", {"q": "carpet persian ottoman", "page_size": 100}, "Persian and Ottoman carpets", "Islamic"),
    ("Textiles: European Tapestry", {"q": "tapestry european", "page_size": 100}, "European tapestries and embroideries", "Western European"),
    ("Fashion: 18th Century", {"q": "dress 18th century fashion", "page_size": 100}, "18th-century European fashion", "Western European"),
    ("Fashion: Victorian", {"q": "victorian dress fashion", "page_size": 100}, "Victorian-era clothing and accessories", "Western European"),
    ("Jewelry: Ancient", {"q": "ancient jewelry gold", "page_size": 100}, "Ancient jewelry and gold work", "Ancient"),
    ("Jewelry: European", {"q": "european jewelry gems", "page_size": 100}, "European gemstone jewelry", "Western European"),
    ("Furniture: European Baroque", {"q": "baroque furniture", "page_size": 100}, "Baroque-period furniture", "Western European"),
    ("Furniture: East Asian Lacquer", {"q": "japanese lacquer furniture", "page_size": 100}, "Japanese and Chinese lacquerware", "East Asian"),
    ("Metalwork: Islamic", {"q": "islamic metalwork bronze", "page_size": 100}, "Islamic bronzes and metalwork", "Islamic"),
    ("Metalwork: European", {"q": "european silver gold", "page_size": 100}, "European silver and gold metalwork", "Western European"),
    ("Glass: European", {"q": "european glass murano", "page_size": 100}, "European art glass including Murano", "Western European"),
    ("Sculpture: European", {"q": "european sculpture marble", "page_size": 100}, "European marble sculpture", "Western European"),
    ("Prints and Drawings: European", {"q": "european prints drawings", "page_size": 100}, "European prints and drawings", "Western European"),
    ("Photography: 19th Century", {"q": "photography 19th century", "page_size": 100}, "Early photography", "Western European"),
    ("Books and Manuscripts", {"q": "manuscript book illuminated", "page_size": 100}, "Illuminated manuscripts and rare books", "Western European"),
    ("Indian Sculpture", {"q": "indian sculpture bronze", "page_size": 100}, "Indian religious sculpture", "South Asian"),
    ("Chinese Art", {"q": "chinese art painting", "page_size": 100}, "Chinese art and painting", "East Asian"),
    ("Korean Art", {"q": "korean celadon", "page_size": 100}, "Korean celadon and art", "East Asian"),
    ("Islamic Architecture", {"q": "islamic architecture tile", "page_size": 100}, "Islamic architectural tiles and elements", "Islamic"),
    ("African Art", {"q": "african art mask", "page_size": 100}, "African masks and sculpture", "African"),
    ("Art Nouveau", {"q": "art nouveau", "page_size": 100}, "Art Nouveau decorative arts", "Western European"),
    ("Art Deco", {"q": "art deco", "page_size": 100}, "Art Deco design and objects", "Western European"),
    ("Victorian Decorative Arts", {"q": "victorian decorative arts", "page_size": 100}, "Victorian decorative arts", "Western European"),
    ("Japanese Woodblock Prints", {"q": "japanese woodblock ukiyo-e", "page_size": 100}, "Japanese ukiyo-e woodblock prints", "East Asian"),
    ("South Asian Textiles", {"q": "south asian embroidery silk", "page_size": 100}, "South Asian embroidered textiles", "South Asian"),
    ("Byzantine and Medieval", {"q": "byzantine medieval", "page_size": 100}, "Byzantine and medieval artifacts", "Ancient"),
    ("Renaissance Art Objects", {"q": "renaissance", "page_size": 100}, "Renaissance decorative arts", "Western European"),
    ("Neo-Classical Design", {"q": "neo-classical", "page_size": 100}, "Neo-classical furniture and objects", "Western European"),
    ("Pre-Raphaelite and Victorian Painting", {"q": "pre-raphaelite painting victorian", "page_size": 100}, "Pre-Raphaelite and Victorian painting", "Western European"),
    ("Southeast Asian Art", {"q": "southeast asian art", "page_size": 100}, "Art from Southeast Asia", "South Asian"),
    ("Arms and Armor V&A", {"q": "armor sword weapon", "page_size": 100}, "Historical arms and armor", "Western European"),
    ("Musical Instruments", {"q": "musical instrument", "page_size": 100}, "Historical musical instruments", "Other"),
    ("Graphic Design Posters", {"q": "poster graphic design", "page_size": 100}, "Graphic design and poster art", "Western European"),
    ("Contemporary Crafts", {"q": "contemporary craft", "page_size": 100}, "Contemporary craft and design", "Other"),
    ("Middle Eastern Art", {"q": "middle east arab art", "page_size": 100}, "Art from the Middle East", "Islamic"),
]


def collect_vam_extended(max_per_theme: int = 150) -> tuple[list, list]:
    exhibitions, objects = [], []
    obj_seen = set()

    for theme, params, description, culture_hint in tqdm(VAM_THEMES, desc="V&A themes"):
        theme_objs = []
        page = 1
        while len(theme_objs) < max_per_theme:
            p = {**params, "page": page}
            d = http_get(VAM_BASE, p)
            if not d or not d.get("records"):
                break
            for rec in d["records"]:
                oid = rec.get("systemNumber", "")
                if not oid or oid in obj_seen:
                    continue
                obj_seen.add(oid)
                # extract fields
                title = ""
                titles = rec.get("titles", [])
                if isinstance(titles, list) and titles:
                    title = titles[0].get("title", "") if isinstance(titles[0], dict) else str(titles[0])
                if not title:
                    title = rec.get("_primaryTitle", "")

                date = rec.get("_primaryDate", "") or rec.get("productionDates", [{}])[0].get("date", {}).get("text", "") if isinstance(rec.get("productionDates"), list) else ""
                culture = rec.get("_primaryPlace", "") or culture_hint
                medium = rec.get("materialsAndTechniques", "") or ""
                if isinstance(medium, list):
                    medium = "; ".join(str(m) for m in medium)
                dept = rec.get("department", "") or ""
                desc = rec.get("briefDescription", "") or ""
                img = rec.get("_primaryImageId", "")
                if img:
                    img = f"https://framemark.vam.ac.uk/collections/{img}/full/735,/0/default.jpg"
                classification = rec.get("objectType", "") or ""

                obj = make_obj(
                    "vam_ext", oid,
                    title=title,
                    date=str(date) if date else "",
                    culture=culture,
                    medium=medium,
                    department=dept,
                    description=desc,
                    image_url=img,
                    classification=classification,
                )
                theme_objs.append(obj)
            page += 1
            if page > 4:
                break
            time.sleep(0.15)

        if len(theme_objs) < 5:
            continue
        theme_objs = theme_objs[:max_per_theme]
        objects.extend(theme_objs)
        exh_id = theme.lower().replace(" ", "_").replace(":", "").replace(",", "").replace("&", "and").replace("/", "_")
        exh = make_exh(
            "vam_ext", exh_id,
            title=f"V&A: {theme}",
            theme=theme,
            description=description,
            object_ids=[],
        )
        exh["object_ids"] = [o["id"] for o in theme_objs]
        exhibitions.append(exh)

    log.info(f"V&A extended: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ── Europeana ─────────────────────────────────────────────────────────────────

EUROPEANA_BASE = "https://api.europeana.eu/record/v2/search.json"
EUROPEANA_KEY = "api2demo"

EUROPEANA_THEMES = [
    ("Dutch Golden Age", "Dutch Golden Age painting and prints", "who:Rembrandt OR who:Vermeer OR who:Hals", "Western European"),
    ("French Impressionism", "French Impressionist art", "who:Monet OR who:Renoir OR who:Degas OR what:impressionism", "Western European"),
    ("Renaissance Art", "Italian and European Renaissance masterworks", "what:renaissance AND TYPE:IMAGE", "Western European"),
    ("Byzantine Icons", "Byzantine religious icons and mosaics", "what:byzantine AND TYPE:IMAGE", "Ancient"),
    ("Ancient Egypt", "Ancient Egyptian artifacts and art", "what:egypt AND DATA_PROVIDER:museum", "Ancient"),
    ("Nordic Folk Art", "Traditional Nordic folk art and crafts", "what:folk+art AND where:scandinavia", "Western European"),
    ("Ottoman Empire Art", "Art and artifacts from the Ottoman Empire", "what:ottoman AND TYPE:IMAGE", "Islamic"),
    ("Mughal India", "Mughal-period paintings and decorative arts", "what:mughal AND TYPE:IMAGE", "South Asian"),
    ("African Heritage", "African cultural heritage objects", "what:africa AND TYPE:IMAGE AND what:mask OR what:sculpture", "African"),
    ("Japanese Art Europeana", "Japanese art in European collections", "what:japan AND TYPE:IMAGE", "East Asian"),
    ("Medieval Illuminated Manuscripts", "Illuminated manuscripts from medieval Europe", "what:illuminated+manuscript", "Western European"),
    ("Baroque Architecture", "Baroque church and palace architecture", "what:baroque AND TYPE:IMAGE", "Western European"),
    ("Greek Antiquities", "Ancient Greek vases and sculpture", "what:ancient+greece AND TYPE:IMAGE", "Ancient"),
    ("Chinese Art Europeana", "Chinese art in European collections", "what:china AND TYPE:IMAGE", "East Asian"),
    ("Spanish Masters", "Spanish painting from El Greco to Goya", "where:spain AND what:painting AND TYPE:IMAGE", "Western European"),
    ("German Expressionism Euro", "Expressionist art from Germany", "what:expressionism AND where:germany AND TYPE:IMAGE", "Western European"),
    ("Russian Art", "Russian imperial and avant-garde art", "what:art AND where:russia AND TYPE:IMAGE", "Western European"),
    ("Victorian Photography", "Victorian-era photography from Britain", "what:photography AND where:britain", "Western European"),
    ("Pre-Columbian Americas", "Pre-Columbian artifacts from the Americas", "what:pre-columbian", "American"),
    ("Islamic Manuscripts", "Islamic calligraphy and manuscripts", "what:islamic+manuscript", "Islamic"),
    ("Flemish Paintings", "Flemish painting from Van Eyck to Rubens", "where:flanders AND what:painting", "Western European"),
    ("Ancient Rome", "Roman sculpture and decorative arts", "what:roman AND TYPE:IMAGE AND what:sculpture", "Ancient"),
    ("20th Century Modernism", "Modernist art of the 20th century", "what:modernism AND TYPE:IMAGE", "Western European"),
    ("Celtic Art", "Celtic metalwork and decorative arts", "what:celtic AND TYPE:IMAGE", "Western European"),
    ("Ancient China", "Ancient Chinese bronzes and jade", "what:china AND what:ancient AND TYPE:IMAGE", "East Asian"),
    ("Italian Baroque", "Italian Baroque painting and sculpture", "where:italy AND what:baroque AND TYPE:IMAGE", "Western European"),
    ("Nordic Mythology Art", "Art depicting Norse mythology", "what:viking OR what:norse AND TYPE:IMAGE", "Western European"),
    ("Buddhist Art", "Buddhist art across Asia", "what:buddhism AND TYPE:IMAGE", "East Asian"),
    ("Hindu Art", "Hindu sculpture and religious art", "what:hindu AND TYPE:IMAGE", "South Asian"),
    ("Persia and Iran", "Persian miniatures and metalwork", "where:persia OR where:iran AND TYPE:IMAGE", "Islamic"),
    ("Ottoman Textiles", "Ottoman carpets, textiles, and embroidery", "what:ottoman AND what:textile", "Islamic"),
    ("Ancient Greece", "Sculpture and vases from ancient Greece", "what:greece AND what:ancient", "Ancient"),
    ("Medieval Castles", "Medieval fortress and castle architecture", "what:medieval AND what:castle", "Western European"),
    ("Art Nouveau Europe", "Art Nouveau across Europe", "what:art+nouveau AND TYPE:IMAGE", "Western European"),
    ("Colonial Americas Art", "Art from colonial-era Americas", "what:colonial AND where:america", "American"),
    ("Byzantine Architecture", "Byzantine churches and mosaics", "what:byzantine AND what:church", "Ancient"),
    ("Impressionist Landscapes", "Landscape painting by Impressionists", "what:impressionism AND what:landscape", "Western European"),
    ("Dutch Flower Painting", "Dutch still-life flower painting", "what:still+life AND where:netherlands", "Western European"),
    ("Korean Traditional Arts", "Traditional Korean arts and crafts", "where:korea AND TYPE:IMAGE", "East Asian"),
    ("Southeast Asian Heritage", "Cultural heritage from Southeast Asia", "what:southeast+asia AND TYPE:IMAGE", "South Asian"),
    ("Ancient Mesopotamia", "Art and artifacts from ancient Mesopotamia", "what:mesopotamia", "Ancient"),
    ("Scottish Heritage", "Scottish art and cultural heritage", "where:scotland AND TYPE:IMAGE", "Western European"),
    ("Polish Folk Art", "Traditional Polish folk art and crafts", "what:folk+art AND where:poland", "Western European"),
    ("Habsburg Empire Art", "Art from the Habsburg imperial collection", "what:habsburg AND TYPE:IMAGE", "Western European"),
    ("Aztec and Maya Art", "Pre-Columbian art from Mesoamerica", "what:aztec OR what:maya AND TYPE:IMAGE", "American"),
    ("Jewish Heritage", "Jewish religious and cultural artifacts", "what:jewish AND TYPE:IMAGE", "Western European"),
    ("Natural History Illustrations", "Scientific natural history illustrations", "what:natural+history AND TYPE:IMAGE", "Western European"),
    ("Early Photography Europe", "Early photographic works from Europe", "what:daguerreotype OR what:calotype AND TYPE:IMAGE", "Western European"),
    ("Venetian Art", "Painting and art from Venice", "where:venice AND TYPE:IMAGE", "Western European"),
    ("Persian Miniature Painting", "Persian illuminated manuscripts and miniatures", "what:persian+miniature AND TYPE:IMAGE", "Islamic"),
    ("African Masks and Sculpture", "Traditional African ceremonial objects", "what:africa AND what:mask", "African"),
    ("Tibetan Buddhist Art", "Tibetan thangka painting and Buddhist art", "what:tibet AND TYPE:IMAGE", "East Asian"),
    ("English Landscape Painting", "English landscape painting from Constable to Turner", "where:england AND what:landscape", "Western European"),
    ("Mayan Codices", "Mayan manuscripts and stone inscriptions", "what:mayan AND TYPE:IMAGE", "American"),
    ("Mughal Miniature Painting", "Mughal court paintings", "what:mughal AND what:miniature", "South Asian"),
    ("Chinese Calligraphy", "Chinese calligraphy and ink painting", "what:chinese+calligraphy", "East Asian"),
    ("Japanese Lacquerware", "Japanese lacquer art and inro", "what:japanese AND what:lacquer", "East Asian"),
    ("African Beadwork", "African beadwork and jewelry", "what:african AND what:beadwork", "African"),
    ("Ottoman Architecture", "Ottoman mosque and palace architecture", "what:ottoman AND what:architecture", "Islamic"),
    ("Greek Mythology in Art", "Paintings and sculpture depicting Greek myths", "what:mythology AND TYPE:IMAGE", "Ancient"),
]


def collect_europeana_extended(max_per_theme: int = 100) -> tuple[list, list]:
    exhibitions, objects = [], []
    obj_seen = set()

    for theme, description, query, culture_hint in tqdm(EUROPEANA_THEMES, desc="Europeana themes"):
        theme_objs = []
        for start in range(1, max_per_theme + 1, 100):
            params = {
                "wskey": EUROPEANA_KEY,
                "query": query,
                "rows": min(100, max_per_theme - len(theme_objs)),
                "start": start,
                "profile": "rich",
            }
            d = http_get(EUROPEANA_BASE, params)
            if not d or not d.get("items"):
                break
            for item in d["items"]:
                oid = item.get("id", "").replace("/", "_").lstrip("_")
                if not oid or oid in obj_seen:
                    continue
                obj_seen.add(oid)
                title = item.get("title", ["Untitled"])[0] if isinstance(item.get("title"), list) else item.get("title", "")
                date = ""
                dates = item.get("year", []) or item.get("edmTimespanLabel", [])
                if isinstance(dates, list) and dates:
                    date = dates[0] if isinstance(dates[0], str) else dates[0].get("def", "")
                culture = item.get("edmCountry", [""])[0] if isinstance(item.get("edmCountry"), list) else ""
                if not culture:
                    culture = culture_hint
                medium = item.get("dcMedium", [""])[0] if isinstance(item.get("dcMedium"), list) else ""
                dept = item.get("dataProvider", [""])[0] if isinstance(item.get("dataProvider"), list) else ""
                desc_list = item.get("dcDescription", []) or item.get("dcDescriptionLangAware", {})
                desc = desc_list[0] if isinstance(desc_list, list) and desc_list else ""
                img = item.get("edmPreview", [""])[0] if isinstance(item.get("edmPreview"), list) else ""
                classification = item.get("type", "") or ""
                obj = make_obj(
                    "europeana_ext", oid,
                    title=title,
                    date=date,
                    culture=culture or culture_hint,
                    medium=medium,
                    department=dept,
                    description=desc,
                    image_url=img,
                    classification=classification,
                )
                theme_objs.append(obj)
            if len(theme_objs) >= max_per_theme:
                break
            time.sleep(0.1)

        if len(theme_objs) < 5:
            continue
        theme_objs = theme_objs[:max_per_theme]
        objects.extend(theme_objs)
        exh_id = theme.lower().replace(" ", "_").replace(":", "").replace(",", "").replace("&", "and").replace("/", "_")
        exh = make_exh(
            "europeana_ext", exh_id,
            title=f"Europeana: {theme}",
            theme=theme,
            description=description,
            object_ids=[],
        )
        exh["object_ids"] = [o["id"] for o in theme_objs]
        exhibitions.append(exh)

    log.info(f"Europeana extended: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ── Met Museum ────────────────────────────────────────────────────────────────

MET_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

# Met departments: 1=American, 3=Ancient Near East, 4=Arms/Armor, 5=Asian, 6=Egypt,
# 7=European Paintings, 8=Greek/Roman, 9=Islamic, 10=Robert Lehman, 11=Medieval,
# 12=Oceania, 13=Photography, 14=Prints/Drawings, 15=Musical Inst,
# 16=Costume, 17=Modern/Contemporary, 18=European Sculpture/Decorative
MET_THEMES = [
    ("American Art", 1, "America", "American art in the permanent collection"),
    ("Ancient Near East", 3, "Ancient", "Ancient Near Eastern art and artifacts"),
    ("Arms and Armor Met", 4, "Western European", "Arms and armor from around the world"),
    ("Asian Art Met", 5, "East Asian", "Asian art across many cultures and media"),
    ("Egyptian Art Met", 6, "Ancient", "Ancient Egyptian art and artifacts"),
    ("European Paintings Met", 7, "Western European", "European painting from medieval to modern"),
    ("Greek and Roman Art Met", 8, "Ancient", "Art of ancient Greece and Rome"),
    ("Islamic Art Met", 9, "Islamic", "Islamic art from the 7th century onward"),
    ("Medieval Art Met", 11, "Ancient", "Medieval European art and treasury"),
    ("Oceanic Art Met", 12, "Other", "Art from Oceania, Africa, and the Americas"),
    ("Photographs Met", 13, "Western European", "Photography in the collection"),
    ("Prints and Drawings Met", 14, "Western European", "Prints and drawings from Europe"),
    ("Musical Instruments Met", 15, "Other", "Historical musical instruments"),
    ("Costume Institute", 16, "Western European", "Fashion and textiles through the ages"),
    ("Modern Art Met", 17, "Western European", "Modern and contemporary art"),
    ("European Decorative Arts", 18, "Western European", "European sculpture and decorative arts"),
]


def collect_met_extended(max_per_theme: int = 150) -> tuple[list, list]:
    exhibitions, objects = [], []
    obj_seen = set()

    for theme, dept_id, culture_hint, description in tqdm(MET_THEMES, desc="Met themes"):
        # Search for object IDs in department
        d = http_get(f"{MET_BASE}/search", {
            "departmentId": dept_id,
            "q": "*",
            "hasImages": "true",
        })
        if not d or not d.get("objectIDs"):
            continue
        all_ids = d["objectIDs"]
        # Sample up to max_per_theme
        import random
        rng = random.Random(42 + dept_id)
        sample_ids = rng.sample(all_ids, min(max_per_theme * 2, len(all_ids)))

        theme_objs = []
        for oid in sample_ids:
            if len(theme_objs) >= max_per_theme:
                break
            str_oid = str(oid)
            if str_oid in obj_seen:
                continue
            obj_data = http_get(f"{MET_BASE}/objects/{oid}")
            if not obj_data:
                continue
            title = obj_data.get("title", "")
            if not title:
                continue
            obj_seen.add(str_oid)
            culture = obj_data.get("culture", "") or obj_data.get("country", "") or culture_hint
            obj = make_obj(
                "met_ext", str_oid,
                title=title,
                date=obj_data.get("objectDate", ""),
                culture=culture,
                medium=obj_data.get("medium", ""),
                department=obj_data.get("department", ""),
                description=obj_data.get("objectName", ""),
                image_url=obj_data.get("primaryImage", ""),
                classification=obj_data.get("classification", ""),
                period=obj_data.get("period", ""),
            )
            theme_objs.append(obj)
            time.sleep(0.05)

        if len(theme_objs) < 5:
            continue
        objects.extend(theme_objs)
        exh_id = theme.lower().replace(" ", "_").replace(":", "").replace(",", "").replace("&", "and").replace("/", "_")
        exh = make_exh(
            "met_ext", exh_id,
            title=f"Met: {theme}",
            theme=theme,
            description=description,
            object_ids=[],
        )
        exh["object_ids"] = [o["id"] for o in theme_objs]
        exhibitions.append(exh)

    log.info(f"Met extended: {len(exhibitions)} exhibitions, {len(objects)} objects")
    return exhibitions, objects


# ── Merge ────────────────────────────────────────────────────────────────────

def load_v2() -> tuple[list, list]:
    exh_path = DATA / "exhibitions_v2.jsonl"
    obj_path = DATA / "objects_v2.jsonl"
    exhibitions = []
    objects = []
    if exh_path.exists():
        with open(exh_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    exhibitions.append(json.loads(line))
    if obj_path.exists():
        with open(obj_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    objects.append(json.loads(line))
    return exhibitions, objects


def write_jsonl(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records)} records → {path}")


def print_stats(exhibitions: list, objects: list) -> None:
    src_exh = Counter(e.get("source", "?") for e in exhibitions)
    src_obj = Counter(o.get("source", "?") for o in objects)
    tes_elig = sum(1 for e in exhibitions if len(e.get("object_ids", [])) >= 5)

    lines = [
        "=" * 60,
        "ExhibitionBench Dataset Statistics (v3)",
        "=" * 60,
        f"Total exhibitions : {len(exhibitions)}",
    ]
    for k, v in sorted(src_exh.items(), key=lambda x: -x[1]):
        lines.append(f"  {k:<25}: {v}")
    lines.append(f"\nTotal objects     : {len(objects)}")
    for k, v in sorted(src_obj.items(), key=lambda x: -x[1]):
        lines.append(f"  {k:<25}: {v}")

    cultures = Counter(o.get("culture", "") for o in objects)
    top_cultures = cultures.most_common(15)
    lines.append("\nCultural coverage (top 15):")
    for c, cnt in top_cultures:
        lines.append(f"  {c or 'Unknown':<30}: {cnt:5d} ({cnt/len(objects)*100:.1f}%)")

    lines += [
        f"\nTES-eligible exhibitions (>=5 objects): {tes_elig}",
        "=" * 60,
    ]
    print("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

COLLECTORS = {
    "aic":       collect_aic_extended,
    "cleveland": collect_cleveland_extended,
    "vam":       collect_vam_extended,
    "europeana": collect_europeana_extended,
    "met":       collect_met_extended,
}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="*", default=["all"],
                        help="Sources to collect: aic cleveland vam europeana met all")
    parser.add_argument("--no-merge", action="store_true", help="Don't merge with v2")
    parser.add_argument("--max-per-theme", type=int, default=120,
                        help="Max objects per virtual exhibition (default 120)")
    args = parser.parse_args()

    sources = args.source
    if "all" in sources:
        sources = list(COLLECTORS.keys())

    DATA.mkdir(parents=True, exist_ok=True)

    all_exhs, all_objs = [], []
    for src in sources:
        if src not in COLLECTORS:
            log.warning(f"Unknown source: {src}")
            continue
        log.info(f"=== Collecting {src} ===")
        exhs, objs = COLLECTORS[src](max_per_theme=args.max_per_theme)
        all_exhs.extend(exhs)
        all_objs.extend(objs)

    if not args.no_merge:
        v2_exhs, v2_objs = load_v2()
        log.info(f"Loaded v2: {len(v2_exhs)} exhibitions, {len(v2_objs)} objects")
        # Deduplicate by id
        existing_exh_ids = {e["id"] for e in v2_exhs}
        existing_obj_ids = {o["id"] for o in v2_objs}
        new_exhs = [e for e in all_exhs if e["id"] not in existing_exh_ids]
        new_objs = [o for o in all_objs if o["id"] not in existing_obj_ids]
        log.info(f"New (deduplicated): {len(new_exhs)} exhibitions, {len(new_objs)} objects")
        all_exhs = v2_exhs + new_exhs
        all_objs = v2_objs + new_objs

    out_exh = DATA / "exhibitions_v3.jsonl"
    out_obj = DATA / "objects_v3.jsonl"
    write_jsonl(out_exh, all_exhs)
    write_jsonl(out_obj, all_objs)

    print_stats(all_exhs, all_objs)
    print(f"\nOutputs:\n  Exhibitions : {out_exh}\n  Objects     : {out_obj}")


if __name__ == "__main__":
    main()

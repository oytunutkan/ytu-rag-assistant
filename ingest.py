# ingest.py
# YTÜ mevzuat dosyalarını ChromaDB'ye aktarır.
# Admin panelinden çağrılabilir güvenli ingest + dinamik kategori desteği.

import os
import glob
import shutil
import re
import json
import copy
import gc
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from category_rules import CATEGORY_RULES


# =========================================================
# TEMEL PATH / SABİTLER
# =========================================================

DATA_DIR = "data"
ADMIN_UPLOAD_DIR = os.path.join(DATA_DIR, "admin_uploads")

CHROMA_PATH = "chroma_db"
CHROMA_NEW_PATH = "chroma_db_new"

COLLECTION_NAME = "ytu_mevzuat"
EMBEDDING_MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"

DYNAMIC_RULES_PATH = "dynamic_category_rules.json"
DOCUMENT_REGISTRY_PATH = "document_registry.json"

MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 2000
MIN_CHUNK_OVERLAP = 0
MAX_CHUNK_OVERLAP = 500


# =========================================================
# STATİK CHUNK AYARLARI
# =========================================================

CATEGORY_CHUNK_CONFIG = {
    "bitirme_calismasi":     {"chunk_size": 500,  "chunk_overlap": 75},
    "sinav_itiraz":          {"chunk_size": 500,  "chunk_overlap": 50},
    "yuzde_on":              {"chunk_size": 500,  "chunk_overlap": 75},
    "akademik_danismanlik":  {"chunk_size": 500,  "chunk_overlap": 75},

    "mazeret_sinavi":        {"chunk_size": 450,  "chunk_overlap": 100},
    "mezuniyet_sinavi":      {"chunk_size": 650,  "chunk_overlap": 100},
    "sinav_yonergesi":       {"chunk_size": 750,  "chunk_overlap": 100},

    "staj":                  {"chunk_size": 500,  "chunk_overlap": 75},
    "ders_kayit":            {"chunk_size": 650,  "chunk_overlap": 100},
    "yaz_okulu":             {"chunk_size": 750,  "chunk_overlap": 100},
    "cap":                   {"chunk_size": 900,  "chunk_overlap": 150},
    "yandal":                {"chunk_size": 500,  "chunk_overlap": 75},
    "azami_sure":            {"chunk_size": 650,  "chunk_overlap": 120},
    "esdegerlik_intibak":    {"chunk_size": 750,  "chunk_overlap": 120},

    "ozel_ogrenci_ytuye_gelen":  {"chunk_size": 750, "chunk_overlap": 120},
    "ozel_ogrenci_ytuden_giden": {"chunk_size": 750, "chunk_overlap": 120},

    "lisansustu_genel_basvuru":                   {"chunk_size": 800, "chunk_overlap": 150},
    "lisansustu_tezli_yuksek_lisans":             {"chunk_size": 900, "chunk_overlap": 150},
    "lisansustu_tezsiz_yuksek_lisans":            {"chunk_size": 850, "chunk_overlap": 150},
    "lisansustu_doktora":                         {"chunk_size": 900, "chunk_overlap": 150},
    "lisansustu_sanatta_yeterlik":                {"chunk_size": 900, "chunk_overlap": 150},
    "lisansustu_bilimsel_hazirlik_kayit_ogretim": {"chunk_size": 850, "chunk_overlap": 150},
    "lisansustu_kontenjan_danismanlik_ek_sure":   {"chunk_size": 750, "chunk_overlap": 120},

    "diploma_bilgileri":                 {"chunk_size": 600, "chunk_overlap": 100},
    "diploma_mezuniyet_tarihleri":       {"chunk_size": 400, "chunk_overlap": 75},
    "diploma_eki_yandal_belgeler":       {"chunk_size": 550, "chunk_overlap": 100},
    "diploma_teslim_kayip_ikinci_nusha": {"chunk_size": 650, "chunk_overlap": 120},

    "kurum_ici_yatay_gecis":            {"chunk_size": 1000, "chunk_overlap": 150},
    "kurumlar_arasi_yatay_gecis":       {"chunk_size": 1000, "chunk_overlap": 150},
    "merkezi_yerlestirme_yatay_gecis":  {"chunk_size": 900,  "chunk_overlap": 150},
    "yurt_disi_yatay_gecis":            {"chunk_size": 1000, "chunk_overlap": 150},
    "dikey_gecis":                      {"chunk_size": 850,  "chunk_overlap": 120},
    "yatay_gecis_genel":                {"chunk_size": 900,  "chunk_overlap": 150},
    "isletmede_mesleki_egitim":         {"chunk_size": 1000, "chunk_overlap": 150},

    "isletmede_mesleki_egitim_tanimlar": {
        "chunk_size": 650,
        "chunk_overlap": 100,
    },
    "isletmede_mesleki_egitim_komisyonlar_gorevler": {
        "chunk_size": 750,
        "chunk_overlap": 120,
    },
    "isletmede_mesleki_egitim_isletme_egitici_sorumlu_ogrenci": {
        "chunk_size": 750,
        "chunk_overlap": 120,
    },
    "isletmede_mesleki_egitim_basvuru_degerlendirme_itiraz": {
        "chunk_size": 800,
        "chunk_overlap": 120,
    },
    "isletmede_mesleki_egitim_diger_hukumler": {
        "chunk_size": 750,
        "chunk_overlap": 120,
    },

    "onceki_ogrenme": {"chunk_size": 500, "chunk_overlap": 100},

    "genel": {"chunk_size": 600, "chunk_overlap": 75},
}


V12_CHUNK_OVERRIDES = {
    "mazeret_sinavi":        {"chunk_size": 650,  "chunk_overlap": 150},
    "sinav_yonergesi":       {"chunk_size": 650,  "chunk_overlap": 130},
    "staj":                  {"chunk_size": 720,  "chunk_overlap": 150},
    "ders_kayit":            {"chunk_size": 800,  "chunk_overlap": 160},
    "onceki_ogrenme":        {"chunk_size": 700,  "chunk_overlap": 150},
    "ozel_ogrenci_ytuye_gelen":  {"chunk_size": 850, "chunk_overlap": 170},
    "ozel_ogrenci_ytuden_giden": {"chunk_size": 850, "chunk_overlap": 170},
    "esdegerlik_intibak":    {"chunk_size": 850,  "chunk_overlap": 160},
    "diploma_bilgileri":                 {"chunk_size": 720, "chunk_overlap": 140},
    "diploma_mezuniyet_tarihleri":       {"chunk_size": 520, "chunk_overlap": 100},
    "diploma_eki_yandal_belgeler":       {"chunk_size": 700, "chunk_overlap": 140},
    "diploma_teslim_kayip_ikinci_nusha": {"chunk_size": 760, "chunk_overlap": 150},
    "kurum_ici_yatay_gecis":            {"chunk_size": 1050, "chunk_overlap": 180},
    "kurumlar_arasi_yatay_gecis":       {"chunk_size": 1050, "chunk_overlap": 180},
    "merkezi_yerlestirme_yatay_gecis":  {"chunk_size": 980,  "chunk_overlap": 170},
    "yurt_disi_yatay_gecis":            {"chunk_size": 1050, "chunk_overlap": 180},
    "lisansustu_tezli_yuksek_lisans":             {"chunk_size": 950, "chunk_overlap": 180},
    "lisansustu_doktora":                         {"chunk_size": 980, "chunk_overlap": 180},
    "lisansustu_bilimsel_hazirlik_kayit_ogretim": {"chunk_size": 900, "chunk_overlap": 170},
    "lisansustu_kontenjan_danismanlik_ek_sure":   {"chunk_size": 850, "chunk_overlap": 160},
}
CATEGORY_CHUNK_CONFIG.update(V12_CHUNK_OVERRIDES)


FINAL_CHUNK_OVERRIDES = {
    "staj": {"chunk_size": 620, "chunk_overlap": 130},
    "yandal": {"chunk_size": 620, "chunk_overlap": 130},
    "cap": {"chunk_size": 800, "chunk_overlap": 160},
    "onceki_ogrenme": {"chunk_size": 680, "chunk_overlap": 150},
    "ozel_ogrenci_ytuye_gelen": {"chunk_size": 760, "chunk_overlap": 150},
    "ozel_ogrenci_ytuden_giden": {"chunk_size": 760, "chunk_overlap": 150},
    "sinav_yonergesi": {"chunk_size": 680, "chunk_overlap": 140},
    "isletmede_mesleki_egitim_tanimlar": {"chunk_size": 650, "chunk_overlap": 130},
    "isletmede_mesleki_egitim_komisyonlar_gorevler": {"chunk_size": 650, "chunk_overlap": 130},
    "isletmede_mesleki_egitim_isletme_egitici_sorumlu_ogrenci": {"chunk_size": 650, "chunk_overlap": 130},
    "isletmede_mesleki_egitim_basvuru_degerlendirme_itiraz": {"chunk_size": 720, "chunk_overlap": 150},
    "ders_kayit": {"chunk_size": 700, "chunk_overlap": 150},
    "yaz_okulu": {"chunk_size": 700, "chunk_overlap": 140},
}
CATEGORY_CHUNK_CONFIG.update(FINAL_CHUNK_OVERRIDES)


# =========================================================
# GENEL YARDIMCI FONKSİYONLAR
# =========================================================

def normalize_text(text: str) -> str:
    return (
        str(text).lower()
        .replace("ı", "i").replace("İ", "i")
        .replace("ğ", "g").replace("ü", "u")
        .replace("ş", "s").replace("ö", "o")
        .replace("ç", "c")
        .replace("’", "'").replace("‘", "'").replace("`", "'")
        .replace("\\", "/")
    )


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return copy.deepcopy(default)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON dosyası okunamadı: {path} | Hata: {exc}") from exc


def save_json_file(path: str, data: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def unique_list(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        key = normalize_text(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def normalize_path_key(path: str) -> str:
    return normalize_text(Path(path).as_posix())


def validate_chunk_config(
    chunk_size: Any,
    chunk_overlap: Any,
    label: str = "chunk config",
) -> Dict[str, int]:
    size = safe_int(chunk_size)
    overlap = safe_int(chunk_overlap)

    if size is None:
        raise ValueError(f"{label}: chunk_size sayısal olmalıdır.")
    if overlap is None:
        raise ValueError(f"{label}: chunk_overlap sayısal olmalıdır.")

    if not (MIN_CHUNK_SIZE <= size <= MAX_CHUNK_SIZE):
        raise ValueError(
            f"{label}: chunk_size {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE} arasında olmalıdır. "
            f"Gelen değer: {size}"
        )

    if not (MIN_CHUNK_OVERLAP <= overlap <= MAX_CHUNK_OVERLAP):
        raise ValueError(
            f"{label}: chunk_overlap {MIN_CHUNK_OVERLAP}-{MAX_CHUNK_OVERLAP} arasında olmalıdır. "
            f"Gelen değer: {overlap}"
        )

    if overlap >= size:
        raise ValueError(
            f"{label}: chunk_overlap, chunk_size değerinden küçük olmalıdır. "
            f"chunk_size={size}, chunk_overlap={overlap}"
        )

    return {
        "chunk_size": size,
        "chunk_overlap": overlap,
    }


# =========================================================
# DYNAMIC CATEGORY RULES
# =========================================================

def get_dynamic_categories(dynamic_rules: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Beklenen format:
    {
      "categories": {
        "kategori_key": {
          "display_name": "...",
          "files": [...],
          "keywords": [...],
          "weighted_keywords": [
            {"keyword": "...", "level": "strong"}
          ],
          "chunk_size": 800,
          "chunk_overlap": 150
        }
      }
    }

    Esneklik için dosya direkt kategori dict'i olarak gelirse onu da kabul eder.
    """
    if not isinstance(dynamic_rules, dict):
        return {}

    categories = dynamic_rules.get("categories")

    if isinstance(categories, dict):
        return categories

    # Eski/alternatif format toleransı:
    # {"staj": {"files": [], "keywords": []}}
    possible_categories = {}
    for key, value in dynamic_rules.items():
        if isinstance(value, dict) and ("files" in value or "keywords" in value):
            possible_categories[key] = value

    return possible_categories


def load_dynamic_rules(path: str = DYNAMIC_RULES_PATH) -> Dict[str, Any]:
    default_data = {
        "categories": {}
    }
    return load_json_file(path, default_data)


def merge_category_rules(
    static_rules: Dict[str, Dict[str, Any]],
    dynamic_rules: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Statik category_rules.py ile admin panelinden gelen dynamic_category_rules.json
    kurallarını birleştirir.

    Ingest tarafında önemli alanlar:
    - files
    - keywords

    weighted_keywords app/rank tarafında kullanılacak; burada sadece taşınıyor.
    """
    merged = copy.deepcopy(static_rules)
    dynamic_categories = get_dynamic_categories(dynamic_rules)

    for category_key, dynamic_rule in dynamic_categories.items():
        if not isinstance(dynamic_rule, dict):
            continue

        category_key = str(category_key).strip()
        if not category_key:
            continue

        target = merged.setdefault(category_key, {
            "files": [],
            "keywords": [],
        })

        static_files = target.get("files", [])
        dynamic_files = dynamic_rule.get("files", [])
        target["files"] = unique_list(list(static_files) + list(dynamic_files))

        static_keywords = target.get("keywords", [])
        dynamic_keywords = dynamic_rule.get("keywords", [])
        target["keywords"] = unique_list(list(static_keywords) + list(dynamic_keywords))

        # App tarafında kullanılması için korunur.
        if "display_name" in dynamic_rule:
            target["display_name"] = dynamic_rule.get("display_name")

        if "weighted_keywords" in dynamic_rule:
            target["weighted_keywords"] = dynamic_rule.get("weighted_keywords", [])

    return merged


def get_dynamic_chunk_configs(dynamic_rules: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Dinamik kategoriye özel varsayılan chunk ayarı varsa okur.
    Bu ayar özellikle yeni kategorilerde kullanılır.
    Yeni/admin doküman için document_registry içindeki dosya bazlı ayar daha önceliklidir.
    """
    configs = {}
    dynamic_categories = get_dynamic_categories(dynamic_rules)

    for category_key, rule in dynamic_categories.items():
        if not isinstance(rule, dict):
            continue

        if "chunk_size" not in rule and "chunk_overlap" not in rule:
            continue

        config = validate_chunk_config(
            rule.get("chunk_size", CATEGORY_CHUNK_CONFIG["genel"]["chunk_size"]),
            rule.get("chunk_overlap", CATEGORY_CHUNK_CONFIG["genel"]["chunk_overlap"]),
            label=f"dynamic_category_rules.json > {category_key}",
        )
        configs[category_key] = config

    return configs


# =========================================================
# DOCUMENT REGISTRY
# =========================================================

def load_document_registry(path: str = DOCUMENT_REGISTRY_PATH) -> Dict[str, Any]:
    default_data = {
        "documents": [],
        "last_ingest": None,
    }
    registry = load_json_file(path, default_data)

    if not isinstance(registry, dict):
        registry = copy.deepcopy(default_data)

    if "documents" not in registry or not isinstance(registry["documents"], list):
        registry["documents"] = []

    if "last_ingest" not in registry:
        registry["last_ingest"] = None

    return registry


def save_document_registry(
    registry: Dict[str, Any],
    path: str = DOCUMENT_REGISTRY_PATH,
) -> None:
    save_json_file(path, registry)


def build_registry_lookup(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Registry kayıtlarını hem relative_path hem file_name hem normalized path ile bulabilmek için map oluşturur.
    """
    lookup = {}

    for entry in registry.get("documents", []):
        if not isinstance(entry, dict):
            continue

        file_name = entry.get("file_name") or entry.get("dosya_adi")
        relative_path = entry.get("relative_path") or entry.get("path")

        keys = []

        if file_name:
            keys.append(normalize_path_key(file_name))
            keys.append(normalize_text(os.path.basename(file_name)))

        if relative_path:
            keys.append(normalize_path_key(relative_path))
            keys.append(normalize_path_key(os.path.abspath(relative_path)))
            keys.append(normalize_text(os.path.basename(relative_path)))

        for key in keys:
            if key:
                lookup[key] = entry

    return lookup


def find_registry_entry(
    file_path: str,
    registry_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    candidates = [
        normalize_path_key(file_path),
        normalize_path_key(os.path.abspath(file_path)),
        normalize_text(os.path.basename(file_path)),
    ]

    try:
        candidates.append(normalize_path_key(os.path.relpath(file_path)))
    except Exception:
        pass

    for key in candidates:
        if key in registry_lookup:
            return registry_lookup[key]

    return None


def upsert_document_registry_entry(
    entry: Dict[str, Any],
    registry_path: str = DOCUMENT_REGISTRY_PATH,
) -> Dict[str, Any]:
    """
    App tarafı isterse bu fonksiyonu kullanarak yeni yüklenen dosyayı registry'ye ekleyebilir.
    """
    registry = load_document_registry(registry_path)
    documents = registry.setdefault("documents", [])

    file_name = entry.get("file_name") or os.path.basename(entry.get("relative_path", ""))
    relative_path = entry.get("relative_path")

    replaced = False

    for idx, old in enumerate(documents):
        old_file = old.get("file_name") or os.path.basename(old.get("relative_path", ""))
        old_path = old.get("relative_path")

        same_file = normalize_text(old_file) == normalize_text(file_name)
        same_path = relative_path and old_path and normalize_path_key(old_path) == normalize_path_key(relative_path)

        if same_file or same_path:
            documents[idx] = {**old, **entry}
            replaced = True
            break

    if not replaced:
        documents.append(entry)

    save_document_registry(registry, registry_path)
    return registry


# =========================================================
# KATEGORİ VE CHUNK KONFİG ÇÖZÜMLEME
# =========================================================

def detect_category_from_filename(
    file_path: str,
    active_category_rules: Dict[str, Dict[str, Any]],
    registry_entry: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Öncelik:
    1. document_registry.json içinde category_key varsa onu kullan.
    2. active category rules içindeki file key'lerle dosya adından kategori bul.
    3. Bulunamazsa genel.
    """
    if registry_entry:
        category_key = registry_entry.get("category_key") or registry_entry.get("kategori")
        if category_key:
            return str(category_key).strip()

    normalized_name = normalize_text(os.path.basename(file_path))

    candidates = []
    for category, rule in active_category_rules.items():
        for file_key in rule.get("files", []):
            candidates.append((category, file_key))

    candidates.sort(key=lambda item: len(normalize_text(item[1])), reverse=True)

    for category, file_key in candidates:
        if normalize_text(file_key) in normalized_name:
            return category

    return "genel"


def is_admin_document(
    file_path: str,
    registry_entry: Optional[Dict[str, Any]] = None,
) -> bool:
    normalized_path = normalize_path_key(file_path)

    if f"/{normalize_path_key(ADMIN_UPLOAD_DIR)}/" in f"/{normalized_path}/":
        return True

    if registry_entry:
        is_system_file = registry_entry.get("is_system_file")
        if is_system_file is False:
            return True

    return False


def resolve_chunk_config(
    category: str,
    file_path: str,
    registry_entry: Optional[Dict[str, Any]],
    dynamic_chunk_configs: Dict[str, Dict[str, int]],
) -> Dict[str, int]:
    """
    Chunk ayarı önceliği:

    1. Yeni/admin yüklenen dosyanın document_registry içindeki dosya bazlı chunk ayarı.
    2. Dinamik kategoriye yazılmış kategori bazlı chunk ayarı.
    3. Statik kategori chunk ayarı.
    4. Genel varsayılan.

    Mevcut sistem dosyalarının optimize edilmiş chunk ayarları korunur.
    """
    if registry_entry and is_admin_document(file_path, registry_entry):
        chunk_size = registry_entry.get("chunk_size")
        chunk_overlap = registry_entry.get("chunk_overlap")

        if chunk_size is not None or chunk_overlap is not None:
            fallback = CATEGORY_CHUNK_CONFIG.get(category, CATEGORY_CHUNK_CONFIG["genel"])
            return validate_chunk_config(
                chunk_size if chunk_size is not None else fallback["chunk_size"],
                chunk_overlap if chunk_overlap is not None else fallback["chunk_overlap"],
                label=f"document_registry.json > {os.path.basename(file_path)}",
            )

    if category in dynamic_chunk_configs:
        return dynamic_chunk_configs[category]

    return CATEGORY_CHUNK_CONFIG.get(category, CATEGORY_CHUNK_CONFIG["genel"])


# =========================================================
# METADATA VE RETRIEVAL ZENGİNLEŞTİRME
# =========================================================

def add_metadata(
    doc,
    dosya_yolu: str,
    kategori: str,
    config: Dict[str, int],
    registry_entry: Optional[Dict[str, Any]] = None,
) -> None:
    normalized_path = normalize_text(dosya_yolu)

    if "mikro_kart" in normalized_path or "micro_card" in normalized_path:
        doc.metadata["kaynak_tipi"] = "mikro_kart"
        doc.metadata["guvenilirlik_puani"] = 12
    else:
        doc.metadata["kaynak_tipi"] = "resmi_yonetmelik"
        doc.metadata["guvenilirlik_puani"] = 10

    doc.metadata["kategori"] = kategori
    doc.metadata["dosya_adi"] = os.path.basename(dosya_yolu)
    doc.metadata["chunk_size_config"] = config["chunk_size"]
    doc.metadata["chunk_overlap_config"] = config["chunk_overlap"]

    if registry_entry:
        doc.metadata["source_code"] = registry_entry.get("source_code", "")
        doc.metadata["document_type"] = registry_entry.get("document_type", "")
        doc.metadata["display_name"] = registry_entry.get("display_name", "")
        doc.metadata["uploaded_at"] = registry_entry.get("uploaded_at", "")
        doc.metadata["is_system_file"] = bool(registry_entry.get("is_system_file", False))
        doc.metadata["admin_uploaded"] = is_admin_document(dosya_yolu, registry_entry)
    else:
        doc.metadata["source_code"] = ""
        doc.metadata["document_type"] = ""
        doc.metadata["display_name"] = ""
        doc.metadata["uploaded_at"] = ""
        doc.metadata["is_system_file"] = False
        doc.metadata["admin_uploaded"] = False


def extract_madde_label(text: str) -> str:
    if not text:
        return ""

    match = re.search(r"(?im)^\s*(MADDE|Madde)\s+\d+[^\n]{0,120}", text)
    if match:
        return " ".join(match.group(0).split())

    return ""


def enrich_chunk_for_retrieval(chunk, kategori: str) -> None:
    dosya_adi = chunk.metadata.get("dosya_adi", "")
    madde_label = extract_madde_label(chunk.page_content)

    if madde_label:
        chunk.metadata["madde_etiketi"] = madde_label

    prefix_parts = [
        f"Kaynak dosya: {dosya_adi}",
        f"Kategori: {kategori}",
    ]

    if chunk.metadata.get("display_name"):
        prefix_parts.append(f"Doküman adı: {chunk.metadata.get('display_name')}")

    if chunk.metadata.get("source_code"):
        prefix_parts.append(f"Kaynak kodu: {chunk.metadata.get('source_code')}")

    if chunk.metadata.get("document_type"):
        prefix_parts.append(f"Doküman türü: {chunk.metadata.get('document_type')}")

    if chunk.metadata.get("kaynak_tipi") == "mikro_kart":
        prefix_parts.append("Mikro Kart: Evet")

    if madde_label:
        prefix_parts.append(f"Madde etiketi: {madde_label}")

    prefix = "\n".join(prefix_parts) + "\n---\n"

    if not chunk.page_content.startswith("Kaynak dosya:"):
        chunk.page_content = prefix + chunk.page_content


# =========================================================
# DOKÜMAN LİSTELEME
# =========================================================

def discover_document_files(data_dir: str = DATA_DIR) -> List[str]:
    docx_files = glob.glob(os.path.join(data_dir, "**", "*.docx"), recursive=True)
    txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    return sorted(docx_files + txt_files)


def list_known_documents(
    data_dir: str = DATA_DIR,
    dynamic_rules_path: str = DYNAMIC_RULES_PATH,
    document_registry_path: str = DOCUMENT_REGISTRY_PATH,
) -> List[Dict[str, Any]]:
    """
    App admin paneli mevcut dosyaları listelemek isterse bu fonksiyonu kullanabilir.
    Registry'deki kayıtları ve data klasöründeki gerçek dosyaları birleştirir.
    """
    dynamic_rules = load_dynamic_rules(dynamic_rules_path)
    active_rules = merge_category_rules(CATEGORY_RULES, dynamic_rules)

    registry = load_document_registry(document_registry_path)
    registry_lookup = build_registry_lookup(registry)

    rows = []

    for file_path in discover_document_files(data_dir):
        registry_entry = find_registry_entry(file_path, registry_lookup)
        kategori = detect_category_from_filename(file_path, active_rules, registry_entry)
        config = resolve_chunk_config(
            kategori,
            file_path,
            registry_entry,
            get_dynamic_chunk_configs(dynamic_rules),
        )

        rows.append({
            "file_name": os.path.basename(file_path),
            "relative_path": os.path.relpath(file_path),
            "category_key": kategori,
            "display_name": registry_entry.get("display_name", "") if registry_entry else "",
            "source_code": registry_entry.get("source_code", "") if registry_entry else "",
            "document_type": registry_entry.get("document_type", "") if registry_entry else "",
            "is_system_file": registry_entry.get("is_system_file", True) if registry_entry else True,
            "admin_uploaded": is_admin_document(file_path, registry_entry),
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "exists": True,
        })

    return rows


# =========================================================
# CHROMA DB GÜVENLİ YENİLEME
# =========================================================

def close_chroma_db(db: Any) -> None:
    """
    Özellikle Windows'ta Chroma dosya kilidi bırakmasın diye en iyi çabayla kapatma.
    """
    try:
        if hasattr(db, "persist"):
            db.persist()
    except Exception:
        pass

    try:
        client = getattr(db, "_client", None)
        system = getattr(client, "_system", None)
        if system is not None and hasattr(system, "stop"):
            system.stop()
    except Exception:
        pass

    try:
        del db
    except Exception:
        pass

    gc.collect()


def make_backup_path(active_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{active_dir}_backup_{timestamp}"
    backup_path = base
    counter = 1

    while os.path.exists(backup_path):
        backup_path = f"{base}_{counter}"
        counter += 1

    return backup_path


def safe_swap_chroma_dirs(
    new_dir: str,
    active_dir: str,
) -> Optional[str]:
    """
    Yeni DB başarılı üretildikten sonra:
    - eski active_dir yedeklenir
    - new_dir active_dir yapılır

    Hata olursa mümkün olduğunca eski DB geri alınır.
    """
    if not os.path.exists(new_dir):
        raise FileNotFoundError(f"Yeni ChromaDB klasörü bulunamadı: {new_dir}")

    backup_dir = None

    if os.path.exists(active_dir):
        backup_dir = make_backup_path(active_dir)
        print(f"Eski ChromaDB yedekleniyor: {active_dir} -> {backup_dir}", flush=True)
        shutil.move(active_dir, backup_dir)

    try:
        print(f"Yeni ChromaDB aktif ediliyor: {new_dir} -> {active_dir}", flush=True)
        shutil.move(new_dir, active_dir)
    except Exception as exc:
        print("Yeni DB aktif edilirken hata oluştu. Eski DB geri alınmaya çalışılıyor.", flush=True)

        if os.path.exists(active_dir):
            shutil.rmtree(active_dir, ignore_errors=True)

        if backup_dir and os.path.exists(backup_dir):
            shutil.move(backup_dir, active_dir)

        raise RuntimeError(f"ChromaDB swap işlemi başarısız oldu: {exc}") from exc

    return backup_dir


# =========================================================
# INGEST ANA FONKSİYONU
# =========================================================

def run_ingest(
    data_dir: str = DATA_DIR,
    persist_dir: str = CHROMA_PATH,
    temp_persist_dir: str = CHROMA_NEW_PATH,
    dynamic_rules_path: str = DYNAMIC_RULES_PATH,
    document_registry_path: str = DOCUMENT_REGISTRY_PATH,
    safe_swap: bool = True,
) -> Dict[str, Any]:
    """
    Admin panelinden veya terminalden çağrılabilecek ana ingest fonksiyonu.

    Not:
    - Mevcut optimize edilmiş kategori chunk ayarları korunur.
    - Admin panelinden yüklenen dosya için registry'deki chunk_size/chunk_overlap uygulanır.
    - Yeni ChromaDB önce temp klasöre kurulur, başarılı olursa aktif DB ile güvenli swap yapılır.
    """
    print(f"Çalışma klasörü: {os.getcwd()}", flush=True)
    print(f"Data klasörü: {data_dir}", flush=True)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} klasörü bulunamadı.")

    ensure_dir(ADMIN_UPLOAD_DIR)

    dynamic_rules = load_dynamic_rules(dynamic_rules_path)
    active_category_rules = merge_category_rules(CATEGORY_RULES, dynamic_rules)
    dynamic_chunk_configs = get_dynamic_chunk_configs(dynamic_rules)

    registry = load_document_registry(document_registry_path)
    registry_lookup = build_registry_lookup(registry)

    docx_dosyalari = glob.glob(os.path.join(data_dir, "**", "*.docx"), recursive=True)
    txt_dosyalari = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    dokuman_dosyalari = sorted(docx_dosyalari + txt_dosyalari)

    print(f"Bulunan Word dosyası sayısı: {len(docx_dosyalari)}", flush=True)
    print(f"Bulunan TXT mikro kart sayısı: {len(txt_dosyalari)}", flush=True)

    if not dokuman_dosyalari:
        raise FileNotFoundError(f"{data_dir} klasöründe .docx veya .txt dosyası yok.")

    dokumanlar_by_split_config: Dict[Tuple[str, int, int], List[Any]] = {}

    file_stats = []
    admin_file_count = 0

    for dosya_yolu in dokuman_dosyalari:
        if dosya_yolu.lower().endswith(".docx"):
            loader = Docx2txtLoader(dosya_yolu)
        elif dosya_yolu.lower().endswith(".txt"):
            loader = TextLoader(dosya_yolu, encoding="utf-8")
        else:
            continue

        registry_entry = find_registry_entry(dosya_yolu, registry_lookup)

        kategori = detect_category_from_filename(
            dosya_yolu,
            active_category_rules,
            registry_entry,
        )

        config = resolve_chunk_config(
            kategori,
            dosya_yolu,
            registry_entry,
            dynamic_chunk_configs,
        )

        yuklenen_dokumanlar = loader.load()

        for doc in yuklenen_dokumanlar:
            add_metadata(doc, dosya_yolu, kategori, config, registry_entry)

        split_key = (
            kategori,
            config["chunk_size"],
            config["chunk_overlap"],
        )

        dokumanlar_by_split_config.setdefault(split_key, []).extend(yuklenen_dokumanlar)

        if is_admin_document(dosya_yolu, registry_entry):
            admin_file_count += 1

        file_stats.append({
            "file_name": os.path.basename(dosya_yolu),
            "relative_path": os.path.relpath(dosya_yolu),
            "category": kategori,
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "admin_uploaded": is_admin_document(dosya_yolu, registry_entry),
            "document_count": len(yuklenen_dokumanlar),
        })

        print(
            f"Okundu: {dosya_yolu} | Kategori: {kategori} | "
            f"chunk_size={config['chunk_size']} | overlap={config['chunk_overlap']}",
            flush=True,
        )

    print("\nMetinler kategori ve chunk ayarlarına göre parçalanıyor...", flush=True)

    tum_chunklar = []
    split_stats = []

    for (kategori, chunk_size, chunk_overlap), dokumanlar in dokumanlar_by_split_config.items():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\nMADDE ", "\nMadde ", "\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )

        chunklar = splitter.split_documents(dokumanlar)

        for chunk in chunklar:
            chunk.metadata["kategori"] = kategori
            chunk.metadata["chunk_size_config"] = chunk_size
            chunk.metadata["chunk_overlap_config"] = chunk_overlap
            enrich_chunk_for_retrieval(chunk, kategori)

        tum_chunklar.extend(chunklar)

        stat = {
            "category": kategori,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "document_count": len(dokumanlar),
            "chunk_count": len(chunklar),
        }
        split_stats.append(stat)

        print(
            f"  [{kategori}] chunk_size={chunk_size} | overlap={chunk_overlap} | "
            f"{len(dokumanlar)} doküman → {len(chunklar)} chunk",
            flush=True,
        )

    print(f"\nToplam split grubu sayısı: {len(dokumanlar_by_split_config)}", flush=True)
    print(f"Toplam chunk sayısı: {len(tum_chunklar)}", flush=True)

    if not tum_chunklar:
        raise ValueError("Hiç chunk oluşmadı.")

    target_persist_dir = temp_persist_dir if safe_swap else persist_dir

    print("\nHedef ChromaDB klasörü hazırlanıyor...", flush=True)

    if os.path.exists(target_persist_dir):
        print(f"Geçici/eski hedef klasör siliniyor: {target_persist_dir}", flush=True)
        shutil.rmtree(target_persist_dir)

    if not safe_swap and os.path.exists(persist_dir):
        print(f"Eski ChromaDB doğrudan siliniyor: {persist_dir}", flush=True)
        shutil.rmtree(persist_dir)

    print("\nEmbedding modeli yükleniyor...", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("ChromaDB oluşturuluyor...", flush=True)
    db = Chroma.from_documents(
        documents=tum_chunklar,
        embedding=embeddings,
        persist_directory=target_persist_dir,
        collection_name=COLLECTION_NAME,
    )

    kayit_sayisi = db._collection.count()
    print(f"Chroma kayıt sayısı: {kayit_sayisi}", flush=True)

    if kayit_sayisi == 0:
        close_chroma_db(db)
        raise ValueError("ChromaDB oluşturuldu ama kayıt sayısı 0.")

    close_chroma_db(db)

    backup_dir = None

    if safe_swap:
        backup_dir = safe_swap_chroma_dirs(
            new_dir=target_persist_dir,
            active_dir=persist_dir,
        )

    result = {
        "status": "success",
        "created_at": now_iso(),
        "data_dir": data_dir,
        "persist_dir": persist_dir,
        "temp_persist_dir": temp_persist_dir,
        "backup_dir": backup_dir,
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "docx_count": len(docx_dosyalari),
        "txt_count": len(txt_dosyalari),
        "total_file_count": len(dokuman_dosyalari),
        "admin_file_count": admin_file_count,
        "split_group_count": len(dokumanlar_by_split_config),
        "chunk_count": len(tum_chunklar),
        "chroma_record_count": kayit_sayisi,
        "safe_swap": safe_swap,
        "chunk_policy": (
            "Mevcut kategori bazlı optimize edilmiş chunk ayarları korunur. "
            "document_registry.json içindeki chunk ayarları yalnızca admin/yeni dosyalara uygulanır."
        ),
        "file_stats": file_stats,
        "split_stats": split_stats,
    }

    registry["last_ingest"] = {
        key: value
        for key, value in result.items()
        if key not in ["file_stats", "split_stats"]
    }
    save_document_registry(registry, document_registry_path)

    print(f"\nVeritabanı başarıyla '{persist_dir}' klasörüne kaydedildi.", flush=True)
    print("App tarafında cache temizlenirse yeni DB aktif olarak kullanılacaktır.", flush=True)

    return result


# =========================================================
# CLI
# =========================================================

def main() -> None:
    result = run_ingest(
        data_dir=DATA_DIR,
        persist_dir=CHROMA_PATH,
        temp_persist_dir=CHROMA_NEW_PATH,
        dynamic_rules_path=DYNAMIC_RULES_PATH,
        document_registry_path=DOCUMENT_REGISTRY_PATH,
        safe_swap=True,
    )

    print("\nINGEST ÖZETİ")
    print("-" * 40)
    print(f"Durum: {result['status']}")
    print(f"Toplam dosya: {result['total_file_count']}")
    print(f"Admin dosyası: {result['admin_file_count']}")
    print(f"Toplam chunk: {result['chunk_count']}")
    print(f"Chroma kayıt: {result['chroma_record_count']}")
    print(f"Backup: {result['backup_dir']}")


if __name__ == "__main__":
    main()
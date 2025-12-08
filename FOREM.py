import requests
import pandas as pd
import unicodedata
from typing import Dict, Any, List, Set, Tuple, Optional


BASE_URL = "https://www.odwb.be/api/explore/v2.1/catalog/datasets/offres-d-emploi-forem/records"


def _strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _norm(text: str) -> str:
    return _strip_accents(text).lower().strip()


LANG_TO_ISO2 = {
    # French labels and common variants -> ISO2 upper
    "fr": "FR",
    "francais": "FR",
    "français": "FR",
    "en": "EN",
    "anglais": "EN",
    "nl": "NL",
    "neerlandais": "NL",
    "néerlandais": "NL",
    "de": "DE",
    "allemand": "DE",
    "es": "ES",
    "espagnol": "ES",
    "it": "IT",
    "italien": "IT",
    "ar": "AR",
    "arabe": "AR",
    "pt": "PT",
    "portugais": "PT",
}


def fetch_recent_offers(limit: int = 100, order_by: str = "numerooffreforem DESC") -> pd.DataFrame:
    """Fetch recent FOREM offers as a DataFrame.

    Columns of interest include: langues, languescodeiso2, regimetravail, etc.
    """
    # API max limit is 100
    limit = max(1, min(int(limit), 100))
    params = {"limit": limit, "order_by": order_by}
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("results", [])
    df = pd.json_normalize(rows)
    # Ensure list-like columns are lists
    for col in ["langues", "languescodeiso2", "lieuxtravaillocalite", "lieuxtravailcodepostal"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: list(x) if isinstance(x, (list, tuple)) else ([] if pd.isna(x) else [x]))
    return df


def _offer_lang_sets(row: pd.Series) -> Tuple[Set[str], Set[str]]:
    """Return (iso2_set_upper, name_set_norm) for an offer row."""
    iso2: Set[str] = set()
    if "languescodeiso2" in row and isinstance(row["languescodeiso2"], list):
        iso2 = {str(x).upper() for x in row["languescodeiso2"] if isinstance(x, str) and x}
    names: Set[str] = set()
    if "langues" in row and isinstance(row["langues"], list):
        names = {_norm(str(x)) for x in row["langues"] if isinstance(x, str) and x}
    # Derive iso2 from names when missing
    if not iso2 and names:
        for n in names:
            iso = LANG_TO_ISO2.get(n)
            if iso:
                iso2.add(iso)
    return iso2, names


def _user_lang_sets(profile: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """Return (iso2_set_upper, name_set_norm) from profile JSON (CV.py shape)."""
    iso2: Set[str] = set()
    names: Set[str] = set()
    langs = profile.get("langues") or []
    if isinstance(langs, dict):  # tolerate single object
        langs = [langs]
    for item in langs:
        if isinstance(item, dict):
            label = item.get("langue") or item.get("name") or item.get("label")
        else:
            label = str(item)
        if not label:
            continue
        t = _norm(str(label))
        names.add(t)
        if len(t) == 2 and t.isalpha():
            iso2.add(t.upper())
        else:
            iso = LANG_TO_ISO2.get(t)
            if iso:
                iso2.add(iso)
    return iso2, names


def filter_by_languages(df: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
    """Keep offers that require at least one language the user has.

    If profile has no languages, returns df unchanged.
    """
    user_iso2, user_names = _user_lang_sets(profile)
    if not user_iso2 and not user_names:
        return df

    def match(row: pd.Series) -> bool:
        offer_iso2, offer_names = _offer_lang_sets(row)
        # If an offer has no language requirement, keep it (broad match)
        if not offer_iso2 and not offer_names:
            return True
        if user_iso2 and (offer_iso2 & user_iso2):
            return True
        if user_names and (offer_names & user_names):
            return True
        # Cross compare using mapping
        mapped_user_iso = {LANG_TO_ISO2.get(n) for n in user_names if LANG_TO_ISO2.get(n)}
        if mapped_user_iso and (offer_iso2 & set(mapped_user_iso)):
            return True
        return False

    return df[df.apply(match, axis=1)]


def _canon_regime(value: str) -> str:
    v = _norm(value)
    if any(k in v for k in ["plein", "full"]):
        return "temps plein"
    if any(k in v for k in ["partiel", "mi-temps", "mi temps", "part time", "mi", "half"]):
        return "temps partiel"
    return v


def filter_by_regime(df: pd.DataFrame, desired: Optional[str]) -> pd.DataFrame:
    """Filter by working time regime if desired is provided.
    - desired "temps plein" -> keep offers whose regimetravail is Temps plein
    - desired "temps partiel" or mi-temps -> keep Temps partiel
    If desired is None/empty, returns df unchanged.
    """
    if not desired:
        return df
    target = _canon_regime(desired)
    if not target:
        return df

    def ok(row: pd.Series) -> bool:
        val = row.get("regimetravail")
        if not isinstance(val, str) or not val:
            return True  # keep if not specified by offer
        r = _canon_regime(val)
        if target == "temps plein":
            return "plein" in r
        if target == "temps partiel":
            return any(k in r for k in ["partiel", "mi-temps", "mi temps"])
        return True

    return df[df.apply(ok, axis=1)]


def _canon_contrat(value: str) -> str:
    v = _norm(value)
    mapping = {
        "interimaire avec option sur duree indeterminee": "intérimaire avec option sur durée indéterminée",
        "intérimaire avec option sur durée indeterminee": "intérimaire avec option sur durée indéterminée",
        "intérimaire avec option sur duree indéterminée": "intérimaire avec option sur durée indéterminée",
        "intérimaire avec option sur durée indéterminée": "intérimaire avec option sur durée indéterminée",
        "interimaire": "intérimaire",
        "intérimaire": "intérimaire",
        "duree indéterminée": "durée indéterminée",
        "duree indeterminee": "durée indéterminée",
        "durée indeterminee": "durée indéterminée",
        "durée indéterminée": "durée indéterminée",
        "cdi": "durée indéterminée",
        "duree déterminée": "durée déterminée",
        "duree determinee": "durée déterminée",
        "durée determinee": "durée déterminée",
        "durée déterminée": "durée déterminée",
        "cdd": "durée déterminée",
        "etudiant": "etudiant",
        "étudiant": "etudiant",
        "remplacement": "remplacement",
        "contrat collaboration indépendant": "contrat collaboration indépendant",
        "contrat collaboration independant": "contrat collaboration indépendant",
        "freelance": "contrat collaboration indépendant",
        "indépendant": "contrat collaboration indépendant",
        "independant": "contrat collaboration indépendant",
        "flexi-jobs": "flexi-jobs",
        "flexijobs": "flexi-jobs",
        "journalier": "journalier (occasionnel ou saisonnier)",
        "occasionnel": "journalier (occasionnel ou saisonnier)",
        "saisonnier": "journalier (occasionnel ou saisonnier)",
        "salarié statutaire": "salarié statutaire",
        "salarie statutaire": "salarié statutaire",
        "nettement défini": "nettement défini",
        "nettement defini": "nettement défini",
    }
    return mapping.get(v, v)


def filter_by_contrat(df: pd.DataFrame, desired: Optional[str]) -> pd.DataFrame:
    if not desired:
        return df
    target = _canon_contrat(desired)
    if not target:
        return df

    def ok(row: pd.Series) -> bool:
        val = row.get("typecontrat") or row.get("type_contrat")
        if not isinstance(val, str) or not val:
            return True  # keep if offer doesn't specify
        return _canon_contrat(val) == target

    return df[df.apply(ok, axis=1)]


def search_offers(profile: Dict[str, Any], limit: int = 200) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch and filter offers using profile data from CV.py JSON.

    Strategy:
    - Apply all available filters (languages, regime).
    - If empty, progressively relax (languages only, regime only, none).
    Returns (df, info) where info describes applied filters and relaxation.
    """
    raw = fetch_recent_offers(limit=limit)
    applied = []
    df = raw.copy()

    # Apply all filters: languages, regime, contract
    df_all = filter_by_languages(df, profile)
    df_all = filter_by_regime(df_all, (profile or {}).get("regime_travail_recherche"))
    df_all = filter_by_contrat(df_all, (profile or {}).get("type_contrat_recherche"))
    if len(df_all) > 0:
        return df_all, {"applied": ["languages", "regime", "contract"], "relaxed": 0, "total": len(raw), "kept": len(df_all)}

    # Relax: languages + regime (drop contract)
    df_lang_reg = filter_by_languages(df, profile)
    df_lang_reg = filter_by_regime(df_lang_reg, (profile or {}).get("regime_travail_recherche"))
    if len(df_lang_reg) > 0:
        return df_lang_reg, {"applied": ["languages", "regime"], "relaxed": 1, "total": len(raw), "kept": len(df_lang_reg)}

    # Relax: languages only
    df_lang = filter_by_languages(df, profile)
    if len(df_lang) > 0:
        return df_lang, {"applied": ["languages"], "relaxed": 2, "total": len(raw), "kept": len(df_lang)}

    # Relax: regime only
    df_reg = filter_by_regime(df, (profile or {}).get("regime_travail_recherche"))
    if len(df_reg) > 0:
        return df_reg, {"applied": ["regime"], "relaxed": 2, "total": len(raw), "kept": len(df_reg)}

    # Final fallback: return recent unfiltered
    return df, {"applied": [], "relaxed": 3, "total": len(raw), "kept": len(df)}


if __name__ == "__main__":
    # Demo rapide: construit un mini profil à partir d'un CV texte s'il existe
    try:
        import os
        from CV import read_cv, extract_langues, extract_pref_contrat, SCRIPT_DIR

        cv_path = os.path.join(SCRIPT_DIR, "mon_cv.txt")
        cv_text = read_cv(cv_path)
        local = {
            "langues": extract_langues(cv_text),
            **extract_pref_contrat(cv_text),
        }
        print("Profil détecté (local):", local)
        results, info = search_offers(local, limit=100)
        print("Filtrage:", info)
        cols = [
            "numerooffreforem",
            "titreoffre",
            "nomemployeur",
            "regimetravail",
            "langues",
            "languescodeiso2",
            "url",
        ]
        subset = [c for c in cols if c in results.columns]
        print(results[subset].head(20).to_string(index=False))
    except Exception as e:
        print("Demo échouée:", e)
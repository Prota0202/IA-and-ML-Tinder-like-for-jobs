import os
import re
import json
import sys
import math
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import requests
from green_metrics import tracker

from dotenv import load_dotenv

# Import LangChain/Mistral with graceful fallback
LANGCHAIN_AVAILABLE = False
try:
    from langchain.prompts import PromptTemplate  # type: ignore
    from langchain.chains import LLMChain  # type: ignore
    from langchain_mistralai.chat_models import ChatMistralAI  # type: ignore
    from langchain_core.callbacks import BaseCallbackHandler  # type: ignore
    LANGCHAIN_AVAILABLE = True
except Exception:
    PromptTemplate = None  # type: ignore
    LLMChain = None  # type: ignore
    ChatMistralAI = None  # type: ignore
    BaseCallbackHandler = None  # type: ignore
    import requests  # fallback HTTP direct if needed

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

# Determine CV path: CLI arg preferred, otherwise mon_cv.txt in same folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CV = os.path.join(SCRIPT_DIR, "mon_cv.txt")
CV_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CV

# ----------------------------- Reading CV -----------------------------


def read_cv(path: str) -> str:
    """Read a CV text file; try both given path and relative to script directory."""
    candidate_paths = [path]
    if not os.path.isabs(path):
        candidate_paths.append(os.path.join(SCRIPT_DIR, path))
    for p in candidate_paths:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except FileNotFoundError:
            continue
    return ""


# ------------------------- Heuristics & Extraction -----------------------


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _norm_lower(s: str) -> str:
    return _strip_accents(s).lower().strip()


# Improved language extraction (handles [translate:Français] and many variants)
def extract_langues(cv_text: str) -> List[Dict[str, Any]]:
    """
    Robust extraction of languages from CV text.
    Returns list of dicts: {'langue': '<canonical lower no-accent>', 'niveau': '<raw level or None>'}
    """
    if not cv_text or not isinstance(cv_text, str):
        return []
    def norm_key(s: str) -> str:
        return _norm_lower(s)

    langues: List[Dict[str, Any]] = []

    # 1) Try to find explicit LANGUES block
    m = re.search(r"\bLANGUES\b(.*?)(?:\n\s*\n|\Z)", cv_text, flags=re.IGNORECASE | re.DOTALL)
    candidates: List[str] = []
    if m:
        candidates = [l.strip() for l in m.group(1).splitlines() if l.strip()]
    else:
        # fallback: lines containing translate tag or language tokens
        for l in cv_text.splitlines():
            if re.search(r"\[translate:|\b(francais|français|anglais|néerlandais|neerlandais|espagnol|allemand|italien|portugais|arabe|b1|b2|c1|c2|natif|professionnel)\b", l, flags=re.IGNORECASE):
                candidates.append(l.strip())

    for line in candidates:
        lang = None
        niveau = None
        # pattern [translate:Français] : Natif
        m1 = re.search(r"\[translate\s*:\s*([^\]\:]+)\]\s*[:\-]?\s*(.*)", line, flags=re.IGNORECASE)
        if m1:
            lang = m1.group(1).strip()
            niveau = m1.group(2).strip() or None
        else:
            # pattern 'Français : Natif' or 'Français - Natif'
            m2 = re.match(r"([^:\-]+?)\s*[:\-]\s*(.+)", line)
            if m2:
                lang = m2.group(1).strip()
                niveau = m2.group(2).strip()
            else:
                # pattern 'Français B2' or 'Français (B2)'
                m3 = re.match(r"([A-Za-zéèÉÈçÇ\s]+?)\s*(?:\(|\:)?\s*([A-Za-z0-9+\- ]+)\)?$", line)
                if m3:
                    lang = m3.group(1).strip()
                    niveau = m3.group(2).strip()
                else:
                    # comma-separated list fallback
                    parts = [p.strip() for p in re.split(r"[;,/]", line) if p.strip()]
                    if parts:
                        # attempt to parse each comma part as language[:niveau]
                        for part in parts:
                            if ":" in part:
                                sub = part.split(":", 1)
                                l = sub[0].strip()
                                n = sub[1].strip()
                                key = norm_key(l)
                                if not any(d.get("langue") == key for d in langues):
                                    langues.append({"langue": key, "niveau": n or None})
                        continue
                    else:
                        continue
        if not lang:
            continue
        key = norm_key(lang)
        # canonical mapping
        mapping = {
            "francais": "francais",
            "français": "francais",
            "neerlandais": "neerlandais",
            "néerlandais": "neerlandais",
            "anglais": "anglais",
        }
        key = mapping.get(key, key)
        if niveau:
            niveau = niveau.replace("–", "-").strip()
            niveau = re.sub(r"\s+", " ", niveau)
        if not any(d.get("langue") == key for d in langues):
            langues.append({"langue": key, "niveau": niveau or None})

    # ultimate fallback: search tokens in whole text
    if not langues:
        low = _norm_lower(cv_text)
        for t in ["francais", "anglais", "neerlandais", "espagnol", "allemand", "italien", "portugais", "arabe"]:
            if t in low:
                langues.append({"langue": t, "niveau": None})
    return langues


# Improved studies extraction
_MONTHS_FR = r"(Janvier|Février|Fevrier|Mars|Avril|Mai|Juin|Juillet|Août|Aout|Septembre|Octobre|Novembre|Décembre|Decembre)"
_RE_PERIOD = re.compile(rf"(({_MONTHS_FR})\s+\d{{4}}|\d{{4}})", flags=re.IGNORECASE)


def extract_etudes(cv_text: str) -> List[Dict[str, Any]]:
    """
    Heuristic extraction for studies / education and short courses.
    Returns list of dicts: {'diplome', 'domaine', 'institution', 'periode'}
    """
    if not cv_text or not isinstance(cv_text, str):
        return []

    text = cv_text.replace("\r\n", "\n").replace("\r", "\n")
    # attempt to find FORMATION block; if not found, analyze whole text
    m = re.search(r"(?:^|\n)(FORMATION|FORMATION\s+ACADEMIQUE|FORMATION\s+COMPL[eé]MENTAIRE|FORMATION\s+ET)(.*?)(?:\n\n|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    block = m.group(2).strip() if m else text

    paras = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
    results: List[Dict[str, Any]] = []

    inst_indicators = ["école", "ecole", "universit", "haute", "institut", "college", "centre", "h e r s", "h e r s"]

    def _find_period(s: str) -> Optional[str]:
        m_obt = re.search(r"Obtenu\s*[:\-]\s*([A-Za-z0-9\s]+)", s, flags=re.IGNORECASE)
        if m_obt:
            return m_obt.group(1).strip()
        m2 = _RE_PERIOD.search(s)
        if m2:
            return m2.group(0).strip()
        return None

    for p in paras:
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        if not lines:
            continue

        diploma = None
        institution = None
        periode = None
        domaine = None

        first = lines[0]
        if re.search(r"\b(Dipl[oô]me|Diplome|Master|Licence|Bachelier|Bachelor|Certificat|Certification|Formation)\b", first, flags=re.IGNORECASE):
            diploma = first
            remaining = lines[1:]
        else:
            diploma_idx = None
            for i, ln in enumerate(lines):
                if re.search(r"\b(Dipl[oô]me|Diplome|Master|Licence|Bachelier|Bachelor|Certificat|Certification)\b", ln, flags=re.IGNORECASE):
                    diploma_idx = i
                    break
            if diploma_idx is not None:
                diploma = lines[diploma_idx]
                if diploma_idx > 0:
                    institution = lines[diploma_idx - 1]
                remaining = lines[diploma_idx + 1 :]
            else:
                diploma = first
                remaining = lines[1:]

        if not institution:
            for ln in remaining[:2]:
                low = ln.lower()
                if any(k in low for k in inst_indicators) or (len(ln) > 20 and any(ch.isupper() for ch in ln[:5])):
                    institution = ln
                    break

        perio = _find_period(p)
        if perio:
            periode = perio
        else:
            for ln in lines:
                pr = _find_period(ln)
                if pr:
                    periode = pr
                    break

        for ln in lines:
            if re.search(r"\b(Sp[eé]cialisatio|Sp[eé]cialit|Specialisation|Sp[eé]cialisation)\b", ln, flags=re.IGNORECASE):
                parts = re.split(r":", ln, maxsplit=1)
                domaine = parts[1].strip() if len(parts) > 1 else ln.strip()
                break

        def _clean(s: Optional[str]) -> Optional[str]:
            if not s:
                return None
            return re.sub(r"\s{2,}", " ", s).strip(" -–:;,")

        diploma = _clean(diploma)
        institution = _clean(institution)
        periode = _clean(periode)
        domaine = _clean(domaine)

        if diploma and len(diploma) < 3:
            diploma = None

        entry = {"diplome": diploma or None, "domaine": domaine or None, "institution": institution or None, "periode": periode or None}
        if not any(e.get("diplome") == entry["diplome"] and e.get("institution") == entry["institution"] for e in results):
            results.append(entry)

    # Additional pass: detect 'Formation Complémentaire' lines elsewhere
    try:
        more_block = re.search(r"(Formation Compl[uú]mentaire|Formation Compl[eè]mentaire|Compl[eè]mentaire)(.*?)(?:\n\n|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        if more_block:
            mb = more_block.group(2).strip()
            for line in [l.strip(" -•") for l in mb.splitlines() if l.strip()]:
                if len(line) > 5:
                    inst = None
                    period = None
                    m_period = _RE_PERIOD.search(line)
                    if m_period:
                        period = m_period.group(0)
                    parts = re.split(r"\s+[-–]\s+|\s*\|\s*", line)
                    if len(parts) >= 2:
                        d = parts[0].strip()
                        inst = parts[1].strip()
                        results.append({"diplome": d or None, "domaine": None, "institution": inst or None, "periode": period or None})
                    else:
                        m_paren = re.match(r"(.+?)\s*\((.+?)\)", line)
                        if m_paren:
                            results.append({"diplome": m_paren.group(1).strip(), "domaine": None, "institution": m_paren.group(2).strip(), "periode": period or None})
                        else:
                            results.append({"diplome": line, "domaine": None, "institution": None, "periode": period or None})
    except Exception:
        pass

    return results


# Improved localisation extraction
def extract_localisation(cv_text: str) -> Dict[str, Any]:
    """
    Return {'ville': <city or None>, 'code_postal': <postal code or None>}
    Heuristics:
    - Look at first 8 lines for 'Bruxelles' or postal code 4 digits (Belgium)
    - Try to parse lines 'Adresse: Bruxelles, Belgique' etc.
    - Fallback to common city list
    """
    loc = {"ville": None, "code_postal": None}
    if not cv_text:
        return loc
    lines = [l.strip() for l in cv_text.splitlines() if l.strip()]
    # check first lines for address-like info
    for line in lines[:12]:
        ll = line.lower()
        # postal code 4 digits (Belgium)
        m = re.search(r"\b(\d{4})\b", line)
        if m and not loc["code_postal"]:
            loc["code_postal"] = m.group(1)
        # explicit city mentions
        if "bruxelles" in ll and not loc["ville"]:
            loc["ville"] = "Bruxelles"
        # address line like 'Adresse: Bruxelles, Belgique'
        if "adresse" in ll or "adresse:" in ll:
            # attempt to extract after ':'
            parts = line.split(":", 1)
            if len(parts) > 1:
                after = parts[1].strip()
                if after:
                    # take first token or city-like token
                    loc["ville"] = after.split(",")[0].strip()
    # fallback: extract city if postal code found but no city
    if loc["code_postal"] and not loc["ville"]:
        # try to find token around cp in lines
        for line in lines:
            if loc["code_postal"] in line:
                tokens = [t for t in re.split(r"[,\-\|]", line) if t.strip()]
                for t in tokens:
                    if loc["code_postal"] in t:
                        # find neighbor token
                        parts = re.split(r"\s+", t)
                        try:
                            idx = parts.index(loc["code_postal"])
                            # try previous or next token
                            if idx > 0:
                                loc["ville"] = parts[idx - 1].strip(", ")
                            elif idx + 1 < len(parts):
                                loc["ville"] = parts[idx + 1].strip(", ")
                        except Exception:
                            continue
    # final fallback: common cities
    if not loc["ville"]:
        common = ["bruxelles", "liege", "namur", "charleroi", "gent", "antwerpen", "brugge"]
        lowtext = cv_text.lower()
        for c in common:
            if c in lowtext:
                loc["ville"] = c.capitalize()
                break
    return loc


# Contract preference heuristics (kept from original with minor compatibility)
def extract_pref_contrat(cv_text: str) -> Dict[str, Any]:
    lower = cv_text.lower()
    pref = {"type_contrat_recherche": None, "regime_travail_recherche": None}

    regime_map = {
        "temps plein": "Temps plein",
        "plein temps": "Temps plein",
        "full time": "Temps plein",
        "temps partiel": "Temps partiel",
        "part time": "Temps partiel",
        "mi-temps": "Temps partiel",
        "half time": "Temps partiel",
    }
    for key, canon in regime_map.items():
        if key in lower:
            pref["regime_travail_recherche"] = canon
            break

    contrat_map = {
        "intérimaire avec option sur duré(e?) indéterminé(e?)": "Intérimaire avec option sur durée indéterminée",
        "interimaire avec option sur duree indeterminee": "Intérimaire avec option sur durée indéterminée",
        "intérimaire": "Intérimaire",
        "interimaire": "Intérimaire",
        "durée indéterminée": "Durée indéterminée",
        "duree indeterminee": "Durée indéterminée",
        "cdi": "Durée indéterminée",
        "durée déterminée": "Durée déterminée",
        "duree determinee": "Durée déterminée",
        "cdd": "Durée déterminée",
        "étudiant": "Etudiant",
        "etudiant": "Etudiant",
        "remplacement": "Remplacement",
        "contrat collaboration indépendant": "Contrat collaboration indépendant",
        "freelance": "Contrat collaboration indépendant",
        "indépendant": "Contrat collaboration indépendant",
        "flexi-jobs": "Flexi-Jobs",
    }

    found = None
    # long keys first
    for key in sorted(contrat_map.keys(), key=len, reverse=True):
        if key in lower:
            found = contrat_map[key]
            break
    if found:
        pref["type_contrat_recherche"] = found

    return pref


# ------------------------------ Prompt LLM ----------------------------
# Keep the original prompt (LLM expected to return strict JSON)
PROMPT_STR = (
    "Tu reçois le texte intégral d'un CV. Objectif: préparer la recherche d'emploi. "
    "1) Extrait STRICTEMENT les informations présentes. 2) Génère des questions pertinentes pour CHAQUE champ manquant. "
    "Ne fabrique pas d'information absente.\n"
    "Renvoie SEULEMENT un JSON valide du format exact: {\n"
    "  \"etudes\": [ { \"diplome\":..., \"domaine\":..., \"institution\":..., \"periode\":... } ],\n"
    "  \"domaines_preference\": [ ... ],\n"
    "  \"langues\": [ { \"langue\":..., \"niveau\":... } ],\n"
    "  \"type_contrat_recherche\": ...,\n"
    "  \"regime_travail_recherche\": ...,\n"
    "  \"localisation\": { \"ville\":..., \"code_postal\":... },\n"
    "  \"autres\": [ ... ],\n"
    "  \"PreviousCompanies\": <entier, 0 si inconnu>,\n"
    "  \"ExperienceYears\": <nombre, 0 si inconnu>,\n"
    "  \"EducationLevel\": <entier, 0 si inconnu; 1=Bac, 2=Bac+3, 3=Master, 4=PhD>,\n"
    "  \"Age\": <nombre, NaN si inconnu>,\n"
    "  \"questions\": [ {\n"
    "      \"champ\": \"nom_du_champ\",\n"
    "      \"question\": \"Texte question claire\",\n"
    "      \"format_attendu\": \"exemple ou options\",\n"
    "      \"type\": \"choice|text\",\n"
    "      \"options\": [\"option1\", \"option2\"]\n"
    "  } ]\n"
    "}\n"
    "Règles supplémentaires:\n"
    "- Interprète synonymes type de contrat: CDI -> Durée indéterminée; CDD -> Durée déterminée; Freelance/indépendant -> Contrat collaboration indépendant; mi-temps -> Temps partiel.\n"
    "- Valeurs autorisées type_contrat_recherche (sélectionne la meilleure si le CV en offre plusieurs): Intérimaire avec option sur durée indéterminée, Durée indéterminée, Intérimaire, Durée déterminée, Etudiant, Remplacement, Contrat collaboration indépendant, Flexi-Jobs, Journalier (occasionnel ou saisonnier), Salarié statutaire, Nettement défini.\n"
    "- Valeurs autorisées regime_travail_recherche: Temps plein, Temps partiel.\n"
    "- Si localisation partielle (absence code postal) laisse code_postal à null.\n"
    "- N'AJOUTE DES QUESTIONS QUE pour les champs NULL/VIDES après extraction stricte.\n"
    "- Pour les questions: si \"champ\" est \"regime_travail_recherche\", mets \"type\": \"choice\" et \"options\": [\"Temps plein\", \"Temps partiel\"]. Si \"champ\" est \"type_contrat_recherche\", mets \"type\": \"choice\" et \"options\" = liste complète ci-dessus. Si \"champ\" est \"code_postal\" ou \"ville\", mets \"type\": \"text\". Pour \"langues\" et \"etudes\", mets \"type\": \"text\" et un \"format_attendu\" explicite (ex: langues: \"francais:C2; anglais:B2\").\n"
    "- Le JSON doit être directement parsable sans texte autour.\n\nCV:\n{cv_text}"
)

# ------------------------------ LLM Chain -----------------------------


def build_chain():
    if not API_KEY or not LANGCHAIN_AVAILABLE or LLMChain is None or ChatMistralAI is None:
        return None
    llm = ChatMistralAI(mistral_api_key=API_KEY, model="mistral-small-latest")
    prompt = PromptTemplate(input_variables=["cv_text"], template=PROMPT_STR)
    return LLMChain(llm=llm, prompt=prompt)


# --- LangChain callback pour logger les tokens et alimenter le tracker ---
class TokenLogger(BaseCallbackHandler if BaseCallbackHandler else object):
    def on_llm_end(self, response, **kwargs):  # type: ignore
        try:
            usage = (response.llm_output or {}).get("token_usage", {})
            inp = usage.get("prompt_tokens", 0)
            out = usage.get("completion_tokens", 0)
            tracker.add_mistral_tokens(int(inp or 0), int(out or 0))
        except Exception:
            pass


def call_mistral_fallback(cv_text: str) -> str:
    """Direct HTTP call to Mistral if LangChain unavailable. Returns raw content or error message."""
    if not API_KEY:
        return "(Pas de clé API)"
    
   # tracker.start()  # Début du tracking
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": PROMPT_STR.replace("{cv_text}", cv_text)}],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            return f"Erreur HTTP {resp.status_code}: {resp.text}"
        data = resp.json()
        
        # Récupère les tokens depuis la réponse Mistral et les envoie au tracker
        tracker.add_mistral_tokens(data)
        
        #tracker.stop()  # Fin du tracking
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        tracker.stop()
        return f"Exception requête: {e}"


def call_mistral_generic(prompt_text: str) -> str:
    """Generic HTTP call to Mistral for any prompt (not just CV extraction)."""
    if not API_KEY:
        return "(Pas de clé API)"
    
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            return f"Erreur HTTP {resp.status_code}: {resp.text}"
        data = resp.json()
        
        # Récupère les tokens depuis la réponse Mistral et les envoie au tracker
        tracker.add_mistral_tokens(data)
        
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return f"Exception requête: {e}"


# ---------------------------- Parsing JSON ----------------------------


def safe_json_extract(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    raw = text[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ------------------------- Merge & Fill Profile -----------------------


def fuse(local: Dict[str, Any], llm: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "etudes": llm.get("etudes") or local.get("etudes") or [],
        "domaines_preference": llm.get("domaines_preference") or [],
        "langues": llm.get("langues") or local.get("langues") or [],
        "type_contrat_recherche": llm.get("type_contrat_recherche") or local.get("type_contrat_recherche"),
        "regime_travail_recherche": llm.get("regime_travail_recherche") or local.get("regime_travail_recherche"),
        "localisation": llm.get("localisation") or local.get("localisation") or {"ville": None, "code_postal": None},
        "autres": llm.get("autres") or local.get("autres") or [],
    }


def find_missing(profile: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    if not profile.get("etudes"):
        missing.append("etudes")
    if not profile.get("langues"):
        missing.append("langues")
    if not profile.get("type_contrat_recherche"):
        missing.append("type_contrat_recherche")
    if not profile.get("regime_travail_recherche"):
        missing.append("regime_travail_recherche")
    loc = profile.get("localisation") or {}
    if not (loc.get("ville")):
        missing.append("ville")
    if not (loc.get("code_postal")):
        missing.append("code_postal")
    return missing


def interactive_fill_with_questions(profile: Dict[str, Any], llm_json: Dict[str, Any]) -> Dict[str, Any]:
    questions = llm_json.get("questions") or []
    if not questions:
        return profile
    if not sys.stdin.isatty():
        print("[LOCAL] Mode non interactif: questions ignorées.")
        return profile
    print("[LOCAL] Questions pour compléter le profil :")
    for q in questions:
        champ = q.get("champ")
        texte = q.get("question")
        fmt = q.get("format_attendu")
        if not champ or not texte:
            continue
        answer = input(f" - {texte} (format: {fmt}): ").strip()
        if not answer:
            continue
        if champ == "langues":
            lang_list = []
            for chunk in answer.split(";"):
                parts = chunk.split(":")
                if len(parts) >= 2:
                    lang_list.append({"langue": parts[0].strip().lower(), "niveau": parts[1].strip()})
                elif parts[0].strip():
                    lang_list.append({"langue": parts[0].strip().lower(), "niveau": None})
            if lang_list:
                profile["langues"] = lang_list
        elif champ == "etudes":
            etude_objs = []
            for chunk in answer.split(";"):
                parts = [p.strip() for p in chunk.split("|")]
                if len(parts) >= 4:
                    etude_objs.append({"diplome": parts[0], "domaine": parts[1] or None, "institution": parts[2] or None, "periode": parts[3] or None})
                elif parts and parts[0]:
                    etude_objs.append({"diplome": parts[0], "domaine": None, "institution": None, "periode": None})
            if etude_objs:
                profile["etudes"] = etude_objs
        elif champ in ["type_contrat_recherche", "regime_travail_recherche"]:
            profile[champ] = answer
        elif champ in ["localisation"]:
            parts = [p.strip() for p in answer.split("|")]
            loc = profile.get("localisation") or {"ville": None, "code_postal": None}
            if parts:
                loc["ville"] = parts[0]
            if len(parts) > 1:
                loc["code_postal"] = parts[1]
            profile["localisation"] = loc
    return profile


# ---------------------------- Main Flow -------------------------------


def main():
    cv_text = read_cv(CV_PATH)
    if not cv_text:
        print(f"CV introuvable ou vide: {CV_PATH}\nAstuce: lance 'python CV.py chemin/vers/mon_cv.txt' ou place 'mon_cv.txt' dans {SCRIPT_DIR}")
        return

    print("Lecture du CV effectuée. Longueur:", len(cv_text), "caractères")

    local_data = {
        "langues": extract_langues(cv_text),
        "etudes": extract_etudes(cv_text),
        "localisation": extract_localisation(cv_text),
        **extract_pref_contrat(cv_text),
        "autres": [],  # placeholder: could use an extract_autres function if needed
    }
    print("Heuristiques locales:")
    print(json.dumps(local_data, ensure_ascii=False, indent=2))

    if not API_KEY:
        print("MISTRAL_API_KEY manquant: saut de l'appel LLM.")
        llm_json = {}
    else:
        chain = build_chain()
        if chain is not None:
            try:
                # Utilise TokenLogger pour tracer les tokens côté LangChain
                if BaseCallbackHandler:
                    raw_resp = chain.invoke({"cv_text": cv_text}, config={"callbacks": [TokenLogger()]})
                    raw = raw_resp.get("text") if isinstance(raw_resp, dict) else raw_resp
                else:
                    raw = chain.run({"cv_text": cv_text})
            except Exception as e:
                raw = f"Erreur LLM: {e}"
        else:
            raw = call_mistral_fallback(cv_text)
        print("Réponse brute LLM:")
        print(raw)
        llm_json = safe_json_extract(raw)
        print("JSON parsé LLM:")
        print(json.dumps(llm_json, ensure_ascii=False, indent=2) if llm_json else "(Parsing échoué)")

    profile = fuse(local_data, llm_json)
    if llm_json.get("questions"):
        profile = interactive_fill_with_questions(profile, llm_json)
    else:
        missing = find_missing(profile)
        profile = interactive_fill_with_questions(profile, {"questions": [{"champ": m, "question": f"Veuillez renseigner {m}", "format_attendu": "Texte"} for m in missing]})

    print("\nProfil final consolidé:")
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print("\nProchaine étape: utiliser ce profil pour requêter une API d'offres.")


if __name__ == "__main__":
    main()
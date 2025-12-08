# NOTE: below is the original webapp.py with only minimal edits to integrate geo.add_distance_column.
# I kept the original structure and replaced/added the small sections required.
import os
import io
import csv
import json
import secrets
import tempfile
from typing import Dict, Any, List

from flask import Flask, request, redirect, url_for, session, make_response

try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None  # type: ignore

try:
    from zamzar import ZamzarClient  # type: ignore
except Exception:
    ZamzarClient = None  # type: ignore

from dotenv import load_dotenv
load_dotenv()
ZAMZAR_KEY = os.getenv("ZAMZAR_API_KEY")

from CV import (
    extract_langues,
    extract_etudes,
    extract_localisation,
    extract_pref_contrat,
    fuse,
    find_missing,
    build_chain,
    call_mistral_fallback,
    safe_json_extract,
    API_KEY,
)
from FOREM import search_offers
try:
    import joblib  # type: ignore
except Exception:
    joblib = None  # type: ignore
import numpy as np
import pandas as pd
import re
from datetime import datetime

# === NEW import: geocoding helper ===
from geo import add_distance_column

ALLOWED_CONTRACTS = [
    "Intérimaire avec option sur durée indéterminée",
    "Durée indéterminée",
    "Intérimaire",
    "Durée d'éterminée",
    "Etudiant",
    "Remplacement",
    "Contrat collaboration indépendant",
    "Flexi-Jobs",
    "Journalier (occasionnel ou saisonnier)",
    "Salarié statutaire",
    "Nettement défini",
]
REGIME_OPTIONS = ["Temps plein", "Temps partiel"]


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    return app


app = create_app()

# In-memory store (avoid putting big data in cookie session)
# NOTE: fine for local dev; for production, use a DB or Flask-Session.
STORE_TEXT: dict[str, str] = {}
STORE_OFFERS: dict[str, list[dict]] = {}
STORE_ACCEPTED: dict[str, list[dict]] = {}
EMPLOYABILITY_MODEL_PATH = os.getenv("EMPLOYABILITY_MODEL", os.path.join(os.path.dirname(__file__), "models", "hire_tabular_nocal.joblib"))
EMPLOYABILITY_MODEL_PATH_CORE = os.getenv("EMPLOYABILITY_MODEL_CORE", os.path.join(os.path.dirname(__file__), "models", "hire_tabular_core.joblib"))


@app.before_request
def ensure_sid():
    if "sid" not in session:
        session["sid"] = secrets.token_urlsafe(16)


def extract_text_local(file_path: str) -> str:
    """
    Essaie d'extraire le texte d'un PDF local :
    1) PyPDF2 (rapide)
    2) pdfminer.six (plus robuste sur certains PDF)
    Renvoie un message explicite si aucune méthode n'est disponible.
    """
    # 1) PyPDF2
    if PyPDF2:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                parts = []
                for page in reader.pages:
                    try:
                        txt = page.extract_text()
                        if txt:
                            parts.append(txt)
                    except Exception:
                        continue
            combined = "\n".join(parts)
            if combined and combined.strip():
                return combined
            # si résultat vide, on fall back vers pdfminer
        except Exception:
            pass

    # 2) pdfminer.six fallback
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text  # lazy import
    except Exception:
        pdfminer_extract_text = None

    if pdfminer_extract_text:
        try:
            txt = pdfminer_extract_text(file_path)
            if txt and txt.strip():
                return txt
            return "(Extraction locale PDF: résultat vide)"
        except Exception as e:
            return f"Erreur PDF locale (pdfminer): {e}"

    return "(PyPDF2 non installé et pdfminer non installé)"


def extract_text_zamzar(file_path: str) -> str:
    if not (ZAMZAR_KEY and ZamzarClient):
        return ""
    try:
        client = ZamzarClient(ZAMZAR_KEY)
        job = client.convert(file_path, "txt")
        out_dir = tempfile.mkdtemp(prefix="zamzar_")
        job.store(out_dir)
        for name in os.listdir(out_dir):
            if name.lower().endswith(".txt"):
                with open(os.path.join(out_dir, name), "r", encoding="utf-8", errors="replace") as f:
                    data = f.read()
                job.delete_all_files()
                return data
        job.delete_all_files()
        return ""
    except Exception as e:
        return f"Erreur Zamzar: {e}"


def build_profile(cv_text: str) -> Dict[str, Any]:
    local = {
        "langues": extract_langues(cv_text),
        "etudes": extract_etudes(cv_text),
        "localisation": extract_localisation(cv_text),
        **extract_pref_contrat(cv_text),
    }
    # Try LLM for richer extraction + dynamic questions
    llm_json: Dict[str, Any] = {}
    try:
        if API_KEY:
            chain = build_chain()
            if chain is not None:
                raw = chain.run({"cv_text": cv_text})
            else:
                raw = call_mistral_fallback(cv_text)
            llm_json = safe_json_extract(raw)
    except Exception:
        llm_json = {}
    # store questions aside in session
    session["llm_questions"] = (llm_json or {}).get("questions") or []
    return fuse(local, llm_json or {})


def _edu_level_from_etudes(etudes: list[dict] | None) -> int:
    if not etudes:
        return 0
    text = " ".join([str(e.get("diplome", "")) for e in etudes]).lower()
    if any(k in text for k in ["phd", "doctorat", "doctorate"]):
        return 4
    if any(k in text for k in ["master", "msc", "maitrise"]):
        return 3
    if any(k in text for k in ["bachelor", "licence", "bsc", "ba"]):
        return 2
    if any(k in text for k in ["diplome", "certificat", "certification"]):
        return 1
    return 0


def extract_tabular_features(profile: Dict[str, Any]) -> dict:
    etudes = profile.get("etudes") or []
    edu_level = _edu_level_from_etudes(etudes)
    # Heuristics: count previous companies via experiences list if provided by LLM; else 0
    prev_companies = 0
    for k in ("experiences", "experience", "jobs", "entreprises"):
        arr = profile.get(k) or []
        if isinstance(arr, list):
            prev_companies = max(prev_companies, len(arr))
    # Experience years if present in profile
    exp_years = 0
    for k in ("experience_years", "annees_experience", "exp_years"):
        v = profile.get(k)
        try:
            if v is not None:
                exp_years = max(exp_years, int(v))
        except Exception:
            pass
    # Fallback: derive years from raw CV text (French date ranges)
    if not exp_years:
        try:
            sid = session.get("sid")
            raw = STORE_TEXT.get(sid or "") or ""
            t = raw.lower()
            MONTHS = {
                'janvier': 1, 'février': 2, 'fevrier': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6,
                'juillet': 7, 'août': 8, 'aout': 8, 'septembre': 9, 'octobre': 10, 'novembre': 11,
                'décembre': 12, 'decembre': 12
            }
            date_pat = re.compile(r"(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|d[ée]cembre)\s+(\d{4})\s*[–-]\s*(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|d[ée]cembre|pr[ée]sent|present)\s*(\d{4})?", re.IGNORECASE)
            exp_months = 0
            now = datetime.now()
            for m in date_pat.finditer(t):
                sm, sy = m.group(1), m.group(2)
                em, ey = m.group(3), m.group(4)
                start_mo = MONTHS.get(sm, 1)
                start_yr = int(sy)
                if em and ('present' in em or 'présent' in em):
                    end_mo = now.month
                    end_yr = now.year
                else:
                    end_mo = MONTHS.get(em, start_mo) if em else start_mo
                    end_yr = int(ey) if ey and ey.isdigit() else start_yr
                months = max(0, (end_yr - start_yr) * 12 + (end_mo - start_mo))
                exp_months += min(months, 10 * 12)
            # simple year spans
            for m in re.finditer(r"(19\d{2}|20\d{2})\s*[–-]\s*(19\d{2}|20\d{2})", t):
                try:
                    y1, y2 = int(m.group(1)), int(m.group(2))
                    if y2 > y1:
                        exp_months += min((y2 - y1) * 12, 10 * 12)
                except Exception:
                    pass
            exp_years = int(round(exp_months / 12.0)) if exp_months > 0 else exp_years
        except Exception:
            pass
    # Age if present
    age = profile.get("age")
    try:
        age = int(age) if age is not None else np.nan
    except Exception:
        age = np.nan
    # Fallback for previous companies via role headers in raw text
    if prev_companies == 0:
        try:
            sid = session.get("sid")
            t = (STORE_TEXT.get(sid or "") or "").lower()
            role_headers = re.findall(r"\n([a-z][^\n]{0,60})\n(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|d[ée]cembre)\s+\d{4}", t, flags=re.IGNORECASE)
            prev_companies = max(prev_companies, len(role_headers))
        except Exception:
            pass
    return {
        "ExperienceYears": exp_years,
        "PreviousCompanies": prev_companies,
        "EducationLevel": edu_level,
        "Age": age,
    }


def score_employability(profile: Dict[str, Any]) -> float | None:
    try:
        if joblib is None:
            return None
        # Prefer core-features model if available
        model_path = EMPLOYABILITY_MODEL_PATH_CORE if os.path.exists(EMPLOYABILITY_MODEL_PATH_CORE) else EMPLOYABILITY_MODEL_PATH
        if not os.path.exists(model_path):
            return None
        bundle = joblib.load(model_path)
        pipeline = bundle.get("pipeline", bundle.get("model", bundle)) if isinstance(bundle, dict) else bundle
        features = bundle.get("features") if isinstance(bundle, dict) else None
        feats = extract_tabular_features(profile)
        # Restrict to core features only
        core_keys = ["ExperienceYears", "PreviousCompanies", "EducationLevel", "Age"]
        if features:
            row = {name: (feats.get(name, np.nan) if name in core_keys else np.nan) for name in features}
        else:
            row = {k: feats.get(k, np.nan) for k in core_keys}
        X = pd.DataFrame([row])
        if hasattr(pipeline, "predict_proba"):
            return float(pipeline.predict_proba(X)[:, 1][0])
        if hasattr(pipeline, "decision_function"):
            val = float(pipeline.decision_function(X)[0])
            return 1.0 / (1.0 + np.exp(-val))
        pred = int(pipeline.predict(X)[0])
        return float(pred)
    except Exception:
        return None


def html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'><title>{title}</title>

<style>body{{font-family:Arial;margin:30px;max-width:900px}}input,select,textarea{{width:100%;margin:4px 0;padding:6px}}.offer{{border:1px solid #ccc;padding:12px;margin-bottom:14px;border-radius:6px}}</style>

</head><body>

<nav><a href='{url_for('index')}'>Accueil</a> <a href='{url_for('profile')}'>Profil</a> <a href='{url_for('offers')}'>Offres</a> <a href='{url_for('accepted')}'>Acceptées</a></nav><hr/>{body}

</body></html>"""


@app.route("/", methods=["GET"])
def index():
    body = """
    <h1>Assistant Emploi (PDF → Profil → Offres)</h1>
    <p>Étapes: 1) Charger CV PDF, 2) Compléter profil, 3) Filtrer offres FOREM, 4) Accepter/Rejeter.</p>
    <form action='/upload' method='post' enctype='multipart/form-data'>
      <label>CV PDF:</label><input type='file' name='cv_pdf' accept='application/pdf' required />
      <button class='btn primary' type='submit'>Téléverser & Extraire</button>
    </form>
    """
    sid = session.get("sid")
    if sid and STORE_TEXT.get(sid):
        body += "<p style='color:green'>CV déjà importé. <a href='/profile'>Aller au profil</a></p>"
    return html_page("Accueil", body)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("cv_pdf")
    if not file:
        return redirect(url_for("index"))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    file.save(tmp.name)
    text = extract_text_zamzar(tmp.name) or extract_text_local(tmp.name)
    # debug: afficher un aperçu du texte extrait pour vérifier l'extraction
    try:
        preview = text[:2000].replace("\n", "\\n")
        print("[DEBUG] texte extrait (début, 2000 chars):")
        print(preview)
    except Exception as e:
        print("[DEBUG] erreur affichage preview texte extrait:", e)
        
    os.unlink(tmp.name)
    sid = session.get("sid")
    if sid:
        STORE_TEXT[sid] = text[:200000]
        STORE_OFFERS.pop(sid, None)
        STORE_ACCEPTED.pop(sid, None)
    session.pop("profile_json", None)
    session.pop("offers_info", None)
    session.pop("offer_index", None)
    return redirect(url_for("profile"))


def render_profile_form(profile: Dict[str, Any]) -> str:
    questions = session.get("llm_questions") or []
    # If LLM did not provide structured questions, fallback to simple missing list
    missing = find_missing(profile) if not questions else []
    def safe(v):
        return "" if v is None else str(v)
    loc = profile.get("localisation") or {"ville": None, "code_postal": None}
    html = ["<h2>Profil extrait</h2>"]
    html.append(f"<pre style='background:#f8f9fa;padding:10px;border-radius:6px'>{json.dumps(profile, ensure_ascii=False, indent=2)}</pre>")
    html.append("<h3>Compléter les informations manquantes</h3>")
    html.append("<form method='post' action='/profile'>")
    if questions:
        # Render based on questions list provided by LLM
        for q in questions:
            champ = (q or {}).get("champ")
            qtext = (q or {}).get("question") or champ
            qtype = (q or {}).get("type") or "text"
            options = (q or {}).get("options") or []
            if champ in ("langues", "etudes"):
                placeholder = "francais:C2; anglais:B2" if champ == "langues" else "Diplôme|Domaine|Institution|Période;..."
                rows = 3 if champ == "etudes" else 1
                if rows == 1:
                    html.append(f"<label>{qtext}<input name='{champ}' placeholder='{placeholder}'/></label>")
                else:
                    html.append(f"<label>{qtext}<textarea name='{champ}' rows='{rows}' placeholder='{placeholder}'></textarea></label>")
            elif champ == "type_contrat_recherche":
                opts = options or ALLOWED_CONTRACTS
                html.append("<label>Type de contrat<select name='type_contrat'><option value=''>-- Non précisé --</option>" + "".join(f"<option>{c}</option>" for c in opts) + "</select></label>[...]")
            elif champ == "regime_travail_recherche":
                opts = options or REGIME_OPTIONS
                html.append("<fieldset><legend>Régime de travail</legend>" + "".join(f"<label><input type='radio' name='regime' value='{r}'> {r}</label>" for r in opts) + "<label><input type='radio' name='regime' value=''>Non spécifié</label></fieldset>")
            
            elif champ in ("ville", "code_postal"):
                html.append(f"<label>{qtext}<input name='{champ}' value='{safe(loc.get(champ))}'/></label>")
            else:
                # generic text
                html.append(f"<label>{qtext}<input name='{champ}'/></label>")
    else:
        # Fallback simple: render by missing list
        if "langues" in missing:
            html.append("<label>Langues (format langue:niveau séparé par ';')<input name='langues' placeholder='francais:C2; anglais:B2'/></label>")
        if "etudes" in missing:
            html.append("<label>Études (Diplôme|Domaine|Institution|Période;...)<textarea name='etudes' rows='3'></textarea></label>")
        if "type_contrat_recherche" in missing:
            html.append("<label>Type de contrat recherché<select name='type_contrat'><option value=''>-- Non précisé --</option>" + "".join(f"<option>{c}</option>" for c in ALLOWED_CONTRACTS) + "[...]")   
        if "regime_travail_recherche" in missing:
            html.append("<fieldset><legend>Régime de travail</legend>" + "".join(f"<label><input type='radio' name='regime' value='{r}'> {r}</label>" for r in REGIME_OPTIONS) + "<label><input type='radio' name='regime' value=''>Non spécifié</label></fieldset>")
        if "ville" in missing:
            html.append(f"<label>Ville<input name='ville' value='{safe(loc.get('ville'))}'/></label>")
        if "code_postal" in missing:
            html.append(f"<label>Code Postal<input name='code_postal' value='{safe(loc.get('code_postal'))}'/></label>")
    html.append("<button class='btn primary' type='submit'>Enregistrer & Rechercher Offres</button></form>")
    return "".join(html)


@app.route("/profile", methods=["GET", "POST"])
def profile():
    sid = session.get("sid")
    text = STORE_TEXT.get(sid or "")
    if not text:
        return redirect(url_for("index"))
    profile_json = session.get("profile_json")
    if profile_json is None:
        profile_json = build_profile(text)
        session["profile_json"] = profile_json
    if request.method == "POST":
        updated = dict(profile_json)
        langues_raw = request.form.get("langues")
        if langues_raw:
            arr = []
            for chunk in langues_raw.split(";"):
                parts = chunk.split(":")
                if len(parts) >= 2:
                    arr.append({"langue": parts[0].strip().lower(), "niveau": parts[1].strip()})
                elif parts[0].strip():
                    arr.append({"langue": parts[0].strip().lower(), "niveau": None})
            if arr:
                updated["langues"] = arr
        etu_raw = request.form.get("etudes")
        if etu_raw:
            etudes = []
            for chunk in etu_raw.split(";"):
                parts = [p.strip() for p in chunk.split("|")]
                if len(parts) >= 4:
                    etudes.append({"diplome": parts[0], "domaine": parts[1] or None, "institution": parts[2] or None, "periode": parts[3] or None})
                elif parts and parts[0]:
                    etudes.append({"diplome": parts[0], "domaine": None, "institution": None, "periode": None})
            if etudes:
                updated["etudes"] = etudes
        contrat = request.form.get("type_contrat") or None
        if contrat:
            updated["type_contrat_recherche"] = contrat
        regime = request.form.get("regime") or None
        if regime:
            updated["regime_travail_recherche"] = regime or None
        loc = updated.get("localisation") or {"ville": None, "code_postal": None}
        ville = request.form.get("ville") or loc.get("ville")
        cp = request.form.get("code_postal") or loc.get("code_postal")
        loc["ville"] = ville or None
        loc["code_postal"] = cp or None
        updated["localisation"] = loc
        session["profile_json"] = updated
<<<<<<< Updated upstream

        # --- ORIGINAL: offers_df, info = search_offers(updated, limit=100)
        # --- UPDATED: add distance sorting using geo.add_distance_column
=======
        # compute employability after updates
        session["employability_prob"] = score_employability(updated)
>>>>>>> Stashed changes
        offers_df, info = search_offers(updated, limit=100)

        # Build user location string from profile localisation
        user_loc = None
        loc_profile = updated.get("localisation") or {}
        if loc_profile.get("ville"):
            user_loc = loc_profile.get("ville")
            if loc_profile.get("code_postal"):
                user_loc = f"{loc_profile.get('ville')}, {loc_profile.get('code_postal')}"

        # Candidate columns likely present in the offres dataset
        place_cols = ["Lieux de travail_clean", "Lieux de travail", "Lieux de travail_list", "lieux de travail", "commune"]

        try:
            # add_distance_column will geocode and sort; safe fallback if it fails
            offers_df = add_distance_column(offers_df, user_place=user_loc, place_columns=place_cols)
        except Exception:
            # keep original offers_df on error
            pass

        if sid:
            STORE_OFFERS[sid] = offers_df.to_dict(orient="records")
            STORE_ACCEPTED[sid] = []
        session["offers_info"] = info
        session["offer_index"] = 0
        return redirect(url_for("offers"))
    # compute employability on initial render
    session["employability_prob"] = session.get("employability_prob") or score_employability(profile_json)
    body = ""
    prob = session.get("employability_prob")
    if prob is not None:
        pct = f"{prob*100:.1f}%"
        body += f"<div class='offer' style='background:#f8fafc'><strong>Employabilité estimée:</strong> {pct}</div>"
    body += render_profile_form(profile_json)
    body += "<p><a class='btn neutral' href='/'>&larr; Retour</a></p>"
    return html_page("Profil", body)


@app.route("/offers", methods=["GET", "POST"])
def offers():
    sid = session.get("sid")
    offers = STORE_OFFERS.get(sid or "")
    if offers is None:
        return redirect(url_for("profile"))
    idx = session.get("offer_index", 0)
    accepted = STORE_ACCEPTED.get(sid or "", [])
    info = session.get("offers_info", {})
    if request.method == "POST":
        action = request.form.get("action")
        if action == "accept" and idx < len(offers):
            accepted.append(offers[idx])
        if action in {"accept", "reject", "skip"}:
            idx += 1
        session["offer_index"] = idx
        if sid:
            STORE_ACCEPTED[sid] = accepted
        return redirect(url_for("offers"))
    body = f"<h2>Offres filtrées</h2><p>Filtres appliqués: {', '.join(info.get('applied', [])) or 'aucun'} | Conservées: {info.get('kept')}</p>"
    if idx < len(offers):
        o = offers[idx]
        body += f"<div class='offer'><h3>{o.get('titreoffre')}</h3>"
        body += f"<p><strong>Employeur:</strong> {o.get('nomemployeur')}</p>"
        body += f"<p><strong>Régime:</strong> {o.get('regimetravail')} | <strong>Contrat:</strong> {o.get('typecontrat')}</p>"
        langs = o.get('langues') or []
        body += f"<p><strong>Langues:</strong> {', '.join(langs) if langs else 'Non spécifiées'}</p>"

        # === NEW: show computed distance if present ===
        dist = o.get("distance_km")
        if dist is not None:
            body += f"<p><strong>Distance:</strong> {dist} km</p>"
        else:
            body += f"<p><strong>Distance:</strong> N/A</p>"

        body += f"<p><a target='_blank' href='{o.get('url')}'>Lien offre</a></p>"
        body += "<form method='post'><button name='action' value='accept' class='btn primary'>Accepter</button>"
        body += "<button name='action' value='reject' class='btn danger'>Refuser</button>"
        body += "<button name='action' value='skip' class='btn neutral'>Passer</button></form></div>"
        body += f"<p>Offre {idx+1}/{len(offers)}</p>"
    else:
        body += "<p>Fin des offres.</p><p><a class='btn primary' href='/accepted'>Voir offres acceptées</a></p>"
    if accepted:
        body += f"<p>Acceptées: {len(accepted)} (<a href='/accepted'>voir</a>)</p>"
    body += "<p><a class='btn neutral' href='/profile'>&larr; Retour Profil</a></p>"
    return html_page("Offres", body)


@app.route("/accepted", methods=["GET"])
def accepted():
    sid = session.get("sid")
    accepted = STORE_ACCEPTED.get(sid or "", [])
    body = "<h2>Offres acceptées</h2>"
    if not accepted:
        body += "<p>Aucune encore.</p>"
    else:
        for o in accepted:
            body += f"<div class='offer accepted'><strong>{o.get('titreoffre')}</strong><br/>"\
                     f"{o.get('nomemployeur')} | {o.get('regimetravail')} | {o.get('typecontrat')}<br/>"\
                     f"<a target='_blank' href='{o.get('url')}'>Lien</a></div>"
        body += "<form action='/download' method='get'><button class='btn primary' type='submit'>Télécharger CSV</button></form>"
    body += "<p><a class='btn neutral' href='/offers'>&larr; Retour Offres</a></p>"
    return html_page("Acceptées", body)


@app.route("/download", methods=["GET"])
def download_csv():
    sid = session.get("sid")
    accepted = STORE_ACCEPTED.get(sid or "", [])
    output = io.StringIO()
    if accepted:
        fieldnames = sorted({k for o in accepted for k in o.keys()})
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for o in accepted:
            writer.writerow(o)
    resp = make_response(output.getvalue())
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=offres_acceptées.csv"
    return resp


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="127.0.0.1", port=port, debug=True)
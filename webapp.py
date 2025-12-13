import os
import io
import csv
import json
import secrets
import tempfile
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from flask import Flask, request, redirect, url_for, session, make_response
import requests

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    from zamzar import ZamzarClient
except Exception:
    ZamzarClient = None

from dotenv import load_dotenv
load_dotenv()
ZAMZAR_KEY = os.getenv("ZAMZAR_API_KEY")

# === Debug LLM ===
LLM_DEBUG = os.getenv("LLM_DEBUG", "0") == "1"
try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:
    BaseCallbackHandler = None

class LLMLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        if LLM_DEBUG:
            print("[LLM DEBUG] === Prompt envoyé à Mistral ===")
            if prompts:
                p = prompts[0]
                print(p if len(p) < 12000 else p[:12000] + "\n...[truncated]...")

    def on_llm_end(self, response, **kwargs):
        if LLM_DEBUG:
            print("[LLM DEBUG] === Réponse Mistral (analyse CV) ===")
            try:
                gen = response.generations[0][0]
                content = getattr(gen, "text", None) or getattr(getattr(gen, "message", None), "content", "")
            except Exception:
                content = str(response)
            print(content if len(content) < 12000 else content[:12000] + "\n...[truncated]...")


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
from geo import add_distance_column
from feature_engineering import enrich_features

INFERENCE_THRESHOLD = 0.79
APP_MODEL = joblib.load('models/hire_tabular_core.joblib')

def infer_score(features: dict):
    cols = APP_MODEL.feature_names_in_ if hasattr(APP_MODEL, "feature_names_in_") else APP_MODEL.get_booster().feature_names
    row = {k: features.get(k, np.nan) for k in cols}
    X = enrich_features(pd.DataFrame([row]))
    X = X.reindex(columns=cols, fill_value=np.nan)
    proba = float(APP_MODEL.predict_proba(X)[:, 1][0])
    return proba

def score_llm_profile_offer(profile, offer):
    import re
    prompt = (
        "Voici un profil de candidat:\n" + json.dumps(profile, ensure_ascii=False, indent=2) +
        "\n\nVoici une offre d'emploi:\n" + json.dumps(offer, ensure_ascii=False, indent=2) +
        "\n\nQUESTION: Sur une échelle de 0 à 1, donne un score de compatibilité entre ce profil et cette offre.\n"
        "LE SCORE DOIT ÊTRE UNIQUE, DÉCIMAL, ENTRE 0 ET 1, SANS AUCUNE AUTRE INFORMATION, NI JSON, NI TEXTE, NI COMMENTAIRE, JUSTE UN FLOAT SUR UNE LIGNE !"
    )
    try:
        response = call_mistral_fallback(prompt)
        response_text = str(response).strip()
        print("=== LLM Response reçu pour le score :", repr(response_text))
        for line in response_text.split("\n"):
            find_num = re.findall(r"\d+\.\d+|\d+", line)
            for val in find_num:
                v = float(val)
                if 0.0 < v < 1.0:
                    print("SCORING (première ligne float <1):", v)
                    return v
        print("SCORING : aucune valeur raisonnable trouvée dans", repr(response_text))
        return 0.0
    except Exception as e:
        print("Erreur scoring LLM:", e)
        return 0.0

ALLOWED_CONTRACTS = [
    "Intérimaire avec option sur durée indéterminée", "Durée indéterminée",
    "Intérimaire", "Durée d'éterminée", "Etudiant", "Remplacement",
    "Contrat collaboration indépendant", "Flexi-Jobs",
    "Journalier (occasionnel ou saisonnier)", "Salarié statutaire",
    "Nettement défini",
]
REGIME_OPTIONS = ["Temps plein", "Temps partiel"]

def create_app():
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    return app

app = create_app()
STORE_TEXT: dict[str, str] = {}
STORE_OFFERS: dict[str, list[dict]] = {}
STORE_ACCEPTED: dict[str, list[dict]] = {}

@app.before_request
def ensure_sid():
    if "sid" not in session:
        session["sid"] = secrets.token_urlsafe(16)

def extract_text_local(file_path):
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
                if combined.strip():
                    return combined
        except Exception: pass
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
    except Exception:
        pdfminer_extract_text = None
    if pdfminer_extract_text:
        try:
            txt = pdfminer_extract_text(file_path)
            if txt and txt.strip(): return txt
            return "(Extraction locale PDF: résultat vide)"
        except Exception as e:
            return f"Erreur PDF locale (pdfminer): {e}"
    return "(PyPDF2 non installé et pdfminer non installé)"

def extract_text_zamzar(file_path):
    if not (ZAMZAR_KEY and ZamzarClient): return ""
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
    except Exception as e: return f"Erreur Zamzar: {e}"

def build_profile(cv_text):
    local = {
        "langues": extract_langues(cv_text),
        "etudes": extract_etudes(cv_text),
        "localisation": extract_localisation(cv_text),
        **extract_pref_contrat(cv_text),
    }
    llm_json = {}
    try:
        if API_KEY:
            chain = build_chain()
            if chain:
                if BaseCallbackHandler and LLM_DEBUG:
                    raw = chain.invoke({"cv_text": cv_text}, config={"callbacks": [LLMLogger()]})
                else:
                    raw = chain.run({"cv_text": cv_text})
            else:
                raw = call_mistral_fallback(cv_text)
        else:
            raw = call_mistral_fallback(cv_text)

        if LLM_DEBUG:
            print("[LLM DEBUG] === Sortie brute LLM ===")
            print(str(raw)[:12000])

        llm_json = safe_json_extract(raw)

        # Sécuriser les 4 features avec defaults
        import math
        def as_num0(v): 
            try:
                if isinstance(v, bool): return 0
                if v is None: return 0
                return float(v)
            except Exception:
                return 0
        def as_nan(v):
            try:
                if isinstance(v, bool): return math.nan
                if v is None: return math.nan
                return float(v)
            except Exception:
                return math.nan

        llm_json["PreviousCompanies"] = as_num0(llm_json.get("PreviousCompanies"))
        llm_json["ExperienceYears"]   = as_num0(llm_json.get("ExperienceYears"))
        llm_json["EducationLevel"]    = as_num0(llm_json.get("EducationLevel"))
        llm_json["Age"]               = as_nan(llm_json.get("Age"))

        if LLM_DEBUG:
            print("[LLM DEBUG] === JSON LLM (features ML) ===")
            print(json.dumps({k: llm_json.get(k) for k in ["PreviousCompanies","ExperienceYears","EducationLevel","Age"]}, ensure_ascii=False, indent=2))

    except Exception:
        llm_json = {}

    # Avant fusion: log
    if LLM_DEBUG:
        print("[FUSE DEBUG] Avant fusion -> local:", {k: local.get(k) for k in ["PreviousCompanies","ExperienceYears","EducationLevel","Age"]})
        print("[FUSE DEBUG] Avant fusion -> llm_json:", {k: llm_json.get(k) for k in ["PreviousCompanies","ExperienceYears","EducationLevel","Age"]})

    fused = fuse(local, llm_json or {})

    # Après fusion: forcer que les 4 features gardent la valeur LLM (si dispo)
    for key in ["PreviousCompanies","ExperienceYears","EducationLevel","Age"]:
        if key in llm_json and llm_json[key] is not None:
            fused[key] = llm_json[key]

    # Cast final pour éviter strings
    import math
    def to_float_or(v, default):
        try:
            if isinstance(v, bool): return default
            return float(v)
        except Exception:
            return default
    fused["PreviousCompanies"] = to_float_or(fused.get("PreviousCompanies"), 0.0)
    fused["ExperienceYears"]   = to_float_or(fused.get("ExperienceYears"), 0.0)
    fused["EducationLevel"]    = to_float_or(fused.get("EducationLevel"), 0.0)
    # Age: autorise NaN si inconnu
    fused["Age"]               = to_float_or(fused.get("Age"), math.nan)

    # Debug: montrer les features envoyées au modèle
    try:
        print("[ML DEBUG] Features pour le modèle (après fusion):", {
            "PreviousCompanies": fused.get("PreviousCompanies"),
            "ExperienceYears": fused.get("ExperienceYears"),
            "EducationLevel": fused.get("EducationLevel"),
            "Age": fused.get("Age"),
        })
    except Exception:
        pass

    return fused

def html_page(title, body):
    return f"""<!DOCTYPE html>
<html lang='fr'>
<head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>{title} | JobMatch Pro</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@500;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', Arial, sans-serif;
            background: #f6f7fa;
            margin: 0;
            color: #232735;
        }}
        .navbar {{
            background: #232735;
            color: #fff;
            height: 58px;
            display: flex;
            align-items: center;
            padding: 0 36px;
            box-shadow: 0 2px 8px #21253c35;
        }}
        .navbar-brand {{
            font-weight: 700;
            font-size: 1.26em;
            letter-spacing: .7px;
            margin-right: 28px;
        }}
        .navbar nav {{
            display: flex;
            gap: 30px;
            font-size: 1em;
        }}
        .navbar nav a {{
            color: inherit;
            text-decoration: none;
            font-weight: 500;
            padding: 4px 0;
            transition:color .15s;
            border-bottom:2px solid transparent;
        }}
        .navbar nav a.active, .navbar nav a:hover {{
            color: #357EFE;
            border-bottom: 2px solid #357EFE;
        }}
        main {{
            max-width: 640px;
            margin: 40px auto;
            background: #fff;
            box-shadow: 0 2px 14px rgba(44,54,71,0.08);
            border-radius: 18px;
            padding: 44px 36px 28px 36px;
        }}
        h1,h2,h3 {{
            font-weight:700;
            color:#232735;
            margin-top:0;
        }}
        label {{
            font-weight:500;
            margin-top:13px; display:block;
            color:#42518a;
        }}
        input, select, textarea {{
            width:100%;padding:10px 12px;
            border-radius:6px;box-sizing:border-box;
            border:1.3px solid #e4e7f2;
            background:#f6f7fa;
            font-size:1.08em;
            margin:7px 0 16px 0;
            transition: border .14s;
        }}
        input:focus, select:focus, textarea:focus {{
            border-color: #357EFE;
            outline: none;
            background: #eef3fd;
        }}
        .btn {{
            appearance: none;
            border:none;
            outline:none;
            background: #357EFE;
            color:#fff;
            font-family:inherit;
            font-size:1.08em;
            font-weight:600;
            padding:11px 34px;
            border-radius:18px;
            margin:8px 12px 0 0;
            cursor:pointer;
            box-shadow:0 2px 12px #357efe1c;
            transition:background .15s, box-shadow .14s;
            letter-spacing:.03em;
            display:inline-block;
        }}
        .btn:hover {{
            background: #294fb9;
            box-shadow:0 6px 20px #357efe25;
        }}
        .btn.secondary {{
            background: #e7eaf4;
            color: #283c66;
        }}
        .btn.danger {{
            background:#e53e3e;color:#fff;
        }}
        .section-card {{
            background:#f9fbfd;border-radius:13px;
            padding:24px 18px;margin-bottom:26px;
            box-shadow:0 1px 7px #21253c0d;
            border:1px solid #eef3fd;
        }}
        pre {{
            background:#f6f7fa;
            border-radius:9px;
            padding:12px 13px;
            font-size:0.97em;
            box-shadow:0 1px 8px #23273513;
            border:1px solid #e4e7f2;
        }}
        fieldset{{ background:#f7f9fc;border-radius:7px;padding:15px 16px;border:1px solid #e4e7f2;margin-bottom:14px; }}
        .file-upload {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }}
        .file-upload input[type=file] {{
            display: none;
        }}
        .file-upload-label {{
            background: #357EFE;
            color: #fff;
            font-weight: 600;
            padding: 11px 24px;
            border-radius: 18px;
            cursor: pointer;
            transition: background 0.16s, box-shadow 0.13s;
            box-shadow: 0 2px 12px #357efe16;
            border: none;
            font-size: 1.05em;
            position: relative;
        }}
        .file-upload-label:hover {{
            background: #294fb9;
        }}
        .file-upload-fileinfo {{
            color: #232735;
            font-size: 0.99em;
            background: #eef3fd;
            padding: 7px 16px;
            border-radius: 11px;
            margin-left:2px;
        }}
        @media screen and (max-width:600px){{
            main{{padding:10px 2vw;}}
            .navbar{{padding:0 13px;}}
        }}
    </style>
    <script>
    function updateFileName(input) {{
        var info = document.getElementById('file-upload-fileinfo');
        if(input.files.length > 0) {{
            info.textContent = input.files[0].name;
        }} else {{
            info.textContent = "Aucun fichier choisi";
        }}
    }}
    </script>
</head>
<body>
    <header class="navbar">
        <span class="navbar-brand">JobMatch Pro</span>
        <nav>
            <a href='{url_for("index")}' {'class="active"' if title=="Accueil" else ''}>Accueil</a>
            <a href='{url_for("profile")}' {'class="active"' if title=="Profil" else ''}>Profil</a>
            <a href='{url_for("offers")}' {'class="active"' if title=="Offres" else ''}>Offres</a>
            <a href='{url_for("accepted")}' {'class="active"' if title=="Acceptées" else ''}>Acceptées</a>
            <a href='{url_for("quiz")}' {'class="active"' if title=="Quiz" else ''}>Se tester</a>
        </nav>
    </header>
    <main>
    {body}
    </main>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    body = """
    <h1>The Tinder of LinkedIn</h1>
    <div class="section-card">
      <p>Importez votre CV, puis profitez de la recherche intelligente d’offres.</p>
      <form action='/upload' method='post' enctype='multipart/form-data'>
        <div class="file-upload">
          <label class="file-upload-label" for="file-upload-input">Choisir un fichier PDF</label>
          <input id="file-upload-input" type="file" name="cv_pdf" accept="application/pdf" required onchange="updateFileName(this)">
          <span class="file-upload-fileinfo" id="file-upload-fileinfo">Aucun fichier choisi</span>
        </div>
        <button class='btn' type='submit'>Téléverser & Extraire</button>
      </form>
    </div>
    """
    sid = session.get("sid")
    if sid and STORE_TEXT.get(sid):
        body += '<div class="section-card" style="color:#357EFE">CV déjà importé. <a href="/profile">Aller au profil</a></div>'
    return html_page("Accueil", body)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("cv_pdf")
    if not file:
        return redirect(url_for("index"))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        file.save(tmp.name)
        # Close the handle before any reader/converter uses the file (Windows requirement)
        tmp.close()

        # Prefer Zamzar if configured, else local extraction
        text = extract_text_zamzar(tmp.name) or extract_text_local(tmp.name)

        sid = session.get("sid")
        if sid:
            STORE_TEXT[sid] = (text or "")[:200000]
            STORE_OFFERS.pop(sid, None)
            STORE_ACCEPTED.pop(sid, None)
        session.pop("profile_json", None)
        session.pop("offers_info", None)
        session.pop("offer_index", None)
        return redirect(url_for("profile"))
    finally:
        # Best-effort cleanup (ignore if another process still holds it)
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

def render_profile_form(profile):
    questions = session.get("llm_questions") or []
    missing = find_missing(profile) if not questions else []
    def safe(v): return "" if v is None else str(v)
    loc = profile.get("localisation") or {"ville": None, "code_postal": None}
    try:
        proba = infer_score(profile)
        score_html = f"<div class='section-card' style='margin-bottom:20px;'>Score d'employabilité prédit&nbsp;: <b>{proba*100:.1f}%</b></div>"
    except Exception as e:
        score_html = f"<div class='section-card' style='color:red;margin-bottom:12px;'>Erreur score modèle: {e}</div>"
    html = [
        score_html,
        "<h2>Profil extrait</h2>",
        f"<pre>{json.dumps(profile, ensure_ascii=False, indent=2)}</pre>"
    ]
    html.append("<h3>Compléter les informations manquantes</h3>")
    html.append("<form method='post' action='/profile'>")
    if questions:
        for q in questions:
            champ = (q or {}).get("champ")
            qtext = (q or {}).get("question") or champ
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
                html.append("<label>Type de contrat<select name='type_contrat'><option value=''>-- Non précisé --</option>" + "".join(f"<option>{c}</option>" for c in opts) + "</select></label>")
            elif champ == "regime_travail_recherche":
                opts = options or REGIME_OPTIONS
                html.append("<fieldset><legend>Régime de travail</legend>" + "".join(f"<label><input type='radio' name='regime' value='{r}'> {r}</label>" for r in opts) + "<label><input type='radio' name='regime' value=''>Autre</label></fieldset>")
            elif champ in ("ville", "code_postal"):
                html.append(f"<label>{qtext}<input name='{champ}' value='{safe(loc.get(champ))}'/></label>")
            else:
                html.append(f"<label>{qtext}<input name='{champ}'/></label>")
    else:
        if "langues" in missing:
            html.append("<label>Langues (francais:C2; anglais:B2)<input name='langues' placeholder='francais:C2; anglais:B2'/></label>")
        if "etudes" in missing:
            html.append("<label>Études (Diplôme|Domaine|Institution|Période;...)<textarea name='etudes' rows='3'></textarea></label>")
        if "type_contrat_recherche" in missing:
            html.append("<label>Type de contrat recherché<select name='type_contrat'><option value=''>-- Non précisé --</option>" + "".join(f"<option>{c}</option>" for c in ALLOWED_CONTRACTS) + "</select></label>")
        if "regime_travail_recherche" in missing:
            html.append("<fieldset><legend>Régime de travail</legend>" + "".join(f"<label><input type='radio' name='regime' value='{r}'> {r}</label>" for r in REGIME_OPTIONS) + "<label><input type='radio' name='regime' value=''>Autre</label></fieldset>")
        if "ville" in missing: html.append(f"<label>Ville<input name='ville' value='{safe(loc.get('ville'))}'/></label>")
        if "code_postal" in missing: html.append(f"<label>Code Postal<input name='code_postal' value='{safe(loc.get('code_postal'))}'/></label>")
    html.append("<button class='btn' type='submit'>Enregistrer & Rechercher Offres</button></form>")
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
            if arr: updated["langues"] = arr
        etu_raw = request.form.get("etudes")
        if etu_raw:
            etudes = []
            for chunk in etu_raw.split(";"):
                parts = [p.strip() for p in chunk.split("|")]
                if len(parts) >= 4:
                    etudes.append({"diplome": parts[0], "domaine": parts[1] or None, "institution": parts[2] or None, "periode": parts[3] or None})
                elif parts and parts[0]:
                    etudes.append({"diplome": parts[0], "domaine": None, "institution": None, "periode": None})
            if etudes: updated["etudes"] = etudes
        contrat = request.form.get("type_contrat") or None
        if contrat: updated["type_contrat_recherche"] = contrat
        regime = request.form.get("regime") or None
        if regime: updated["regime_travail_recherche"] = regime or None
        loc = updated.get("localisation") or {"ville": None, "code_postal": None}
        ville = request.form.get("ville") or loc.get("ville")
        cp = request.form.get("code_postal") or loc.get("code_postal")
        loc["ville"] = ville or None
        loc["code_postal"] = cp or None
        updated["localisation"] = loc
        session["profile_json"] = updated

        offers_df, info = search_offers(updated, limit=100)
        user_loc = None
        loc_profile = updated.get("localisation") or {}
        if loc_profile.get("ville"):
            user_loc = loc_profile.get("ville")
            if loc_profile.get("code_postal"):
                user_loc = f"{loc_profile.get('ville')}, {loc_profile.get('code_postal')}"
        place_cols = ["Lieux de travail_clean", "Lieux de travail", "Lieux de travail_list", "lieux de travail", "commune"]
        try:
            offers_df = add_distance_column(offers_df, user_place=user_loc, place_columns=place_cols)
        except Exception: pass

        # === SCORING LLM POUR CHAQUE OFFRE ===
        print("Scoring des offres via LLM (peut prendre du temps)...")
        offers_df["match_score"] = offers_df.apply(lambda row: score_llm_profile_offer(updated, dict(row)), axis=1)
        offers_df = offers_df.sort_values("match_score", ascending=False)

        if sid:
            STORE_OFFERS[sid] = offers_df.to_dict(orient="records")
            STORE_ACCEPTED[sid] = []
        session["offers_info"] = info
        session["offer_index"] = 0
        return redirect(url_for("offers"))
    body = render_profile_form(profile_json)
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
        if action in {"accept", "reject", "skip"}: idx += 1
        session["offer_index"] = idx
        if sid: STORE_ACCEPTED[sid] = accepted
        return redirect(url_for("offers"))
    body = f"<h2>Offres filtrées</h2><div class='section-card'><p>Filtres appliqués: <strong>{', '.join(info.get('applied', [])) or 'aucun'}</strong> | Conservées: <strong>{info.get('kept')}</strong></p></div>"
    if idx < len(offers):
        o = offers[idx]
        body += f"<div class='section-card'><h3>{o.get('titreoffre')}</h3>"
        body += f"<p><strong>Score matching LLM:</strong> {o.get('match_score'):.1%}</p>"
        body += f"<p><strong>Employeur:</strong> {o.get('nomemployeur')}</p>"
        body += f"<p><strong>Régime:</strong> {o.get('regimetravail')} | <strong>Contrat:</strong> {o.get('typecontrat')}</p>"
        langs = o.get('langues') or []
        body += f"<p><strong>Langues:</strong> {', '.join(langs) if langs else 'Non spécifiées'}</p>"
        dist = o.get("distance_km")
        if dist is not None:
            body += f"<p><strong>Distance:</strong> {dist} km</p>"
        else:
            body += f"<p><strong>Distance:</strong> N/A</p>"
        body += f"<p><a target='_blank' href='{o.get('url')}'>Lien offre</a></p>"
        body += "<form method='post'><button name='action' value='accept' class='btn'>Accepter</button>"
        body += "<button name='action' value='reject' class='btn danger'>Refuser</button>"
        body += "<button name='action' value='skip' class='btn secondary'>Passer</button></form></div>"
        body += f"<p>Offre {idx+1}/{len(offers)}</p>"
    else:
        body += "<div class='section-card'><p>Fin des offres.</p><p><a class='btn' href='/accepted'>Voir offres acceptées</a></p></div>"
    if accepted:
        body += f"<div class='section-card'><p>Acceptées: {len(accepted)} (<a href='/accepted'>voir</a>)</p></div>"
    body += "<p><a class='btn secondary' href='/profile'>Retour Profil</a></p>"
    return html_page("Offres", body)

@app.route("/accepted", methods=["GET"])
def accepted():
    sid = session.get("sid")
    accepted = STORE_ACCEPTED.get(sid or "", [])
    body = "<h2>Offres acceptées</h2>"
    if not accepted:
        body += "<div class='section-card'><p>Aucune encore.</p></div>"
    else:
        for o in accepted:
            body += f"<div class='section-card'><strong>{o.get('titreoffre')}</strong><br/>"\
                     f"{o.get('nomemployeur')} | {o.get('regimetravail')} | {o.get('typecontrat')}<br/>"\
                     f"<a target='_blank' href='{o.get('url')}'>Lien offre</a></div>"
        body += "<form action='/download' method='get'><button class='btn' type='submit'>Télécharger CSV</button></form>"
    body += "<p><a class='btn secondary' href='/offers'>Retour Offres</a></p>"
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

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    msg = ""
    if request.method == "POST":
        try:
            features = {}
            for field in request.form:
                features[field] = float(request.form[field])
            proba = infer_score(features)
            msg = f"<div class='section-card'>Votre score prédit d'embauche : <b>{100*proba:.1f}%</b></div>"
        except Exception as e:
            msg = f"<div class='section-card' style='color:red;'>Erreur : {e}</div>"
    body = """
        <h1>🎯 Simulation d'embauche (sans CV)</h1>
        <div class="section-card">
            <form method="post">
                <label>PreviousCompanies</label>
                <input type="number" step="1" name="PreviousCompanies" required>
                <label>ExperienceYears</label>
                <input type="number" step="0.1" name="ExperienceYears" required>
                <label>EducationLevel (1=Bac, 2=Bac+3, 3=Master, 4=PhD)</label>
                <input type="number" step="1" name="EducationLevel" required>
                <label>Age</label>
                <input type="number" step="1" name="Age" required>
                <button class="btn" type="submit">Voir mon score</button>
            </form>
        </div>
    """ + msg
    return html_page("Quiz", body)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="127.0.0.1", port=port, debug=True)
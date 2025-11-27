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
from geo import add_distance_column

ALLOWED_CONTRACTS = [
    "Intérimaire avec option sur durée indéterminée",
    "Durée indéterminée",
    "Intérimaire",
    "Durée déterminée",
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


@app.before_request
def ensure_sid():
    if "sid" not in session:
        session["sid"] = secrets.token_urlsafe(16)


def extract_text_local(file_path: str) -> str:
    if not PyPDF2:
        return "(PyPDF2 non installé)"
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(parts)
    except Exception as e:
        return f"Erreur PDF locale: {e}"


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


def html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'><title>{title}</title>
    <style>body{{font-family:Arial;margin:30px;max-width:900px}}input,select,textarea{{width:100%;margin:4px 0;padding:6px}}.offer{{border:1px solid #ccc;padding:12px;margin-bottom:14px;border-radius:6px}}.btn{{padding:8px 16px;margin:4px;border:0;border-radius:4px;cursor:pointer}}.primary{{background:#2563eb;color:#fff}}.danger{{background:#dc2626;color:#fff}}.neutral{{background:#6b7280;color:#fff}}nav a{{margin-right:12px}}.tag{{display:inline-block;background:#f1f5f9;padding:4px 8px;margin:2px;border-radius:4px;font-size:12px}}.accepted{{background:#16a34a;color:#fff}}</style></head><body>
    <nav><a href='{url_for('index')}'>Accueil</a><a href='{url_for('profile')}'>Profil</a><a href='{url_for('offers')}'>Offres</a><a href='{url_for('accepted')}'>Acceptées</a></nav><hr/>{body}</body></html>"""


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
                html.append("<label>Type de contrat<select name='type_contrat'><option value=''>-- Non précisé --</option>" + "".join(f"<option>{c}</option>" for c in opts) + "</select></label>")
            elif champ == "regime_travail_recherche":
                opts = options or REGIME_OPTIONS
                html.append("<fieldset><legend>Régime de travail</legend>" + "".join(f"<label><input type='radio' name='regime' value='{r}'> {r}</label>" for r in opts) + "<label><input type='radio' name='regime' value=''> Non précisé</label></fieldset>")
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
            html.append("<label>Type de contrat recherché<select name='type_contrat'><option value=''>-- Non précisé --</option>" + "".join(f"<option>{c}</option>" for c in ALLOWED_CONTRACTS) + "</select></label>")
        if "regime_travail_recherche" in missing:
            html.append("<fieldset><legend>Régime de travail</legend>" + "".join(f"<label><input type='radio' name='regime' value='{r}'> {r}</label>" for r in REGIME_OPTIONS) + "<label><input type='radio' name='regime' value=''> Non précisé</label></fieldset>")
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
        offers_df, info = search_offers(updated, limit=100)
        # Build user location string and sort by distance
        user_loc = loc.get("ville") or ""
        if loc.get("code_postal"):
            user_loc = f"{user_loc}, {loc.get('code_postal')}" if user_loc else loc.get("code_postal")
        if user_loc:
            offers_df = add_distance_column(
                offers_df,
                user_place=user_loc,
                place_columns=["lieuxtravaillocalite", "lieuxtravailcodepostal"],
            )
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
        # Display distance if available
        dist = o.get('distance_km')
        if dist is not None and not (isinstance(dist, float) and dist != dist):  # check for NaN
            body += f"<p><strong>Distance:</strong> {dist:.1f} km</p>"
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
# NOTE: below is the original page.py with minimal edits to integrate add_distance_column
import os
import json
import tempfile
from typing import Dict, Any, List

import streamlit as st

# Optional libs (Zamzar + PyPDF2) with graceful fallback
try:
    from zamzar import ZamzarClient  # type: ignore
except Exception:  # pragma: no cover
    ZamzarClient = None  # type: ignore

try:
    import PyPDF2  # type: ignore
except Exception:  # pragma: no cover
    PyPDF2 = None  # type: ignore

from dotenv import load_dotenv
load_dotenv()
ZAMZAR_KEY = os.getenv("ZAMZAR_API_KEY")

# Import local modules
from CV import read_cv, extract_langues, extract_etudes, extract_localisation, extract_pref_contrat, fuse, find_missing
from FOREM import search_offers

# === NEW import ===
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


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Local PDF text extraction using PyPDF2 (fallback if Zamzar absent)."""
    if not PyPDF2:
        return "(PyPDF2 non installé)"
    try:
        reader = PyPDF2.PdfReader(file_bytes)
        text_parts = []
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(text_parts)
    except Exception as e:
        return f"Erreur lecture PDF locale: {e}"


def convert_pdf_via_zamzar(temp_pdf_path: str) -> str:
    """Use Zamzar API to convert PDF to TXT if credentials are present."""
    if not (ZAMZAR_KEY and ZamzarClient):
        return ""
    try:
        client = ZamzarClient(ZAMZAR_KEY)
        job = client.convert(temp_pdf_path, "txt")
        # Store in temp dir then read
        out_dir = tempfile.mkdtemp(prefix="zamzar_")
        job.store(out_dir)
        # Find a .txt file
        for name in os.listdir(out_dir):
            if name.lower().endswith(".txt"):
                with open(os.path.join(out_dir, name), "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read()
                job.delete_all_files()
                return txt
        job.delete_all_files()
        return ""
    except Exception as e:
        return f"Erreur Zamzar: {e}"


def build_initial_profile(cv_text: str) -> Dict[str, Any]:
    local_data = {
        "langues": extract_langues(cv_text),
        "etudes": extract_etudes(cv_text),
        "localisation": extract_localisation(cv_text),
        **extract_pref_contrat(cv_text),
    }
    # Pas d'appel LLM ici (optionnel) -> fuse avec vide
    fused = fuse(local_data, {})
    return fused


def render_missing_inputs(profile: Dict[str, Any]) -> Dict[str, Any]:
    missing = find_missing(profile)
    st.subheader("Compléter les informations manquantes")
    updated = dict(profile)

    # Langues
    if "langues" in missing:
        st.markdown("**Langues** (format: langue:niveau séparé par ';')")
        lang_raw = st.text_input("Ajouter langues", placeholder="francais:C2; anglais:B2")
        if lang_raw:
            lang_list = []
            for chunk in lang_raw.split(";"):
                parts = chunk.split(":")
                if len(parts) >= 2:
                    lang_list.append({"langue": parts[0].strip().lower(), "niveau": parts[1].strip()})
                elif parts[0].strip():
                    lang_list.append({"langue": parts[0].strip().lower(), "niveau": None})
            if lang_list:
                updated["langues"] = lang_list

    # Etudes
    if "etudes" in missing:
        st.markdown("**Etudes** (format: Diplôme|Domaine|Institution|Période;…) ")
        etu_raw = st.text_area("Ajouter études", height=100)
        if etu_raw:
            objs = []
            for chunk in etu_raw.split(";"):
                parts = [p.strip() for p in chunk.split("|")]
                if len(parts) >= 4:
                    objs.append({"diplome": parts[0], "domaine": parts[1] or None, "institution": parts[2] or None, "periode": parts[3] or None})
                elif parts and parts[0]:
                    objs.append({"diplome": parts[0], "domaine": None, "institution": None, "periode": None})
            if objs:
                updated["etudes"] = objs

    # Type contrat
    if "type_contrat_recherche" in missing:
        st.markdown("**Type de contrat recherché**")
        updated["type_contrat_recherche"] = st.selectbox("Choisir type de contrat", options=["(Non précisé)"] + ALLOWED_CONTRACTS)
        if updated["type_contrat_recherche"] == "(Non précisé)":
            updated["type_contrat_recherche"] = None

    # Régime travail
    if "regime_travail_recherche" in missing:
        st.markdown("**Régime de travail**")
        regime = st.radio("Choisir régime", REGIME_OPTIONS + ["(Non précisé)"])
        updated["regime_travail_recherche"] = None if regime == "(Non précisé)" else regime

    # Localisation
    loc = updated.get("localisation") or {"ville": None, "code_postal": None}
    if "ville" in missing:
        loc["ville"] = st.text_input("Ville", value=loc.get("ville") or "") or None
    if "code_postal" in missing:
        loc["code_postal"] = st.text_input("Code postal", value=loc.get("code_postal") or "") or None
    updated["localisation"] = loc

    return updated


def display_offer(row) -> None:
    st.markdown(f"### {row.get('titreoffre')}")
    st.write(f"Employeur: {row.get('nomemployeur')}")
    st.write(f"Régime: {row.get('regimetravail')} | Contrat: {row.get('typecontrat')}")
    langs = row.get("langues") or []
    st.write(f"Langues requises: {', '.join(langs) if langs else 'Non spécifiées'}")
    # === NEW: display distance when present ===
    dist = row.get("distance_km")
    if dist is not None:
        st.write(f"Distance: {dist} km")
    else:
        st.write("Distance: N/A")
    st.write(f"URL: {row.get('url')}")


def main():
    st.title("Assistant Emploi CV → Offres FOREM")
    st.caption("Upload ton CV PDF, complète les infos manquantes et swipe les offres.")

    if "accepted_offers" not in st.session_state:
        st.session_state.accepted_offers = []
    if "offers" not in st.session_state:
        st.session_state.offers = None
    if "offer_index" not in st.session_state:
        st.session_state.offer_index = 0
    if "profile" not in st.session_state:
        st.session_state.profile = None

    pdf_file = st.file_uploader("Téléverser CV (PDF)", type=["pdf"], accept_multiple_files=False)
    if pdf_file is not None and st.session_state.profile is None:
        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            temp_path = tmp.name
        # Try Zamzar first
        txt = convert_pdf_via_zamzar(temp_path)
        if not txt:
            # Fallback local extraction
            with open(temp_path, "rb") as f:
                txt = extract_text_from_pdf(f)
        os.unlink(temp_path)

        st.subheader("Aperçu du texte extrait")
        st.text(txt[:1500] + ("..." if len(txt) > 1500 else ""))

        profile = build_initial_profile(txt)
        st.session_state.profile = profile
        st.success("Profil initial extrait.")

    profile = st.session_state.profile
    if profile is not None and st.session_state.offers is None:
        st.subheader("Profil détecté (heuristiques)")
        st.json(profile)
        updated = render_missing_inputs(profile)
        if st.button("Construire profil complet & Rechercher"):
            st.session_state.profile = updated
            offers_df, info = search_offers(updated, limit=100)

            # Build user location
            user_loc = None
            locp = updated.get("localisation") or {}
            if locp.get("ville"):
                user_loc = locp.get("ville")
                if locp.get("code_postal"):
                    user_loc = f"{locp.get('ville')}, {locp.get('code_postal')}"

            place_cols = ["Lieux de travail_clean", "Lieux de travail", "Lieux de travail_list", "lieux de travail", "commune"]
            try:
                offers_df = add_distance_column(offers_df, user_place=user_loc, place_columns=place_cols)
            except Exception:
                # fallback: keep original offers_df
                pass

            st.session_state.offers = offers_df.reset_index(drop=True)
            st.session_state.search_info = info
            st.session_state.offer_index = 0
            st.success(f"Offres trouvées: {info['kept']} (filtres: {', '.join(info['applied']) or 'aucun'})")

    if st.session_state.offers is not None:
        df = st.session_state.offers
        idx = st.session_state.offer_index
        total = len(df)
        st.subheader(f"Offre {idx+1}/{total}")
        if idx < total:
            row = df.iloc[idx]
            display_offer(row)
            col1, col2, col3 = st.columns(3)
            if col1.button("👍 Accepter", key=f"accept_{idx}"):
                st.session_state.accepted_offers.append(row.to_dict())
                st.session_state.offer_index += 1
            if col2.button("👎 Refuser", key=f"reject_{idx}"):
                st.session_state.offer_index += 1
            if col3.button("⏭ Passer", key=f"skip_{idx}"):
                st.session_state.offer_index += 1
        else:
            st.info("Fin des offres.")

        if st.session_state.accepted_offers:
            st.subheader("Offres acceptées")
            acc_df = json.loads(json.dumps(st.session_state.accepted_offers))  # safe copy
            st.table(acc_df)
            if st.button("Télécharger CSV"):
                import pandas as pd
                csv_data = pd.DataFrame(st.session_state.accepted_offers).to_csv(index=False)
                st.download_button("Télécharger", csv_data, file_name="offres_choisies.csv", mime="text/csv")

    st.markdown("---")
    st.caption("Astuce: relance avec une autre version du CV pour ajuster le profil.")


if __name__ == "__main__":
    # Note: Pour exécuter: `streamlit run page.py`
    main()
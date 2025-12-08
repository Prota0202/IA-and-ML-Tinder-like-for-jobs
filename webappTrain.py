import os
import io
import csv
import pandas as pd
import json
import secrets
import tempfile
import joblib
import numpy as np
from typing import Dict, Any
from flask import Flask, request, redirect, url_for, session, make_response, render_template_string

from feature_engineering import enrich_features

INFERENCE_THRESHOLD = 0.79  # Met ton threshold de précision custom ici !

def load_model():
    return joblib.load('models/hire_tabular_core.joblib')

def infer_single(app_model, features: dict):
    # DataFrame à 1 ligne ; s'adapte à la présence de features manquantes
    cols = app_model.feature_names_in_ if hasattr(app_model, "feature_names_in_") else app_model.get_booster().feature_names
    row = {k: features.get(k, np.nan) for k in cols}
    X = enrich_features(pd.DataFrame([row]))
    X = X.reindex(columns=cols, fill_value=np.nan)
    proba = float(app_model.predict_proba(X)[:,1][0])
    pred = int(proba >= INFERENCE_THRESHOLD)
    return proba, pred

def create_app():
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    app_model = load_model()

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
            background: #f4f6f8;
            color: #222;
            margin: 0;
        }}
        .navbar {{
            background: #232735;
            color: #fff;
            height: 52px;
            display: flex;
            align-items: center;
            padding: 0 24px;
            box-shadow: 0 1px 5px #21253c15;
        }}
        .navbar-brand {{
            font-weight: 600;
            font-size: 1.13em;
            margin-right: 18px;
            letter-spacing:0.3px;
        }}
        .navbar nav {{
            display: flex;
            gap: 22px;
            font-size: 0.98em;
        }}
        .navbar nav a {{
            color: inherit;
            text-decoration: none;
            font-weight: 500;
            padding: 2px 0;
            border-bottom:2px solid transparent;
            transition:color .13s;
        }}
        .navbar nav a.active, .navbar nav a:hover {{
            color: #357EFE;
            border-bottom: 2px solid #357EFE;
        }}
        main {{
            max-width: 470px;
            margin: 30px auto;
            background: #fff;
            box-shadow: 0 2px 10px rgba(44,54,71,0.07);
            border-radius: 13px;
            padding: 30px 24px 22px 24px;
        }}
        h1,h2,h3 {{
            font-weight:600;
            color: #232735;
            margin:17px 0 14px 0;
        }}
        label {{
            font-weight:500;
            margin-top:10px; display:block;
            color:#42518a;
        }}
        input, select, textarea {{
            width:100%;padding:8px 11px;
            border-radius:5px;box-sizing:border-box;
            border:1px solid #e4e7f2;
            background:#f7f9fa;
            font-size:1em;
            margin:5px 0 13px 0;
            transition: border .11s;
        }}
        input:focus, select:focus, textarea:focus {{
            border-color: #357EFE;
            outline: none;
            background: #eef3fd;
        }}
        .btn {{
            border:none;
            background: #357EFE;
            color:#fff;
            font-family:inherit;
            font-size:0.99em;
            font-weight:500;
            padding:8px 22px;
            border-radius:12px;
            margin:6px 10px 0 0;
            cursor:pointer;
            transition:background .12s;
        }}
        .btn:hover {{
            background: #284bb4;
        }}
        .btn.secondary {{
            background: #e7eaf4;
            color: #283c66;
        }}
        .btn.danger {{
            background:#e53e3e;color:#fff;
        }}
        .section-card {{
            background:#f7f9fa;
            border-radius:8px;
            padding:17px 11px;
            margin-bottom:13px;
            box-shadow:0 1px 4px #21253c09;
            border:1px solid #edf2fa;
        }}
        pre {{
            background:#f1f3f6;
            border-radius:6px;
            padding:10px 10px;
            font-size:0.95em;
            border:1px solid #e4e7f2;
            margin-bottom:10px;
        }}
        fieldset{{ background:#f7f9fa;border-radius:7px;padding:11px 10px;border:1px solid #e4e7f2;margin-bottom:9px; }}
        .file-upload {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }}
        .file-upload input[type=file] {{ display: none; }}
        .file-upload-label {{
            background: #357EFE;color: #fff;
            font-weight: 500;
            padding: 8px 14px;
            border-radius: 12px;
            cursor: pointer;
            border: none;
            font-size: 0.99em;
        }}
        .file-upload-label:hover {{ background: #284bb4; }}
        .file-upload-fileinfo {{
            color: #232735;
            font-size: 0.93em;
            background: #eef3fd;
            padding: 4px 10px;
            border-radius: 9px;
            margin-left:2px;
        }}
        @media screen and (max-width:600px){{
            main{{padding:7px 2vw;}}
            .navbar{{padding:0 5px;}}
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
            <!-- Ajoute les autres onglets si nécessaires -->
        </nav>
    </header>
    <main>
    {body}
    </main>
</body>
</html>
"""
    @app.route("/", methods=["GET", "POST"])
    def index():
        proba = None
        pred = None
        msg = ""
        if request.method == "POST":
            try:
                # Simule la récupération des features depuis un formulaire ou des valeurs
                # Ici pour l'exemple on imagine un form feature_name=valeur
                features = {}
                for field in request.form:
                    features[field] = float(request.form[field])
                proba, pred = infer_single(app_model, features)
                msg = f"<div class='section-card'>Probabilité d'embauche prédite : <b>{proba:.2%}</b><br>Decision : <b>{'Oui' if pred else 'Non'}</b> (seuil={INFERENCE_THRESHOLD:.2f})</div>"
            except Exception as e:
                msg = f"<div class='section-card' style='color:red;'>Erreur : {e}</div>"

        body = """
        <h1>Scorer une candidature</h1>
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
                <!-- Ajoute ici d'autres features importantes selon enrich_features -->
                <button class="btn" type="submit">Evaluer</button>
            </form>
        </div>
        """ + msg
        return html_page("Accueil", body)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from webapp import score_llm_profile_offer, infer_score

# Load benchmark
df = pd.read_csv("benchmark/labels.csv")
with open("benchmark/test_cvs.json") as f:
    all_cvs = {cv['id']: cv for cv in json.load(f)}
with open("benchmark/test_jobs.json") as f:
    all_jobs = {job['id']: job for job in json.load(f)}

# Baseline: 0.5 if same city or any common language, else 0
def score_baseline(profile, offer):
    score = 0.0
    try:
        if profile['localisation']['ville'].lower() == offer['localisation'].lower():
            score += 0.5
        profil_langs = set(l['langue'] for l in profile.get('langues', []))
        offer_langs = set(offer.get('langues', []))
        if profil_langs & offer_langs:
            score += 0.5
    except:
        pass
    return min(score, 1.0)

# Tabular: mapping profile to features (adapte si tu as une vraie fonction de mapping)
def profile_to_tabular_features(profile: dict) -> dict:
    import numpy as np
    feats = {}
    # Simpliste ici: à adapter selon tes extraction réelles!
    etud = profile.get("etudes")
    if etud and isinstance(etud, list) and etud:
        highest = etud[0].get("diplome", "").lower()
        if "phd" in highest or "doctorat" in highest:
            feats["EducationLevel"] = 4
        elif "master" in highest:
            feats["EducationLevel"] = 3
        elif "licence" in highest or "bachelor" in highest:
            feats["EducationLevel"] = 2
        elif "bac" in highest:
            feats["EducationLevel"] = 1
        else:
            feats["EducationLevel"] = np.nan
    else:
        feats["EducationLevel"] = np.nan
    feats["Age"] = np.nan # Pas dans le dataset latent !
    feats["PreviousCompanies"] = np.nan
    feats["ExperienceYears"] = np.nan
    return feats

def score_tabular(profile, offer=None):
    feats = profile_to_tabular_features(profile)
    try:
        return float(infer_score(feats))
    except Exception as e:
        print("Tabular scoring failed:", e)
        return 0.0

scorers = {
    "baseline" : score_baseline,
    "tabular"  : score_tabular,
    "LLM"      : score_llm_profile_offer,
}

results = {}

for name, scorer in scorers.items():
    print(f"\n>> {name.upper()} scoring in progress...")
    y_pred, y_true = [], []
    for _, row in df.iterrows():
        cv = all_cvs[row['id_cv']]
        job = all_jobs[row['id_job']]
        if name == "tabular":
            s = scorer(cv)
        else:
            s = scorer(cv, job)
        y_pred.append(s)
        y_true.append(row['ground_truth'])
    results[name] = (y_true, y_pred)
    acc = accuracy_score(y_true, [int(x > 0.5) for x in y_pred])
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = "N/A"
    print(f"{name} - Accuracy: {acc:.3f}, AUC: {auc}")
    # Pour ROC
    if hasattr(auc, "__float__"):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbes ROC - Matching CV/Offres")
plt.legend()
plt.tight_layout()
plt.savefig("benchmark/roc_comparison.png")
print("\nCourbe ROC sauvegardée -> benchmark/roc_comparison.png")

# Affiche les détails (tableau)
table = pd.DataFrame({
    "id_cv": df["id_cv"], "id_job": df["id_job"], "label": df["ground_truth"],
    "baseline": results["baseline"][1],
    "tabular":  results["tabular"][1],
    "LLM":      results["LLM"][1],
})
print("\nTableau comparatif :\n", table)
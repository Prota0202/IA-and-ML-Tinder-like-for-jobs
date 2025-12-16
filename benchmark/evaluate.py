import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from webapp import score_llm_profile_offer, batch_score_offers, infer_score

# === Mode de scoring ===
# Basculer ici ou passer --batch en argument CLI
USE_BATCH = "--batch" in sys.argv
print(f"\n[BENCHMARK] Mode de scoring: {'BATCH (Green AI)' if USE_BATCH else 'NORMAL (ligne par ligne)'}")
print(f"             Hint: lancez avec 'python benchmark/evaluate.py --batch' pour comparer\n")

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
    feats["Age"] = profile.get("age", 30) 
    feats["PreviousCompanies"] = profile.get("previous_companies", 1)
    feats["ExperienceYears"] = profile.get("experience_years", 2)

    feats["InterviewScore"] = 5  # Score neutre
    feats["SkillScore"] = 5      # Score neutre
    feats["PersonalityScore"] = 5 # Score neutre
    feats["RecruitmentStrategy"] = 1 
    feats["DistanceFromCompany"] = 10 # Distance arbitraire moyenne
    feats["Gender"] = 0 # Ou aléatoire si tu veux
    return feats

def score_tabular(profile, offer=None):
    feats = profile_to_tabular_features(profile)
    try:
        return float(infer_score(feats))
    except Exception as e:
        print("Tabular scoring failed:", e)
        return 0.0

def score_with_mode_batch(cv_list, use_batch=False):
    """
    Score une liste de (cv, job) tuples.
    Si use_batch=True, groupe les offres par batch_size=10 pour vraiment batcher.
    Retourne liste de scores dans le même ordre.
    """
    if not use_batch:
        # Mode normal: score chaque offre individuellement
        return [score_llm_profile_offer(cv, job) for cv, job in cv_list]
    
    # Mode BATCH: grouper les offres par CV et les scorer par batch de taille variable
    scores = [None] * len(cv_list)
    batch_size = 10 
    
    # Grouper par CV
    from collections import defaultdict
    by_cv = defaultdict(list)
    for idx, (cv, job) in enumerate(cv_list):
        cv_id = cv['id']
        by_cv[cv_id].append((idx, cv, job))
    
    # Pour chaque CV, batcher les offres
    for cv_id, cv_jobs in by_cv.items():
        cv = cv_jobs[0][1]  # Get the CV object
        
        # Créer un DataFrame avec toutes les offres de ce CV
        jobs_for_df = [job for _, _, job in cv_jobs]
        jobs_df = pd.DataFrame(jobs_for_df)
        
        # Score par batch de 5 offres
        batch_scores = batch_score_offers(cv, jobs_df, batch_size=batch_size)
        
        # Remplacer dans le bon ordre
        for i, (idx, _, _) in enumerate(cv_jobs):
            scores[idx] = batch_scores[i]
    
    return scores

scorers = {
    "baseline" : score_baseline,
    #"tabular"  : score_tabular,
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
        raw_val = float(row['ground_truth'])
        y_true.append(1 if raw_val >= 0.5 else 0)
    results[name] = (y_true, y_pred)

# LLM scoring (batch ou normal)
print(f"\n>> LLM scoring in progress...")
# Préparer liste de (cv, job)
cv_job_list = []
y_true_llm = []
for _, row in df.iterrows():
    cv = all_cvs[row['id_cv']]
    job = all_jobs[row['id_job']]
    cv_job_list.append((cv, job))
    raw_val = float(row['ground_truth'])
    y_true_llm.append(1 if raw_val >= 0.5 else 0)

# Score avec batching
llm_scores = score_with_mode_batch(cv_job_list, use_batch=USE_BATCH)
results["LLM"] = (y_true_llm, llm_scores)

# Afficher résultats pour tous les scorers
for name in results.keys():
    y_true, y_pred = results[name]
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
    #"tabular":  results["tabular"][1],
    "LLM":      results["LLM"][1],
})
print("\nTableau comparatif :\n", table)
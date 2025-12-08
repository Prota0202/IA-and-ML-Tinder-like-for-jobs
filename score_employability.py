import argparse
import os
import re
from typing import Dict, List

try:
    import joblib  # preferred
except Exception:
    joblib = None
    import pickle
import numpy as np
import pandas as pd


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("-", "_").replace(" ", "_") for c in df.columns]
    return df


def _extract_features_from_cv(cv_text: str) -> Dict[str, float]:
    text = cv_text.lower()
    feats: Dict[str, float] = {}

    # ExperienceYears: detect explicit patterns and French date ranges
    years = []
    for m in re.finditer(r"(\b\d{1,2})\s*(\+)?\s*(years|year|ans|an)\b", text):
        try:
            years.append(int(m.group(1)))
        except Exception:
            pass
    # French month ranges like "Janvier 2022 – Présent" or "Septembre 2020 – Décembre 2021"
    MONTHS = {
        'janvier': 1, 'février': 2, 'fevrier': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6,
        'juillet': 7, 'août': 8, 'aout': 8, 'septembre': 9, 'octobre': 10, 'novembre': 11,
        'décembre': 12, 'decembre': 12
    }
    date_pat = re.compile(r"(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|d[ée]cembre)\s+(\d{4})\s*[–-]\s*(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|d[ée]cembre|pr[ée]sent|present)\s*(\d{4})?", re.IGNORECASE)
    exp_months = 0
    from datetime import datetime
    now = datetime.now()
    for m in date_pat.finditer(text):
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
        exp_months += min(months, 10 * 12)  # cap to 10y per role
    # Also support simple year ranges like 2018-2022
    for m in re.finditer(r"(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2})", text):
        try:
            y1, y2 = int(m.group(1)), int(m.group(2))
            if y2 > y1:
                exp_months += min((y2 - y1) * 12, 10 * 12)
        except Exception:
            pass
    exp_years = (exp_months / 12.0) if exp_months > 0 else (max(years) if years else np.nan)
    feats["ExperienceYears"] = float(exp_years) if exp_years is not None else np.nan

    # PreviousCompanies: count role headers or company keywords under experience section
    company_patterns = ["company", "entreprise", "employeur", "société", "societe", "firm", "startup"]
    companies_est = 0
    for kw in company_patterns:
        companies_est += len(re.findall(rf"\b{kw}\b", text))
    # crude role header detection: lines followed by a date
    role_headers = re.findall(r"\n([a-z][^\n]{0,60})\n(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|d[ée]cembre)\s+\d{4}", text, flags=re.IGNORECASE)
    companies_est = max(companies_est, len(role_headers))
    feats["PreviousCompanies"] = float(min(companies_est, 10)) if companies_est > 0 else np.nan

    # EducationLevel: map keywords to ordinal levels
    # 0: unknown, 1: high school, 2: bachelor, 3: master, 4: phd
    edu_level = np.nan
    if re.search(r"phd|doctorat|doctorate", text):
        edu_level = 4
    elif re.search(r"master|msc|maîtrise|m.sc|m.s", text):
        edu_level = 3
    elif re.search(r"bachelor|licence|b.sc|b.s|ba|beng", text):
        edu_level = 2
    elif re.search(r"high\s*school|lycée|lycee|baccalauréat|bac", text):
        edu_level = 1
    feats["EducationLevel"] = float(edu_level) if not np.isnan(edu_level) else np.nan

    # Age: attempt detection (very rough). If birth year present like 1995, estimate.
    age = np.nan
    now_year = 2025
    for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", text):
        year = int(m.group(1))
        if 1950 <= year <= 2010:
            est_age = now_year - year
            if 15 <= est_age <= 80:
                age = est_age
                break
    feats["Age"] = float(age) if not np.isnan(age) else np.nan

    # Only use core features; avoid optional fields that are usually missing

    return feats


def _load_bundle(model_path: str):
    if joblib is not None:
        return joblib.load(model_path)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def score_employability(model_path: str, cv_text: str) -> Dict[str, float]:
    bundle = _load_bundle(model_path)
    pipe = bundle["pipeline"]
    features: List[str] = bundle["features"]
    feats_extracted = _extract_features_from_cv(cv_text)

    # Build single-row DataFrame with required feature columns
    row = {c: feats_extracted.get(c, np.nan) for c in features}
    X = pd.DataFrame([row])
    proba = float(pipe.predict_proba(X)[:, 1][0])
    return {"employability_probability": proba, "features_used": len(features), "missing_features_filled": int(sum(pd.isna(list(row.values()))))}


def main():
    parser = argparse.ArgumentParser(description="Estimate employability probability from a CV text using the tabular model")
    parser.add_argument("--model", required=True, help="Path to trained tabular model bundle (.joblib)")
    parser.add_argument("--cv-file", required=True, help="Path to CV text file")
    parser.add_argument("--show-features", action="store_true", help="Print the feature vector used for scoring")
    args = parser.parse_args()

    cv_text = _read_text(args.cv_file)
    bundle = _load_bundle(args.model)
    features: List[str] = bundle["features"]

    result = score_employability(args.model, cv_text)
    print("\nEmployabilité (proba d'embauche sans job spécifique):")
    print(f"- probability: {result['employability_probability']:.3f}")
    print(f"- features_used: {result['features_used']}")
    print(f"- missing_features_filled_by_imputer: {result['missing_features_filled']}")

    if args.show_features:
        feats_extracted = _extract_features_from_cv(cv_text)
        # Only show core features to avoid noisy optional fields
        core_keys = ["Age", "EducationLevel", "ExperienceYears", "PreviousCompanies"]
        print("\nFeature vector (core, before imputation):")
        for k in core_keys:
            print(f"- {k}: {feats_extracted.get(k, np.nan)}")


if __name__ == "__main__":
    main()

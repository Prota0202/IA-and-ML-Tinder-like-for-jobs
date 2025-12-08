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

    # ExperienceYears: look for patterns like "X years", "X ans", "X+ years"
    years = []
    for m in re.finditer(r"(\b\d{1,2})\s*(\+)?\s*(years|year|ans|an)\b", text):
        try:
            years.append(int(m.group(1)))
        except Exception:
            pass
    # Also sum durations from date ranges like 2018-2022
    ranges = []
    for m in re.finditer(r"(20\d{2})\s*[-–]\s*(20\d{2})", text):
        try:
            ranges.append(int(m.group(2)) - int(m.group(1)))
        except Exception:
            pass
    exp_years = max(years) if years else (sum(r for r in ranges if r > 0) if ranges else np.nan)
    feats["ExperienceYears"] = float(exp_years) if exp_years is not None else np.nan

    # PreviousCompanies: approximate by counting occurrences of keywords
    company_patterns = ["company", "entreprise", "employer", "société", "societe", "firm", "startup"]
    companies_est = 0
    for kw in company_patterns:
        companies_est += len(re.findall(rf"\b{kw}\b", text))
    # Cap to reasonable range
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

    # Optional numeric features present in tabular dataset; leave NaN so imputer fills median
    feats["DistanceFromCompany"] = np.nan
    feats["InterviewScore"] = np.nan
    feats["SkillScore"] = np.nan
    feats["PersonalityScore"] = np.nan

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
        vector = {c: feats_extracted.get(c, np.nan) for c in features}
        print("\nFeature vector (before imputation):")
        for k, v in vector.items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()

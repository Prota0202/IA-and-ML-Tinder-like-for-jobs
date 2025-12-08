import argparse
import os
import pandas as pd
import joblib


def _read_csv_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("-", "_").replace(" ", "_") for c in df.columns]
    return df


def _infer_job_text_columns(df: pd.DataFrame, job_text_col: str | None) -> list[str]:
    if job_text_col and job_text_col in df.columns:
        return [job_text_col]
    candidates = [
        "job_description",
        "description",
        "job_text",
        # French fallbacks
        "intitulé_de_l'offre_clean",
        "intitulé_de_l'offre",
        "intitule_de_l'offre_clean",
        "intitule_de_l'offre",
        "metier_clean",
        "metier",
        "secteur_d'activité_clean",
        "secteur_d'activité",
    ]
    lower = {c.lower(): c for c in df.columns}
    found: list[str] = []
    for k in candidates:
        if k in lower:
            found.append(lower[k])
    if not found:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if obj_cols:
            found = [obj_cols[0]]
    return found[:3]


def score_cv_against_jobs(model_path: str, cv_text: str, jobs_csv: str, job_text_col: str | None = None, top_k: int = 10, normalize_per_query: bool = False):
    pipeline = joblib.load(model_path)
    jobs_df = _standardize_columns(_read_csv_auto(jobs_csv))
    cols = _infer_job_text_columns(jobs_df, job_text_col)

    # Build job text
    if len(cols) > 1:
        job_text = jobs_df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    else:
        job_text = jobs_df[cols[0]].fillna("").astype(str)

    combined = pd.Series(cv_text, index=job_text.index) + " [SEP] " + job_text

    # Probability or decision-function fallback
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(combined)[:, 1]
    else:
        scores = pipeline.decision_function(combined)
        s_min, s_max = scores.min(), scores.max()
        probs = (scores - s_min) / (s_max - s_min + 1e-9)

    # Keep raw stats before any normalization
    raw_min, raw_max = float(probs.min()), float(probs.max())

    # Optional per-query normalization to mitigate uniformly high scores
    if normalize_per_query:
        scale = (raw_max - raw_min) if (raw_max - raw_min) > 1e-9 else 1.0
        probs = (probs - raw_min) / scale

    out = jobs_df.copy()
    out["proba_select"] = probs

    # Drop exact duplicate job titles to reduce repeated rows in preview
    title_cols = [c for c in out.columns if c.lower().startswith("intitul") or c.lower().startswith("metier")]
    if title_cols:
        out = out.drop_duplicates(subset=title_cols)
    out = out.sort_values("proba_select", ascending=False)

    # Aggregate summaries for a single CV
    top = out.head(top_k)
    best = float(top["proba_select"].max()) if not top.empty else 0.0
    mean_top = float(top["proba_select"].mean()) if not top.empty else 0.0
    # Simple credibility flag: if mean of top-K is extremely high and dispersion is tiny
    std_top = float(top["proba_select"].std()) if len(top) > 1 else 0.0
    suspicious = (mean_top >= 0.9 and std_top < 0.02)

    # Also compute raw top-K stats (using raw distribution bounds)
    # Approximate raw top-K mean via inverse min-max if normalized
    raw_best = best if not normalize_per_query else (best * (raw_max - raw_min) + raw_min)
    raw_mean_top = mean_top if not normalize_per_query else (mean_top * (raw_max - raw_min) + raw_min)

    summary = {
        "best_job_probability": best,
        "mean_top{}_probability".format(top_k): mean_top,
        "std_top{}_probability".format(top_k): std_top,
        "suspicious_uniform_scores": suspicious,
        "normalized": normalize_per_query,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_best_job_probability": raw_best,
        "raw_mean_top{}_probability".format(top_k): raw_mean_top,
    }
    return out, summary


def main():
    parser = argparse.ArgumentParser(description="Score a single CV against jobs and return probabilities")
    parser.add_argument("--model", required=True, help="Path to trained text model (.joblib)")
    parser.add_argument("--cv-file", required=True, help="Path to CV text file")
    parser.add_argument("--jobs", required=True, help="Jobs CSV to compare against")
    parser.add_argument("--job-text-col", default=None, help="Specific job text column name if desired")
    parser.add_argument("--top-k", type=int, default=10, help="Top K jobs to summarize")
    parser.add_argument("--normalize-per-query", action="store_true", help="Rescale scores across jobs for this CV (min-max) — emphasizes ranking, not absolute proba")
    parser.add_argument("--show-raw", action="store_true", help="Show raw probability stats alongside normalized summary")
    parser.add_argument("--out", default=None, help="Optional output CSV for ranked jobs")
    args = parser.parse_args()

    with open(args.cv_file, "r", encoding="utf-8", errors="ignore") as f:
        cv_text = f.read()

    ranked, summary = score_cv_against_jobs(
        args.model, cv_text, args.jobs, args.job_text_col, args.top_k, args.normalize_per_query
    )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        ranked.to_csv(args.out, index=False)
        print(f"Résultats classés sauvegardés: {args.out}")

    print("\nRésumé pour le CV:")
    # Always show main summary
    main_keys = [
        "best_job_probability",
        f"mean_top{args.top_k}_probability",
        f"std_top{args.top_k}_probability",
        "suspicious_uniform_scores",
    ]
    for k in main_keys:
        v = summary.get(k, None)
        if v is not None:
            print(f"- {k}: {v:.3f}")

    if summary.get("normalized", False):
        print("\nNote: scores are normalized per CV (0..1). Use ranking or training thresholds for decisions; raw proba shown below if requested.")

    if args.show_raw:
        raw_keys = [
            "raw_min",
            "raw_max",
            "raw_best_job_probability",
            f"raw_mean_top{args.top_k}_probability",
        ]
        print("\nStatistiques brutes (non normalisées):")
        for k in raw_keys:
            v = summary.get(k, None)
            if v is not None:
                print(f"- {k}: {v:.3f}")

    # Preview top rows
    preview_cols = [c for c in ranked.columns if c.lower().startswith("intitul") or c.lower().startswith("metier")]
    display_cols = (preview_cols[:2] + ["proba_select"]) if preview_cols else (list(ranked.columns[:2]) + ["proba_select"])
    print("\nTop offres:")
    print(ranked[display_cols].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()

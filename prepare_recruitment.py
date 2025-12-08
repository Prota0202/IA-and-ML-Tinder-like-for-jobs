import argparse
import os
from typing import List

import pandas as pd


LABEL_CANDIDATES = ["callback", "hiringdecision", "hiring_decision", "decision", "label", "target", "HiringDecision"]
POSITIVE_TOKENS = {"1", "yes", "y", "true", "t", "oui", "callback", "call", "positive", "accept", "accepted", "hire", "hired"}
NEGATIVE_TOKENS = {"0", "no", "n", "false", "f", "non", "reject", "rejected", "negative"}


def detect_label_column(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in LABEL_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # Fallback: last column if binary
    for c in df.columns[::-1]:
        vals = df[c].dropna().unique()
        if len(vals) <= 2:
            return c
    raise ValueError("Impossible de détecter la colonne label (callback / décision). Spécifiez --label-col.")


def map_label(series: pd.Series) -> pd.Series:
    def _to_bin(x):
        if pd.isna(x):
            return pd.NA
        if isinstance(x, (int, float)):
            if x in (0, 1):
                return int(x)
        s = str(x).strip().lower()
        if s in POSITIVE_TOKENS:
            return 1
        if s in NEGATIVE_TOKENS:
            return 0
        # attempt numeric threshold
        try:
            v = float(s)
            return 1 if v >= 0.5 else 0
        except Exception:
            return pd.NA
    out = series.apply(_to_bin)
    return out.astype('Int64')


def is_id_column(name: str, series: pd.Series) -> bool:
    n = name.lower()
    if any(tok in n for tok in ["id", "uuid", "identifier"]):
        return True
    # Many unique values & numeric monotonic -> treat as id
    if pd.api.types.is_integer_dtype(series) and series.nunique() > 0.9 * len(series):
        return True
    return False


def classify_columns(df: pd.DataFrame, label_col: str):
    numeric_cols: List[str] = []
    cat_cols: List[str] = []
    drop_cols: List[str] = []
    for c in df.columns:
        if c == label_col:
            continue
        s = df[c]
        if is_id_column(c, s):
            drop_cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
            continue
        # object -> decide categorical vs drop
        if pd.api.types.is_object_dtype(s):
            # Heuristic: average length
            avg_len = s.dropna().astype(str).str.len().mean() if not s.dropna().empty else 0
            nunique = s.nunique(dropna=True)
            if avg_len <= 60 and nunique <= 50:
                cat_cols.append(c)
            else:
                drop_cols.append(c)
            continue
        # fallback
        drop_cols.append(c)
    return numeric_cols, cat_cols, drop_cols


def clean_dataset(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    # Map label to binary
    df = df.copy()
    df[label_col] = map_label(df[label_col])
    df = df[df[label_col].notna()]

    numeric_cols, cat_cols, drop_cols = classify_columns(df, label_col)

    # Impute numeric
    for c in numeric_cols:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)

    # Impute categorical + cast
    for c in cat_cols:
        df[c] = df[c].fillna("NA").astype(str)

    # One-hot encode categorical
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols], prefix=[c for c in cat_cols], drop_first=True)
    else:
        dummies = pd.DataFrame(index=df.index)

    cleaned = pd.concat([df[numeric_cols], dummies, df[label_col]], axis=1)
    cleaned = cleaned.reset_index(drop=True)
    return cleaned, numeric_cols, cat_cols, drop_cols


def main():
    parser = argparse.ArgumentParser(description="Nettoyage tabulaire du dataset recrutement")
    parser.add_argument("--input", required=True, help="Fichier CSV brut (recruitment_data.csv)")
    parser.add_argument("--output", default="recruitment_data_clean.csv", help="Fichier CSV propre de sortie")
    parser.add_argument("--label-col", default=None, help="Nom explicite de la colonne label si nécessaire")
    args = parser.parse_args()

    # Read CSV auto sep
    try:
        df = pd.read_csv(args.input)
    except Exception:
        df = pd.read_csv(args.input, sep=';')

    label_col = args.label_col or detect_label_column(df)
    print(f"Label détecté: {label_col}")

    cleaned, numeric_cols, cat_cols, dropped = clean_dataset(df, label_col)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    print(f"Fichier propre créé -> {args.output}")
    print(f"Colonnes numériques conservées ({len(numeric_cols)}): {numeric_cols}")
    print(f"Colonnes catégorielles encodées ({len(cat_cols)}): {cat_cols}")
    if dropped:
        print(f"Colonnes supprimées ({len(dropped)}): {dropped}")
    print(f"Shape finale: {cleaned.shape}")


if __name__ == "__main__":
    main()

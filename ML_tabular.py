import argparse
import os
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib


DEFAULT_FEATURES = ["PreviousCompanies", "ExperienceYears", "EducationLevel", "Age"]

LABEL_DEFAULT = "HiringDecision"


def _read_csv_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("-", "_").replace(" ", "_") for c in df.columns]
    return df


def _map_label_to_binary(series: pd.Series) -> pd.Series:
    def to_bin(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            if x in (0, 1):
                return int(x)
        s = str(x).strip().lower()
        pos = {"1", "yes", "y", "true", "t", "select", "selected", "accept", "accepted", "hire", "hired", "positive", "oui"}
        neg = {"0", "no", "n", "false", "f", "reject", "rejected", "decline", "declined", "negative", "non"}
        if s in pos:
            return 1
        if s in neg:
            return 0
        try:
            v = float(s)
            return 1 if v >= 0.5 else 0
        except Exception:
            return np.nan

    y = series.apply(to_bin)
    return y


def _resolve_features(df: pd.DataFrame, features: Optional[List[str]]) -> List[str]:
    if features:
        return [f for f in features if f in df.columns]
    # Try to map common synonyms to dataset columns
    lower_map = {c.lower(): c for c in df.columns}
    desired = {
        "previous companies worked": ["previouscompanies", "previous_companies", "previous_companies_worked"],
        "experience years": ["experienceyears", "experience_years", "years_experience"],
        "education level": ["educationlevel", "education_level"],
        "age": ["age"],
    }
    resolved: List[str] = []
    for _, opts in desired.items():
        for key in opts:
            if key in lower_map:
                resolved.append(lower_map[key])
                break
    # Fallback to defaults if mapping worked partially/fully
    if not resolved:
        resolved = [c for c in DEFAULT_FEATURES if c in df.columns]
    return resolved


def build_pipelines(feature_names: List[str], calibrate: bool = False) -> Dict[str, Pipeline]:
    pre_impute = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), feature_names),
        ],
        remainder="drop",
    )

    base_lr = Pipeline(
        steps=[
            ("imputer", pre_impute),
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    base_rf = Pipeline(
        steps=[
            ("imputer", pre_impute),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )

    base_hgb = Pipeline(
        steps=[
            ("imputer", pre_impute),
            ("clf", HistGradientBoostingClassifier(random_state=42)),
        ]
    )

    def maybe_calibrate(pipe: Pipeline) -> Pipeline:
        if not calibrate:
            return pipe
        return Pipeline(steps=[
            ("model", CalibratedClassifierCV(pipe, method="isotonic", cv=3))
        ])

    pipe_lr = maybe_calibrate(base_lr)
    pipe_rf = maybe_calibrate(base_rf)
    pipe_hgb = maybe_calibrate(base_hgb)

    return {"logreg": pipe_lr, "random_forest": pipe_rf, "histgb": pipe_hgb}


def _auto_features(df: pd.DataFrame, label_col: str, exclude: List[str]) -> List[str]:
    excl = {c.lower() for c in exclude + [label_col]}
    feats: List[str] = []
    for c in df.columns:
        if c.lower() in excl:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats


def _tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_distributions = {
        "n_estimators": [200, 300, 400, 600, 800],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"],
    }
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="roc_auc",
        cv=5,
        verbose=0,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    return best


def _threshold_suggestions(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Avoid division by zero in F1
    f1_scores = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
    idx_best_f1 = int(np.nanargmax(f1_scores))
    thr_best_f1 = float(thresholds[max(0, idx_best_f1 - 1)]) if len(thresholds) else 0.5
    # Precision >= 0.7 target
    thr_p70 = 0.5
    for p, t in zip(precision, np.r_[thresholds, thresholds[-1] if len(thresholds) else 0.5]):
        if p >= 0.7:
            thr_p70 = float(t)
            break
    return {"best_f1_threshold": thr_best_f1, "precision_0.7_threshold": thr_p70}


def cmd_train(args: argparse.Namespace) -> None:
    df_raw = _standardize_columns(_read_csv_auto(args.dataset))
    label_col = args.label_col if args.label_col in df_raw.columns else LABEL_DEFAULT
    if label_col not in df_raw.columns:
        raise ValueError(f"Colonne label '{args.label_col or LABEL_DEFAULT}' introuvable dans le dataset.")

    # Feature selection: explicit list > auto > default four
    if args.features:
        feat = _resolve_features(df_raw, args.features)
    elif args.auto_features:
        feat = _auto_features(df_raw, label_col, exclude=args.exclude or ["Gender"])
    else:
        feat = _resolve_features(df_raw, None)
    if not feat:
        raise ValueError("Aucune des features demandées n'a été trouvée. Vérifiez les noms de colonnes.")

    y = _map_label_to_binary(df_raw[label_col])
    X = df_raw[feat].copy()

    mask = y.notna()
    X = X[mask]
    y = y[mask].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Optional RF tuning on train set (uses numeric arrays after imputer)
    tuned_rf = None
    if args.tune_rf:
        tuned_rf = _tune_random_forest(X_train, y_train)

    pipelines = build_pipelines(feat, calibrate=args.calibrate)
    if tuned_rf is not None:
        # Replace RF in pipeline
        rf_pipe = pipelines["random_forest"]
        # Rebuild pipeline with same preprocessor but tuned RF
        pre = rf_pipe.named_steps.get("imputer")
        pipelines["random_forest"] = Pipeline(steps=[
            ("imputer", pre),
            ("clf", tuned_rf),
        ])
    best_name = None
    best_auc = -1.0
    best_pipe = None
    metrics: Dict[str, Dict[str, float]] = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # Some calibrated wrappers expose predict_proba at top level; otherwise access inner
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            # Fallback not expected; ensure numeric
            raise RuntimeError("Le pipeline ne fourni pas predict_proba pour l'évaluation.")
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        metrics[name] = {"accuracy": acc, "f1": f1, "roc_auc": auc}
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipe = pipe

    print("\n=== Evaluation (hold-out, tabular) ===")
    print(pd.DataFrame(metrics).T)
    print("\nClassification report (best):")
    print(classification_report(y_test, best_pipe.predict(X_test)))

    # Threshold suggestions
    y_proba_best = best_pipe.predict_proba(X_test)[:, 1]
    thr = _threshold_suggestions(y_test.values, y_proba_best)
    print(f"\nThreshold suggestions -> best F1: {thr['best_f1_threshold']:.3f}, precision>=0.70: {thr['precision_0.7_threshold']:.3f}")

    out_path = args.out if args.out else os.path.join("models", "hire_tabular.joblib")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Persist pipeline; it already knows which columns to select via ColumnTransformer
    joblib.dump({"pipeline": best_pipe, "features": feat, "label": label_col}, out_path)
    print(f"Modèle tabulaire '{best_name}' sauvegardé dans: {out_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model)
    pipe: Pipeline = bundle["pipeline"]
    feat: List[str] = bundle["features"]
    df = _standardize_columns(_read_csv_auto(args.input))
    missing = [c for c in feat if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier à scorer: {missing}")
    X = df[feat]
    proba = pipe.predict_proba(X)[:, 1]
    out = df.copy()
    out["proba_hired"] = proba
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"Scores sauvegardés dans {args.out}")
    else:
        print(out.head(20).to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Entraînement tabulaire (probabilité d'embauche)")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Entraîner sur données tabulaires")
    p_train.add_argument("--dataset", required=True, help="CSV tabulaire d'entraînement")
    p_train.add_argument("--label-col", default=LABEL_DEFAULT, help="Colonne label binaire (default: HiringDecision)")
    p_train.add_argument("--features", nargs="*", default=None, help="Liste explicite de colonnes features à utiliser")
    p_train.add_argument("--auto-features", action="store_true", help="Sélectionner automatiquement toutes les colonnes numériques (hors label et exclusions)")
    p_train.add_argument("--exclude", nargs="*", default=["Gender"], help="Colonnes à exclure (défaut: Gender)")
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--random-state", type=int, default=42)
    p_train.add_argument("--tune-rf", action="store_true", help="Ajuster hyperparamètres RandomForest via RandomizedSearchCV")
    p_train.add_argument("--calibrate", action="store_true", help="Calibrer les probabilités (isotonic, cv=3)")
    p_train.add_argument("--out", default=None, help="Chemin du modèle .joblib de sortie")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Scorer de nouvelles lignes tabulaires")
    p_pred.add_argument("--model", required=True, help="Modèle .joblib entraîné")
    p_pred.add_argument("--input", required=True, help="CSV à scorer")
    p_pred.add_argument("--out", default=None, help="CSV de sortie avec proba")
    p_pred.set_defaults(func=cmd_predict)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

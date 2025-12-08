import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import joblib
from feature_engineering import enrich_features

LABEL_CANDIDATES = ["label", "target", "HiringDecision", "hiringdecision", "decision", "callback"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--features', nargs='+', default=None)
    parser.add_argument('--label-col', default=None, help="Nom colonne label (auto si None)")
    parser.add_argument('--out', default="models/hire_tabular_core.joblib")
    args = parser.parse_args()

    # Chargement du dataset
    df = pd.read_csv(args.dataset)
    print(f"{df.shape=}")

    # Label : auto-detection pour robustesse
    if args.label_col:
        label_col = args.label_col
    else:
        for c in LABEL_CANDIDATES:
            if c in df.columns:
                label_col = c
                break
        else:
            raise ValueError(f"Aucune colonne cible trouvée dans: {list(df.columns)}")
    print(f"Colonne label utilisée : {label_col}")

    # Feature engineering avancé
    df = enrich_features(df)
    print("Features après enrichissement:", list(df.columns))

    # Sélection des features explicitement demandées (ou auto)
    features = args.features if args.features else [col for col in df.columns if col != label_col]
    print("Features utilisées:", features)
    X = df[features].copy()
    y = df[label_col]

    # One-hot encoding si besoin (normalement FE gère déjà le principal)
    for col in X.columns:
        if X[col].dtype == object:
            X = pd.get_dummies(X, columns=[col])

    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # Modèles avec class_weight/balanced
    logreg = LogisticRegression(class_weight="balanced", max_iter=500)
    rf = RandomForestClassifier(class_weight="balanced", n_estimators=150, max_depth=7, random_state=42)
    histgb = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.12, random_state=42)
    models = [("logreg", logreg), ("random_forest", rf), ("histgb", histgb)]

    best_f1, best_model, best_pred, best_name = 0, None, None, None
    for name, model in models:
        print(f"\n**** Training {name} ****")
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_valid)[:,1]
        # Optimisation du threshold pour meilleur F1
        precision, recall, thresholds = precision_recall_curve(y_valid, probas)
        f1_scores = 2*(precision*recall)/(precision+recall+1e-8)
        idx_best_f1 = f1_scores.argmax()
        threshold_f1 = thresholds[idx_best_f1] if idx_best_f1 < len(thresholds) else 0.5
        idx_prec = (precision >= 0.7).nonzero()[0]
        thr_prec70 = thresholds[idx_prec[0]] if len(idx_prec)>0 else 0.5

        print(f"Best F1: {f1_scores[idx_best_f1]:.3f} at threshold {threshold_f1:.3f}")
        if len(idx_prec):
            print(f"Threshold for precision>=0.70: {thr_prec70:.3f} (precision={precision[idx_prec[0]]:.3f})")
        else:
            print("Aucun threshold pour precision >= 0.7")

        y_pred = (probas >= threshold_f1).astype(int)
        report = classification_report(y_valid, y_pred, output_dict=True)
        print("Classification report:")
        print(classification_report(y_valid, y_pred))
        if report["weighted avg"]["f1-score"] > best_f1:
            best_f1 = report["weighted avg"]["f1-score"]
            best_model = model
            best_pred = y_pred
            best_name = name

    print(f"\nMeilleur modèle: {best_name} (f1={best_f1:.3f}), sauvegarde dans {args.out}")
    joblib.dump(best_model, args.out)

    print("Classification report (best model):")
    print(classification_report(y_valid, best_pred))

if __name__ == "__main__":
    main()
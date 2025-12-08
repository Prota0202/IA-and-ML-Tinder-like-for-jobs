import argparse
import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


# -----------------------------
# Helpers: IO & column detection
# -----------------------------

def _read_csv_auto(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
	"""Read CSV with basic sep autodetect (comma or semicolon)."""
	try:
		return pd.read_csv(path, nrows=nrows)
	except Exception:
		# Try semicolon
		return pd.read_csv(path, sep=";", nrows=nrows)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df.columns = [str(c).strip().replace("-", "_").replace(" ", "_") for c in df.columns]
	return df


def _infer_text_columns(
	df: pd.DataFrame,
	resume_col: Optional[str] = None,
	job_col: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
	"""Infer reasonable default columns for resume and job description."""
	cols_lower = {c.lower(): c for c in df.columns}

	resume_candidates = [
		"resume",
		"cv",
		"resume_text",
		"cv_text",
		"profil",
		"profil_text",
	]
	job_candidates = [
		"job_description",
		"jobdescription",
		"description",
		"job_desc",
		"job_text",
		# French datasets fallbacks
		"intitulé_de_l'offre",
		"intitulé_de_l'offre_clean",
		"intitule_de_l'offre",
		"intitule_de_l'offre_clean",
		"metier",
		"metier_clean",
		"secteur_d'activité",
		"secteur_d'activité_clean",
	]

	r = resume_col if resume_col in df.columns else None
	if r is None:
		for k in resume_candidates:
			if k in cols_lower:
				r = cols_lower[k]
				break

	j = job_col if job_col in df.columns else None
	if j is None:
		for k in job_candidates:
			if k in cols_lower:
				j = cols_lower[k]
				break

	return r, j


def _infer_label_column(df: pd.DataFrame, label_col: Optional[str]) -> Optional[str]:
	if label_col and label_col in df.columns:
		return label_col
	candidates = [
		"decision",
		"label",
		"status",
		"result",
		"target",
		"selected",
		"hire",
	]
	cols_lower = {c.lower(): c for c in df.columns}
	for k in candidates:
		if k in cols_lower:
			return cols_lower[k]
	return None


def _map_label_to_binary(series: pd.Series) -> pd.Series:
	"""Map assorted label forms to binary 0/1.
	Accepted positives: {1, True, '1', 'yes', 'y', 'true', 't', 'select', 'selected', 'accept', 'accepted', 'hire', 'hired'}
	"""
	def to_bin(x):
		if pd.isna(x):
			return np.nan
		if isinstance(x, (int, float)):
			if x == 1:
				return 1
			if x == 0:
				return 0
		s = str(x).strip().lower()
		pos = {"1", "yes", "y", "true", "t", "select", "selected", "accept", "accepted", "hire", "hired", "positive"}
		neg = {"0", "no", "n", "false", "f", "reject", "rejected", "decline", "declined", "negative"}
		if s in pos:
			return 1
		if s in neg:
			return 0
		# Fallback: try to parse as numeric
		try:
			v = float(s)
			return 1 if v >= 0.5 else 0
		except Exception:
			return np.nan

	y = series.apply(to_bin)
	return y


def _clean_text_series(s: pd.Series) -> pd.Series:
	import re
	s = s.fillna("").astype(str)
	# Lowercase
	s = s.str.lower()
	# Remove emails, urls, phone numbers
	s = s.str.replace(r"[\w\.-]+@[\w\.-]+", " ", regex=True)
	s = s.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
	s = s.str.replace(r"\b\+?\d[\d\s\-]{7,}\b", " ", regex=True)
	# Remove non-letter characters except basic punctuation spaces
	s = s.str.replace(r"[^a-zA-ZÀ-ÿ0-9\s\-\.]", " ", regex=True)
	# Collapse spaces
	s = s.str.replace(r"\s+", " ", regex=True).str.strip()
	return s


def _combine_text(df: pd.DataFrame, resume_col: Optional[str], job_col: Optional[str]) -> pd.Series:
	if resume_col and resume_col in df.columns and job_col and job_col in df.columns:
		return (df[resume_col].fillna("").astype(str) + " [SEP] " + df[job_col].fillna("").astype(str)).str.strip()
	if resume_col and resume_col in df.columns:
		return df[resume_col].fillna("").astype(str)
	if job_col and job_col in df.columns:
		return df[job_col].fillna("").astype(str)
	raise ValueError("Aucune colonne texte trouvée (CV ou description de poste).")


# -----------------------------
# Training/evaluation
# -----------------------------

def build_pipelines(
	max_features: int = 50000,
	ngram_high: int = 2,
	stopwords: Optional[str] = None,
	include_svm: bool = True,
	include_mlp: bool = True,
	include_sgd: bool = True,
	# Gradient Boosting removed due to runtime/benefit tradeoff
) -> Dict[str, Pipeline]:
	# Map stopwords option to sklearn-compatible value
	sw = None
	if stopwords == "english":
		sw = "english"
	elif stopwords == "french":
		# Minimal French stopword list (can be expanded later)
		sw = [
			"le","la","les","de","des","du","un","une","et","en","au","aux","pour","par",
			"sur","dans","avec","sans","ce","cet","cette","ces","qui","que","quoi","dont",
			"où","ou","mais","donc","or","ni","car","ne","pas","plus","moins","tres","très",
			"se","sa","son","ses","leur","leurs","nos","notre","votre","vos","je","tu","il",
			"elle","on","nous","vous","ils","elles"
		]
	tfidf = TfidfVectorizer(
		max_features=max_features,
		ngram_range=(1, ngram_high),
		lowercase=True,
		strip_accents="unicode",
		stop_words=sw,
	)

	pipe_lr = Pipeline(
		steps=[
			("tfidf", tfidf),
			(
				"clf",
				LogisticRegression(
					max_iter=2000,
					class_weight="balanced",
					n_jobs=-1 if hasattr(LogisticRegression(), "n_jobs") else None,
				),
			),
		]
	)

	pipe_rf = Pipeline(
		steps=[
			("tfidf", tfidf),
			("svd", TruncatedSVD(n_components=300, random_state=42)),
			("scaler", StandardScaler(with_mean=False)),
			(
				"clf",
				RandomForestClassifier(
					n_estimators=300,
					random_state=42,
					n_jobs=-1,
					max_depth=None,
					min_samples_leaf=1,
				),
			),
		]
	)

	pipelines: Dict[str, Pipeline] = {"logreg": pipe_lr, "random_forest": pipe_rf}

	if include_svm:
		# Linear SVM on TF-IDF; use decision_function, no proba
		pipe_svm = Pipeline(
			steps=[
				("tfidf", tfidf),
				("clf", LinearSVC(class_weight="balanced", random_state=42))
			]
		)
		pipelines["linear_svm"] = pipe_svm

	if include_sgd:
		# SGDClassifier (log loss) approximates logistic regression and supports decision_function
		pipe_sgd = Pipeline(
			steps=[
				("tfidf", tfidf),
				("clf", SGDClassifier(loss="log_loss", max_iter=2000, class_weight="balanced", random_state=42))
			]
		)
		pipelines["sgd_log"] = pipe_sgd

	if include_mlp:
		# Small MLP on truncated SVD features
		pipe_mlp = Pipeline(
			steps=[
				("tfidf", tfidf),
				("svd", TruncatedSVD(n_components=300, random_state=42)),
				("scaler", StandardScaler(with_mean=False)),
				("clf", MLPClassifier(hidden_layer_sizes=(128,), activation="relu", batch_size=256, max_iter=150, random_state=42))
			]
		)
		pipelines["mlp"] = pipe_mlp


	return pipelines


def train_model(
	df: pd.DataFrame,
	resume_col: Optional[str],
	job_col: Optional[str],
	label_col: str,
	test_size: float = 0.2,
	random_state: int = 42,
	clean_text: bool = False,
	tfidf_max_features: int = 50000,
	tfidf_ngram_high: int = 2,
	rf_max_depth: Optional[int] = None,
	rf_min_samples_leaf: int = 1,
	stopwords_lang: Optional[str] = None,
	include_svm: bool = True,
	include_mlp: bool = True,
	include_sgd: bool = True,
	# GB params removed
) -> Tuple[Pipeline, Dict[str, float], Dict[str, Dict[str, float]]]:
	X_text = _combine_text(df, resume_col, job_col)
	if clean_text:
		X_text = _clean_text_series(X_text)
	y = _map_label_to_binary(df[label_col])
	mask = y.notna() & X_text.notna() & (X_text.str.len() > 0)
	X_text = X_text[mask]
	y = y[mask].astype(int)

	if y.nunique() < 2:
		raise ValueError("Le label n'a pas au moins 2 classes après nettoyage.")

	X_train, X_test, y_train, y_test = train_test_split(
		X_text, y, test_size=test_size, random_state=random_state, stratify=y
	)

	stopwords = None
	if stopwords_lang in {"english", "french"}:
		stopwords = stopwords_lang
	pipelines = build_pipelines(
		max_features=tfidf_max_features,
		ngram_high=tfidf_ngram_high,
		stopwords=stopwords,
		include_svm=include_svm,
		include_mlp=include_mlp,
		include_sgd=include_sgd,
	)

	best_pipeline = None
	best_auc = -1.0
	overall_metrics: Dict[str, float] = {}
	per_model_metrics: Dict[str, Dict[str, float]] = {}

	for name, pipe in pipelines.items():
		# Adjust hyperparameters if applicable
		if name == "random_forest":
			rf: RandomForestClassifier = pipe.named_steps["clf"]
			rf.max_depth = rf_max_depth
			rf.min_samples_leaf = rf_min_samples_leaf
		pipe.fit(X_train, y_train)
		y_pred = pipe.predict(X_test)
		if hasattr(pipe, "predict_proba"):
			y_proba = pipe.predict_proba(X_test)[:, 1]
		elif hasattr(pipe, "decision_function"):
			scores = pipe.decision_function(X_test)
			s_min, s_max = scores.min(), scores.max()
			y_proba = (scores - s_min) / (s_max - s_min + 1e-9)
		else:
			y_proba = None

		acc = accuracy_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred)
		auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

		per_model_metrics[name] = {"accuracy": acc, "f1": f1, "roc_auc": auc}
		if y_proba is not None and auc > best_auc:
			best_auc = auc
			best_pipeline = pipe
			overall_metrics = {"accuracy": acc, "f1": f1, "roc_auc": auc}

	# As a safety net, pick any pipeline if none had proba
	if best_pipeline is None:
		# Choose the first pipeline
		first_name = list(pipelines.keys())[0]
		best_pipeline = pipelines[first_name]
		best_pipeline.fit(X_train, y_train)
		y_pred = best_pipeline.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred)
		overall_metrics = {"accuracy": acc, "f1": f1, "roc_auc": np.nan}

	# Print a compact report
	print("\n=== Evaluation (hold-out) ===")
	print(pd.DataFrame(per_model_metrics).T)
	print("\nClassification report (best):")
	print(classification_report(y_test, best_pipeline.predict(X_test)))

	return best_pipeline, overall_metrics, per_model_metrics


def save_pipeline(pipeline: Pipeline, out_path: str) -> None:
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	joblib.dump(pipeline, out_path)
	print(f"Modèle sauvegardé dans: {out_path}")


def load_pipeline(path: str) -> Pipeline:
	return joblib.load(path)


# -----------------------------
# Scoring CV vs Jobs
# -----------------------------

def _infer_job_text_columns_for_scoring(df: pd.DataFrame, job_text_col: Optional[str]) -> List[str]:
	if job_text_col and job_text_col in df.columns:
		return [job_text_col]
	candidates = [
		"job_description",
		"description",
		"job_text",
		# French fallbacks (title + job name + sector)
		"intitulé_de_l'offre_clean",
		"intitulé_de_l'offre",
		"intitule_de_l'offre_clean",
		"intitule_de_l'offre",
		"metier_clean",
		"metier",
		"secteur_d'activité_clean",
		"secteur_d'activité",
	]
	cols_lower = {c.lower(): c for c in df.columns}
	found: List[str] = []
	for k in candidates:
		if k in cols_lower:
			found.append(cols_lower[k])
	# If nothing found, fallback to first object column
	if not found:
		obj_cols = [c for c in df.columns if df[c].dtype == object]
		if obj_cols:
			found = [obj_cols[0]]
	return found[:3]  # keep up to 3 columns


def score_jobs(
	pipeline: Pipeline,
	cv_text: str,
	jobs_df: pd.DataFrame,
	job_text_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
	df = jobs_df.copy()
	if job_text_cols is None or not job_text_cols:
		job_text_cols = _infer_job_text_columns_for_scoring(df, job_text_col=None)

	# Build text to score: concatenate selected job text cols
	text_jobs = (
		df[job_text_cols].fillna("").astype(str).agg(" ".join, axis=1)
		if len(job_text_cols) > 1
		else df[job_text_cols[0]].fillna("").astype(str)
	)

	# Pair each job with the same CV text (CV [SEP] JOB)
	combined = pd.Series(cv_text, index=text_jobs.index) + " [SEP] " + text_jobs

	# Predict probabilities
	if hasattr(pipeline, "predict_proba"):
		probs = pipeline.predict_proba(combined)[:, 1]
	else:
		# Decision function fallback -> normalize to 0..1 via min-max
		scores = pipeline.decision_function(combined)
		s_min, s_max = scores.min(), scores.max()
		probs = (scores - s_min) / (s_max - s_min + 1e-9)

	out = df.copy()
	out["proba_select"] = probs
	out = out.sort_values("proba_select", ascending=False)
	return out


# -----------------------------
# CLI
# -----------------------------

def cmd_train(args: argparse.Namespace) -> None:
	df = _standardize_columns(_read_csv_auto(args.dataset))
	label_col = _infer_label_column(df, args.label_col)
	if not label_col:
		raise ValueError("Colonne label introuvable. Spécifiez --label-col.")

	resume_col, job_col = _infer_text_columns(df, args.resume_col, args.job_col)
	if not resume_col and not job_col:
		raise ValueError("Aucune colonne texte trouvée. Spécifiez --resume-col et/ou --job-col.")

	print(f"Colonnes utilisées -> resume: {resume_col}, job: {job_col}, label: {label_col}")

	pipeline, best_metrics, per_model = train_model(
		df,
		resume_col=resume_col,
		job_col=job_col,
		label_col=label_col,
		test_size=args.test_size,
		random_state=args.random_state,
		clean_text=args.clean_text,
		tfidf_max_features=args.tfidf_max_features,
		tfidf_ngram_high=args.tfidf_ngram_high,
		rf_max_depth=args.rf_max_depth,
		rf_min_samples_leaf=args.rf_min_samples_leaf,
		stopwords_lang=args.stopwords,
		include_svm=(not args.disable_svm),
		include_mlp=(not args.disable_mlp),
		include_sgd=(not args.disable_sgd),
		# GB disabled/removed
	)

	print("\nMeilleur modèle (hold-out):", best_metrics)
	out_path = args.out if args.out else os.path.join("models", "cv_job_selector.joblib")
	save_pipeline(pipeline, out_path)


def cmd_predict(args: argparse.Namespace) -> None:
	pipeline = load_pipeline(args.model)
	# Load CV text
	if args.cv_str:
		cv_text = args.cv_str
	elif args.cv_file:
		with open(args.cv_file, "r", encoding="utf-8", errors="ignore") as f:
			cv_text = f.read()
	else:
		raise ValueError("Fournissez --cv-str ou --cv-file pour le texte du CV.")

	# Load jobs
	jobs_df = _standardize_columns(_read_csv_auto(args.jobs))
	job_cols = None
	if args.job_text_col:
		job_cols = [args.job_text_col]

	scored = score_jobs(pipeline, cv_text=cv_text, jobs_df=jobs_df, job_text_cols=job_cols)
	if args.top_k and args.top_k > 0:
		scored = scored.head(args.top_k)

	if args.out:
		os.makedirs(os.path.dirname(args.out), exist_ok=True)
		scored.to_csv(args.out, index=False)
		print(f"Résultats sauvegardés dans {args.out}")
	else:
		# Print a compact preview
		preview_cols = [c for c in scored.columns if c.lower().startswith("intitul") or c.lower().startswith("metier")]
		display_cols = preview_cols[:2] + ["proba_select"] if preview_cols else list(scored.columns[:2]) + ["proba_select"]
		print(scored[display_cols].head(20).to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="CV-Job Matching: entraînement et scoring")
	sub = p.add_subparsers(dest="command", required=True)

	p_train = sub.add_parser("train", help="Entraîner un modèle sur un dataset annoté")
	p_train.add_argument("--dataset", required=True, help="CSV avec colonnes texte + label")
	p_train.add_argument("--resume-col", default=None, help="Nom de la colonne CV (auto si non fourni)")
	p_train.add_argument("--job-col", default=None, help="Nom de la colonne Job Description (auto si non fourni)")
	p_train.add_argument("--label-col", default=None, help="Nom de la colonne label (Decision/select/reject)")
	p_train.add_argument("--test-size", type=float, default=0.2)
	p_train.add_argument("--random-state", type=int, default=42)
	p_train.add_argument("--clean-text", action="store_true", help="Nettoyer le texte (emails/urls/tel, ponctuation, espaces)")
	p_train.add_argument("--tfidf-max-features", type=int, default=50000, help="Nombre max de features TF-IDF")
	p_train.add_argument("--tfidf-ngram-high", type=int, default=2, help="N-gram max (1=uni, 2=bi)")
	p_train.add_argument("--stopwords", choices=["english", "french"], default=None, help="Stopwords pour TF-IDF")
	p_train.add_argument("--rf-max-depth", type=int, default=None, help="Max depth du RandomForest")
	p_train.add_argument("--rf-min-samples-leaf", type=int, default=1, help="Min samples leaf du RandomForest")
	p_train.add_argument("--disable-svm", action="store_true", help="Ne pas inclure Linear SVM dans la comparaison")
	p_train.add_argument("--disable-mlp", action="store_true", help="Ne pas inclure MLP (réseau de neurones) dans la comparaison")
	p_train.add_argument("--disable-sgd", action="store_true", help="Ne pas inclure SGDClassifier (log loss) dans la comparaison")
	p_train.add_argument("--out", default=None, help="Chemin du modèle .joblib (défaut: models/cv_job_selector.joblib)")
	p_train.set_defaults(func=cmd_train)

	p_pred = sub.add_parser("predict", help="Scorer un CV contre une liste d'offres")
	p_pred.add_argument("--model", required=True, help="Chemin du modèle .joblib")
	p_pred.add_argument("--jobs", required=True, help="CSV des offres d'emploi")
	p_pred.add_argument("--job-text-col", default=None, help="Colonne texte à utiliser pour les offres")
	g_cv = p_pred.add_mutually_exclusive_group(required=True)
	g_cv.add_argument("--cv-file", help="Fichier texte du CV")
	g_cv.add_argument("--cv-str", help="Texte brut du CV")
	p_pred.add_argument("--top-k", type=int, default=20, help="Nombre de résultats à afficher")
	p_pred.add_argument("--out", default=None, help="CSV de sortie avec probabilités")
	p_pred.set_defaults(func=cmd_predict)

	return p


def main():
	parser = build_arg_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()


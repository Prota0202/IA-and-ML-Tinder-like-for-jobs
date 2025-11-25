import os
import re
import json
from typing import Any, Dict, List

from dotenv import load_dotenv

# Import LangChain/Mistral avec fallback si indisponible
LANGCHAIN_AVAILABLE = False
try:
	from langchain.prompts import PromptTemplate  # type: ignore
	from langchain.chains import LLMChain  # type: ignore
	from langchain_mistralai.chat_models import ChatMistralAI  # type: ignore
	LANGCHAIN_AVAILABLE = True
except Exception:
	# Fournit des symboles neutres pour éviter NameError si annotations ou accès
	PromptTemplate = None  # type: ignore
	LLMChain = None  # type: ignore
	ChatMistralAI = None  # type: ignore
	import requests  # fallback HTTP direct


load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

# Détermination chemin CV: argument CLI prioritaire, sinon fichier dans même dossier.
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CV = os.path.join(SCRIPT_DIR, "mon_cv.txt")
CV_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CV


# ----------------------------- Lecture CV -----------------------------
def read_cv(path: str) -> str:
	# Essaye chemin direct; si relatif et introuvable, tente par rapport au dossier script
	candidate_paths = [path]
	if not os.path.isabs(path):
		candidate_paths.append(os.path.join(SCRIPT_DIR, path))
	for p in candidate_paths:
		try:
			with open(p, "r", encoding="utf-8", errors="replace") as f:
				return f.read()
		except FileNotFoundError:
			continue
	return ""


# ------------------------- Heuristiques locales -----------------------
def extract_langues(cv_text: str) -> List[Dict[str, Any]]:
	langues = []
	block_match = re.search(r"Langues\n(.*?)(?:\n\n|$)", cv_text, flags=re.IGNORECASE | re.DOTALL)
	if block_match:
		for line in block_match.group(1).splitlines():
			line = line.strip()
			if not line:
				continue
			m = re.match(r"([A-Za-zéèêàùïç' ]+)\s*:\s*([A-Za-z0-9+ ]+)", line)
			if m:
				langues.append({"langue": m.group(1).strip().lower(), "niveau": m.group(2).strip()})
	if not langues:  # fallback simple
		for l in ["francais", "anglais", "arabe", "néerlandais", "neerlandais", "espagnol", "allemand", "portugais", "italien"]:
			if l in cv_text.lower():
				langues.append({"langue": l, "niveau": None})
	return langues


def extract_etudes(cv_text: str) -> List[Dict[str, Any]]:
	lines = cv_text.splitlines()
	etudes = []
	pattern_year = re.compile(r"(19|20)\d{2}")
	for i, line in enumerate(lines):
		ll = line.lower()
		if any(k in ll for k in ["master", "bachelier", "licence", "ingénierie", "ingenierie", "diplôme", "diplome"]):
			periode = pattern_year.findall(" ".join(lines[i:i+2]))
			etudes.append({
				"diplome": line.strip(),
				"domaine": None,
				"institution": None,
				"periode": periode or None,
			})
	return etudes


def extract_localisation(cv_text: str) -> Dict[str, Any]:
	loc = {"ville": None, "code_postal": None}
	for line in cv_text.splitlines()[:8]:
		ll = line.lower()
		if "bruxelles" in ll:
			loc["ville"] = "Bruxelles"
		cp = re.search(r"\b\d{4}\b", line)
		if cp:
			loc["code_postal"] = cp.group(0)
	return loc


def extract_pref_contrat(cv_text: str) -> Dict[str, Any]:
	lower = cv_text.lower()
	pref = {"type_contrat_recherche": None, "regime_travail_recherche": None}

	# Mapping régimes (mi-temps => Temps partiel)
	regime_map = {
		"temps plein": "Temps plein",
		"plein temps": "Temps plein",
		"full time": "Temps plein",
		"temps partiel": "Temps partiel",
		"part time": "Temps partiel",
		"mi-temps": "Temps partiel",
		"half time": "Temps partiel",
	}
	for key, canon in regime_map.items():
		if key in lower:
			pref["regime_travail_recherche"] = canon
			break

	# Mapping types de contrat (synonymes -> canon)
	contrat_map = {
		"intérimaire avec option sur durée indéterminée": "Intérimaire avec option sur durée indéterminée",
		"interimaire avec option sur duree indeterminee": "Intérimaire avec option sur durée indéterminée",
		"intérimaire avec option sur durée indeterminee": "Intérimaire avec option sur durée indéterminée",
		"intérimaire avec option sur duree indéterminée": "Intérimaire avec option sur durée indéterminée",
		"intérimaire": "Intérimaire",
		"interimaire": "Intérimaire",
		"durée indéterminée": "Durée indéterminée",
		"durée indeterminee": "Durée indéterminée",
		"duree indéterminée": "Durée indéterminée",
		"duree indeterminee": "Durée indéterminée",
		"cdi": "Durée indéterminée",
		"durée déterminée": "Durée déterminée",
		"duree déterminée": "Durée déterminée",
		"durée determinee": "Durée déterminée",
		"duree determinee": "Durée déterminée",
		"cdd": "Durée déterminée",
		"etudiant": "Etudiant",
		"étudiant": "Etudiant",
		"remplacement": "Remplacement",
		"contrat collaboration indépendant": "Contrat collaboration indépendant",
		"contrat collaboration independant": "Contrat collaboration indépendant",
		"freelance": "Contrat collaboration indépendant",
		"indépendant": "Contrat collaboration indépendant",
		"independant": "Contrat collaboration indépendant",
		"flexi-jobs": "Flexi-Jobs",
		"flexijobs": "Flexi-Jobs",
		"journalier": "Journalier (occasionnel ou saisonnier)",
		"occasionnel": "Journalier (occasionnel ou saisonnier)",
		"saisonnier": "Journalier (occasionnel ou saisonnier)",
		"salarié statutaire": "Salarié statutaire",
		"salarie statutaire": "Salarié statutaire",
		"nettement défini": "Nettement défini",
		"nettement defini": "Nettement défini",
	}

	# Recherche du match le plus long pour éviter collisions partielles
	found = None
	for key in sorted(contrat_map.keys(), key=len, reverse=True):
		if key in lower:
			found = contrat_map[key]
			break
	if found:
		pref["type_contrat_recherche"] = found

	return pref


# ------------------------------ Prompt LLM ----------------------------
# Le LLM doit aussi générer des questions pour les champs manquants.
PROMPT_STR = (
	"Tu reçois le texte intégral d'un CV. Objectif: préparer la recherche d'emploi. "
	"1) Extrait STRICTEMENT les informations présentes. 2) Génère des questions pertinentes pour CHAQUE champ manquant. "
	"Ne fabrique pas d'information absente.\n"
	"Renvoie SEULEMENT un JSON valide du format exact: {\n"
	"  \"etudes\": [ { \"diplome\":..., \"domaine\":..., \"institution\":..., \"periode\":... } ],\n"
	"  \"domaines_preference\": [ ... ],\n"
	"  \"langues\": [ { \"langue\":..., \"niveau\":... } ],\n"
	"  \"type_contrat_recherche\": ...,\n"
	"  \"regime_travail_recherche\": ...,\n"
	"  \"localisation\": { \"ville\":..., \"code_postal\":... },\n"
	"  \"autres\": [ ... ],\n"
	"  \"questions\": [ {\n"
	"      \"champ\": \"nom_du_champ\",\n"
	"      \"question\": \"Texte question claire\",\n"
	"      \"format_attendu\": \"exemple ou options\",\n"
	"      \"type\": \"choice|text\",\n"
	"      \"options\": [\"option1\", \"option2\"]\n"
	"  } ]\n"
	"}\n"
	"Règles supplémentaires:\n"
	"- Interprète synonymes type de contrat: CDI -> Durée indéterminée; CDD -> Durée déterminée; Freelance/indépendant -> Contrat collaboration indépendant; mi-temps -> Temps partiel.\n"
	"- Valeurs autorisées type_contrat_recherche (sélectionne la meilleure si le CV en offre plusieurs): Intérimaire avec option sur durée indéterminée, Durée indéterminée, Intérimaire, Durée déterminée, Etudiant, Remplacement, Contrat collaboration indépendant, Flexi-Jobs, Journalier (occasionnel ou saisonnier), Salarié statutaire, Nettement défini.\n"
	"- Valeurs autorisées regime_travail_recherche: Temps plein, Temps partiel.\n"
	"- Si localisation partielle (absence code postal) laisse code_postal à null.\n"
	"- N'AJOUTE DES QUESTIONS QUE pour les champs NULL/VIDES après extraction stricte.\n"
	"- Pour les questions: si \"champ\" est \"regime_travail_recherche\", mets \"type\": \"choice\" et \"options\": [\"Temps plein\", \"Temps partiel\"]. Si \"champ\" est \"type_contrat_recherche\", mets \"type\": \"choice\" et \"options\" = liste complète ci-dessus. Si \"champ\" est \"code_postal\" ou \"ville\", mets \"type\": \"text\". Pour \"langues\" et \"etudes\", mets \"type\": \"text\" et un \"format_attendu\" explicite (ex: langues: \"francais:C2; anglais:B2\").\n"
	"- Le JSON doit être directement parsable sans texte autour.\n\nCV:\n{cv_text}"
)

def build_chain():  # pas d'annotation union pour compat Python 3.9
	if not API_KEY or not LANGCHAIN_AVAILABLE or LLMChain is None or ChatMistralAI is None:
		return None
	llm = ChatMistralAI(mistral_api_key=API_KEY, model="mistral-small-latest")
	prompt = PromptTemplate(input_variables=["cv_text"], template=PROMPT_STR)
	return LLMChain(llm=llm, prompt=prompt)


def call_mistral_fallback(cv_text: str) -> str:
	"""Appel direct API Mistral si LangChain indisponible."""
	if not API_KEY:
		return "(Pas de clé API)"
	url = "https://api.mistral.ai/v1/chat/completions"
	headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
	payload = {
		"model": "mistral-small-latest",
		"messages": [{"role": "user", "content": PROMPT_STR.replace("{cv_text}", cv_text)}],
		"temperature": 0.2,
	}
	try:
		resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
		if resp.status_code != 200:
			return f"Erreur HTTP {resp.status_code}: {resp.text}"
		data = resp.json()
		return data.get("choices", [{}])[0].get("message", {}).get("content", "")
	except Exception as e:
		return f"Exception requête: {e}"


# ---------------------------- Parsing JSON ----------------------------
def safe_json_extract(text: str) -> Dict[str, Any]:
	if not text:
		return {}
	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1:
		return {}
	raw = text[start:end+1]
	try:
		return json.loads(raw)
	except json.JSONDecodeError:
		return {}


# ------------------------- Fusion des résultats -----------------------
def fuse(local: Dict[str, Any], llm: Dict[str, Any]) -> Dict[str, Any]:
	return {
		"etudes": llm.get("etudes") or local.get("etudes"),
		"domaines_preference": llm.get("domaines_preference") or [],
		"langues": llm.get("langues") or local.get("langues"),
		"type_contrat_recherche": llm.get("type_contrat_recherche") or local.get("type_contrat_recherche"),
		"regime_travail_recherche": llm.get("regime_travail_recherche") or local.get("regime_travail_recherche"),
		"localisation": llm.get("localisation") or local.get("localisation"),
		"autres": llm.get("autres") or [],
	}


def find_missing(profile: Dict[str, Any]) -> List[str]:
	missing = []
	if not profile.get("etudes"):
		missing.append("etudes")
	if not profile.get("langues"):
		missing.append("langues")
	if not profile.get("type_contrat_recherche"):
		missing.append("type_contrat_recherche")
	if not profile.get("regime_travail_recherche"):
		missing.append("regime_travail_recherche")
	loc = profile.get("localisation") or {}
	if not (loc.get("ville")):
		missing.append("ville")
	if not (loc.get("code_postal")):
		missing.append("code_postal")
	return missing


def interactive_fill_with_questions(profile: Dict[str, Any], llm_json: Dict[str, Any]) -> Dict[str, Any]:
	questions = llm_json.get("questions") or []
	if not questions:
		return profile
	import sys
	if not sys.stdin.isatty():
		print("[LOCAL] Mode non interactif: questions ignorées.")
		return profile
	print("[LOCAL] Questions pour compléter le profil :")
	for q in questions:
		champ = q.get("champ")
		texte = q.get("question")
		fmt = q.get("format_attendu")
		if not champ or not texte:
			continue
		answer = input(f" - {texte} (format: {fmt}): ").strip()
		if not answer:
			continue
		if champ == "langues":
			lang_list = []
			for chunk in answer.split(";"):
				parts = chunk.split(":")
				if len(parts) >= 2:
					lang_list.append({"langue": parts[0].strip().lower(), "niveau": parts[1].strip()})
				elif parts[0].strip():
					lang_list.append({"langue": parts[0].strip().lower(), "niveau": None})
			if lang_list:
				profile["langues"] = lang_list
		elif champ == "etudes":
			etude_objs = []
			for chunk in answer.split(";"):
				parts = [p.strip() for p in chunk.split("|")]
				if len(parts) >= 4:
					etude_objs.append({"diplome": parts[0], "domaine": parts[1] or None, "institution": parts[2] or None, "periode": parts[3] or None})
				elif parts and parts[0]:
					etude_objs.append({"diplome": parts[0], "domaine": None, "institution": None, "periode": None})
			if etude_objs:
				profile["etudes"] = etude_objs
		elif champ in ["type_contrat_recherche", "regime_travail_recherche"]:
			profile[champ] = answer
		elif champ in ["localisation"]:
			# Permet formats "Ville | CodePostal" ou juste ville
			parts = [p.strip() for p in answer.split("|")]
			loc = profile.get("localisation") or {"ville": None, "code_postal": None}
			if parts:
				loc["ville"] = parts[0]
			if len(parts) > 1:
				loc["code_postal"] = parts[1]
			profile["localisation"] = loc
	return profile


def main():
	cv_text = read_cv(CV_PATH)
	if not cv_text:
		print(f"CV introuvable ou vide: {CV_PATH}\nAstuce: lance 'python CV.py chemin/vers/mon_cv.txt' ou place 'mon_cv.txt' dans {SCRIPT_DIR}")
		return

	print("Lecture du CV effectuée. Longueur:", len(cv_text), "caractères")

	local_data = {
		"langues": extract_langues(cv_text),
		"etudes": extract_etudes(cv_text),
		"localisation": extract_localisation(cv_text),
		**extract_pref_contrat(cv_text),
	}
	print("Heuristiques locales:")
	print(json.dumps(local_data, ensure_ascii=False, indent=2))

	if not API_KEY:
		print("MISTRAL_API_KEY manquant: saut de l'appel LLM.")
		llm_json = {}
	else:
		chain = build_chain()
		if chain is not None:
			try:
				raw = chain.run({"cv_text": cv_text})
			except Exception as e:
				raw = f"Erreur LLM: {e}"
		else:
			# Fallback HTTP direct si LangChain indisponible
			raw = call_mistral_fallback(cv_text)
		print("Réponse brute LLM:")
		print(raw)
		llm_json = safe_json_extract(raw)
		print("JSON parsé LLM:")
		print(json.dumps(llm_json, ensure_ascii=False, indent=2) if llm_json else "(Parsing échoué)")

	profile = fuse(local_data, llm_json)
	# Priorité aux questions fournies par le LLM; si aucune, fallback ancien mécanisme
	if llm_json.get("questions"):
		profile = interactive_fill_with_questions(profile, llm_json)
	else:
		missing = find_missing(profile)
		profile = interactive_fill_with_questions(profile, {"questions": [{"champ": m, "question": f"Veuillez renseigner {m}", "format_attendu": "Texte"} for m in missing]})

	print("\nProfil final consolidé:")
	print(json.dumps(profile, ensure_ascii=False, indent=2))
	print("\nProchaine étape: utiliser ce profil pour requêter une API d'offres.")


if __name__ == "__main__":
	main()


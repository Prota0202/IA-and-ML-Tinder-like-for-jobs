import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
from streamlit_swipecards import streamlit_swipecards
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

# --- Extraction automatique des langues du CV ---
def extract_langues_from_cv(cv_text):
    langues_possibles = [
        "francais", "anglais", "neerlandais", "arabe",
        "italien", "espagnol", "allemand", "portugais"
    ]
    cv_text = cv_text.lower()
    return set(langue for langue in langues_possibles if langue in cv_text)

cv = open("mon_cv.txt").read()
langues_utilisateur = extract_langues_from_cv(cv)
st.sidebar.write("Langues détectées dans le CV :", ', '.join(langues_utilisateur))

# --- Chargement des données ---
df = pd.read_csv('offres-d-emploi.csv')
features = ['Lieux de travail_clean', "Secteur d'activité_clean", 'Métier_clean', 'Langues_clean']

# --- Scoring ML sur les langues ---
def score_langue(liste_langues, langues_candidat):
    if isinstance(liste_langues, str):
        liste_langues = [x.strip().lower() for x in liste_langues.replace(";", ",").split(',') if x.strip()]
    needed = set(liste_langues) - {"-"}
    if not needed:
        return 0
    count = len(needed & langues_candidat)
    total = len(needed)
    ratio = count / total if total > 0 else 0
    return ratio

df['score_langue'] = df['Langues_clean'].apply(lambda x: score_langue(x, langues_utilisateur))
df['label'] = df['score_langue'].apply(lambda ratio: int(ratio >= 1/3))

# --- Encodage et RandomForest ---
X = df[features].fillna('-')
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)
model = RandomForestClassifier()
model.fit(X_encoded, df['label'])

# --- Prédiction & filtrage par ML ---
pred = model.predict(X_encoded)
df['prediction_rf'] = pred
filtered_jobs = df[df['prediction_rf'] == 1][["Intitulé de l'offre_clean", "Métier_clean", "Langues_clean", "score_langue"]].dropna().head(5)

# --- Fix affichage sidebar, évite l’erreur apostrophe ---
st.sidebar.write("Offres filtrées par ML :", filtered_jobs["Intitulé de l'offre_clean"].tolist())

# --- Setup LangChain/Mistral avec prompt polyvalent ---
prompt = PromptTemplate(
    input_variables=["cv", "job"],
    template=(
        "Analyse le CV suivant par rapport au poste proposé.\n"
        "Donne uniquement un score de correspondance entre 0 (aucune correspondance) et 100 (correspondance parfaite).\n"
        "Le score doit se baser sur les compétences, les expériences et la formation du candidat.\n"
        "Donne un score élevé uniquement si le CV mentionne clairement des compétences ou expériences réellement utiles pour ce poste.\n"
        "Si le CV ne cite aucune compétence pertinente, donne un score faible (proche de 0).\n"
        "Ne prends pas en compte le niveau social, seulement les critères métier/technique et les langues.\n"
        "CV :\n{cv}\nJob :\n{job}"
    ),
)
llm = ChatMistralAI(mistral_api_key=API_KEY, model="mistral-small-latest")
chain = LLMChain(llm=llm, prompt=prompt)

# --- Scoring IA sur jobs validés par ML ---
def get_score_langchain(cv, job_desc):
    try:
        score_text = chain.run({"cv": cv, "job": job_desc})
        score_text = str(score_text).strip()
        match = re.search(r"\b([1]?[0-9]{1,2}|100)\b", score_text)
        if match:
            score = int(match.group(0))
        else:
            score = 0
        return score, score_text
    except Exception as e:
        print("Erreur API LangChain/Mistral :", e)
        return 0, "Erreur API"

results = []
for idx, row in filtered_jobs.iterrows():
    title = row["Intitulé de l'offre_clean"]
    metier = row["Métier_clean"]
    job_langues = row["Langues_clean"]
    job_desc = f"{title} - {metier} (Langues requises: {job_langues})"
    score, explanation = get_score_langchain(cv, job_desc)
    results.append({
        "intitule": title,
        "description": job_desc,
        "score_langue_ml": row["score_langue"],
        "score_ia": score,
        "explication_ia": explanation
    })

results = sorted(results, key=lambda x: -x['score_ia'])

st.title("Le Tinder de LinkedIn")
for r in results:
    st.write(f"Job: {r['intitule']} ({r['score_ia']}%) — Score langue ML: {r['score_langue_ml']:.2f}")

cards = []
for r in results:
    cards.append({
        "name": f"{r['intitule']} ({r['score_ia']}%)",
        "description": f"{r['description']}\nScore langue ML : {r['score_langue_ml']:.2f}\n",
    })

result = streamlit_swipecards(
    cards=cards,
    key="swipe_cards",
    last_card_message="Fin des offres !"
)

if result:
    st.write("Dernière action swipe :", result)


# Petit ajout temporaire en haut du fichier
# Redeploy test for Streamlit Cloud

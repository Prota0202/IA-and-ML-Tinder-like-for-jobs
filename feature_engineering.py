import pandas as pd

def enrich_features(df):
    """ Ajoute de nouvelles features tabulaires utiles pour la prédiction. """
    # Ratio expérience / âge
    if "ExperienceYears" in df.columns and "Age" in df.columns:
        df["ExperienceToAgeRatio"] = df["ExperienceYears"] / (df["Age"] + 1)
    # Nombre d'entreprises par an d'expérience
    if "PreviousCompanies" in df.columns and "ExperienceYears" in df.columns:
        df["CompaniesPerYear"] = df["PreviousCompanies"] / (df["ExperienceYears"] + 1)
    # Niveau d'étude ordinal
    if "EducationLevel" in df.columns:
        mapping = {
            'aucun':0, 'none':0, 'high_school':1, 'bac':1, 'baccalaureat':1, 'baccalauréat':1,
            'bachelor':2, 'licence':2, 'license':2,
            'master':3, 'msc':3, 'maitrise':3, 'maîtrise':3, 'ingénieur':3, 'ingenieur':3,
            'phd':4, 'doctorat':4
        }
        df["EducationLevel_ord"] = df["EducationLevel"].map(lambda x: mapping.get(str(x).lower(), 0))
    # Est junior (moins de 2 ans d'xp) ou senior (plus de 7 ans)
    if "ExperienceYears" in df.columns:
        df["IsJunior"] = (df["ExperienceYears"] < 2).astype(int)
        df["IsSenior"] = (df["ExperienceYears"] > 7).astype(int)
    # Est jeune diplômé
    if "EducationLevel_ord" in df.columns and "Age" in df.columns:
        df["YoungGrad"] = ((df["EducationLevel_ord"] >= 2) & (df["Age"] <= 28)).astype(int)
    # Binaire si ville en capitale (exemple)
    if "City" in df.columns:
        capitals = ["paris", "bruxelles", "brussels", "london", "new york"]
        df["IsCapitalCity"] = df["City"].str.lower().isin(capitals).astype(int)
    # Dummies (modulable selon tes besoins)
    cat_for_dummies = [c for c in df.columns if df[c].dtype==object and df[c].nunique() < 20 and c not in ["EducationLevel", "LastCompany"]]
    if cat_for_dummies:
        df = pd.get_dummies(df, columns=cat_for_dummies, drop_first=True)
    return df
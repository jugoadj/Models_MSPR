import pandas as pd

# Charger les deux datasets
df_principal = pd.read_csv("ensemble_elections.csv")  # Ton fichier avec les données électorales
df_socio = pd.read_csv("donnees_socio_fusionnees.csv", sep=';')  # Le fichier des données socio-économiques

# Assurer que les codes sont sous forme de chaînes de caractères et avec la bonne longueur
df_principal["code_com"] = df_principal["code_com"].astype(str).str.zfill(3)  # Code commune
df_principal["code_dept"] = df_principal["code_dept"].astype(str).str.zfill(2)  # Code département

# Créer la colonne CODGEO dans le fichier principal
df_principal["CODGEO"] = df_principal["code_dept"] + df_principal["code_com"]

# Vérification que les années sont bien au même format
df_principal["annee"] = df_principal["annee"].astype(int)
df_socio["année"] = df_socio["année"].astype(int)

# Fusionner les deux datasets sur `CODGEO` et `annee`
df_merge = pd.merge(
    df_principal,
    df_socio,
    left_on=["CODGEO", "annee"],  # Colonnes à joindre dans le fichier principal
    right_on=["CODGEO", "année"],  # Colonnes à joindre dans le fichier socio-éco
    how="left"  # Garder toutes les lignes du dataset principal
)

# Optionnel : Supprimer les colonnes redondantes comme `CODGEO` et `année` si elles ne sont plus nécessaires
df_merge = df_merge.drop(columns=["CODGEO", "année"])

# Sauvegarder le résultat dans un nouveau fichier CSV
df_merge.to_csv("dataset_fusionneelecsocio.csv", index=False)

# Affichage rapide du résultat fusionné
print(df_merge.head())

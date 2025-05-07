import pandas as pd

# Charger les fichiers
df_2017 = pd.read_csv("donnees_socio_2017.csv", sep=";")
df_2022 = pd.read_csv("datasetest.csv", sep=",")

# Extraire les codes CODGEO uniques de chaque fichier
codes_2017 = set(df_2017['CODGEO'].unique())
codes_2022 = set(df_2022['CODGEO'].unique())

# Vérifier si les deux ensembles sont identiques
if codes_2017 == codes_2022:
    print("✅ Les colonnes CODGEO des deux fichiers sont exactement identiques.")
else:
    print("❌ Les colonnes CODGEO sont différentes entre les deux fichiers.")
    # Afficher les différences
    codes_manquants_dans_2022 = codes_2017 - codes_2022
    codes_manquants_dans_2017 = codes_2022 - codes_2017

    if codes_manquants_dans_2022:
        print("❌ Codes présents en 2017 mais absents dans le data final :", codes_manquants_dans_2022)
    if codes_manquants_dans_2017:
        print("❌ Codes présents dans le data finale  mais absents en 2017 :", codes_manquants_dans_2017)

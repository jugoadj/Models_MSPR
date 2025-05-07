import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le fichier
df = pd.read_csv("donnees_socio_fusionnees.csv", sep=";")

# Vérifier les types de données
print(df.dtypes)

# Convertir les colonnes numériques en float (si nécessaire)
colonnes_a_convertir = [
    "chomeurs15-64ans", "nombres_artisant", "nombre_de-salarie", 
    "chomeurs15-24ans"
]

df[colonnes_a_convertir] = df[colonnes_a_convertir].apply(pd.to_numeric, errors="coerce")

# Supprimer les lignes avec valeurs manquantes
df_clean = df.dropna()

# Vérifier la propreté
print(df_clean.info())


sns.set(style="whitegrid")

moyennes = df_clean.groupby("année")[["chomeurs15-64ans", "chomeurs15-24ans"]].mean()

# Visualisation avec des barres
moyennes.plot(kind="bar", figsize=(10, 6), width=0.8)
plt.title("Évolution du chômage par année (Barres)")
plt.xlabel("Année")
plt.ylabel("Nombre moyen de chômeurs")
plt.xticks(rotation=0)
plt.legend(["Chômeurs 15-64 ans", "Chômeurs 15-24 ans"])
plt.tight_layout()
plt.show()

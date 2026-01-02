from sklearn.datasets import load_iris
import pandas as pd
import os
# Charger le dataset Iris intégré à scikit-learn
iris = load_iris()

# Convertir en DataFrame pandas pour plus de facilité
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])

# Créer le dossier data s'il n'existe pas

os.makedirs("../data", exist_ok=True)

# Sauvegarder en CSV
df.to_csv("../data/iris.csv", index=False)

print("Dataset Iris sauvegardé dans data/iris.csv")
print(df.head())
print("\nClasses :", iris.target_names)

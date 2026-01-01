import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Chemins
DATA_PATH = "data/iris.csv"
MODEL_PATH = "../artifacts/model.pkl"

# Créer le dossier pour les artefacts si nécessaire
os.makedirs("../artifacts", exist_ok=True)

# Charger les données
print("Chargement du dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop(["target", "target_name"], axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modèle baseline
model = LogisticRegression(max_iter=200)

print("Entraînement du modèle baseline...")
model.fit(X_train, y_train)

# Prédictions et métriques
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy sur le test set : {accuracy:.4f}")

# Démarrer un run MLflow
with mlflow.start_run(run_name="baseline_logistic_regression"):
    # Log des paramètres
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("max_iter", 200)

    # Log des métriques
    mlflow.log_metric("accuracy", accuracy)

    # Log du modèle
    mlflow.sklearn.log_model(model, "model")

    # Sauvegarder le modèle localement aussi (pour l'API plus tard)
    import joblib
    joblib.dump(model, MODEL_PATH)

    # Générer et logger une confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=df["target_name"].unique(),
                yticklabels=df["target_name"].unique())
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.title("Matrice de confusion - Baseline")

    cm_path = "../artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)
    print("Artefacts loggés : modèle + matrice de confusion")

print("Run MLflow terminé ")
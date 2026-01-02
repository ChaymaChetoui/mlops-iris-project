import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Arguments
parser = argparse.ArgumentParser(description="Entraînement Iris avec MLflow")
parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "svm"],
                    help="Type de modèle : logistic ou svm")
parser.add_argument("--C", type=float, default=1.0, help="Paramètre de régularisation (pour les deux modèles)")
parser.add_argument("--solver", type=str, default="lbfgs", help="Solver pour LogisticRegression")
parser.add_argument("--kernel", type=str, default="rbf", help="Kernel pour SVM")
parser.add_argument("--max_iter", type=int, default=200, help="Max iterations")
args = parser.parse_args()

# Chemins
DATA_PATH = "data/iris.csv"
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")

# Chargement données
df = pd.read_csv(DATA_PATH)
X = df.drop(["target", "target_name"], axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Choix du modèle
if args.model == "logistic":
    model = LogisticRegression(C=args.C, solver=args.solver, max_iter=args.max_iter)
    model_name = "LogisticRegression"
else:
    model = SVC(C=args.C, kernel=args.kernel)
    model_name = "SVM"

# Entraînement
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Modèle : {model_name}")
print(f"Accuracy : {accuracy:.4f}")

# MLflow tracking
with mlflow.start_run(run_name=f"{model_name}_C={args.C}"):
    # Params
    mlflow.log_param("model_type", model_name)
    mlflow.log_param("C", args.C)
    if args.model == "logistic":
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("max_iter", args.max_iter)
    else:
        mlflow.log_param("kernel", args.kernel)

    # Metrics
    mlflow.log_metric("accuracy", accuracy)

    # Modèle
    if args.model == "logistic":
        mlflow.sklearn.log_model(model, "model")
    else:
        mlflow.sklearn.log_model(model, "model")  # SVM aussi supporté

    # Sauvegarde locale
    joblib.dump(model, MODEL_PATH)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=df["target_name"].unique(),
                yticklabels=df["target_name"].unique())
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.title(f"Confusion Matrix - {model_name} (C={args.C})")
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

print("Run MLflow terminé et artefacts loggés.")

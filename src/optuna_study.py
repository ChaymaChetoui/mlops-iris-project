import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os

# Chargement données
df = pd.read_csv("data/iris.csv")
X = df.drop(["target", "target_name"], axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def objective(trial):
    model_type = trial.suggest_categorical("model_type", ["logistic", "svm"])

    if model_type == "logistic":
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "newton-cg", "liblinear"])
        model = LogisticRegression(C=C, solver=solver, max_iter=500)
    else:
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        model = SVC(C=C, kernel=kernel)

    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score

# Callback corrigé : create_child=True au lieu de nested=True
mlflow_callback = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name="accuracy"
)


# Lancement de l'étude
mlflow.set_experiment("optuna_iris_study")  # Pour regrouper proprement
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, callbacks=[mlflow_callback])

print(f"\nMeilleur score CV : {study.best_value:.4f}")
print("Meilleurs paramètres :", study.best_params)

# Log du meilleur modèle dans un run séparé
best_params = study.best_params.copy()
model_type = best_params.pop("model_type")

if model_type == "logistic":
    best_model = LogisticRegression(**best_params, max_iter=500)
else:
    best_model = SVC(**best_params)

best_model.fit(X_train, y_train)
test_accuracy = accuracy_score(y_test, best_model.predict(X_test))

with mlflow.start_run(run_name="optuna_best_model_final"):
    mlflow.log_params(best_params)
    mlflow.log_param("model_type", model_type)
    mlflow.log_metric("best_cv_accuracy", study.best_value)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.sklearn.log_model(best_model, "final_best_model")
    os.makedirs("../artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/optuna_best_model_final.pkl")

print(f"Accuracy test final : {test_accuracy:.4f}")
print("Étude terminée ! Relance mlflow ui → tu verras le run parent + 10 child runs dépliables.")
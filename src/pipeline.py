import os
from zenml import pipeline, step
from typing import Annotated, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


@step
def load_data() -> pd.DataFrame:
    """Charge le dataset Iris."""
    df = pd.read_csv("data/iris.csv")
    return df


@step
def split_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """Split en train/test."""
    X = df.drop(["target", "target_name"], axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logistic",
    C: float = 1.0
) -> Annotated[str, "model_path"]:
    """Entraîne le modèle et retourne le chemin sauvegardé."""
    if model_type == "logistic":
        model = LogisticRegression(C=C, max_iter=200)
    else:
        model = SVC(C=C)
    model.fit(X_train, y_train)
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/zenml_model.pkl"
    joblib.dump(model, model_path)
    return model_path


@step
def evaluate_model(
    model_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> float:
    """Évalue le modèle et retourne l'accuracy."""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy du pipeline ZenML : {accuracy:.4f}")
    return accuracy


@pipeline
def iris_pipeline(model_type: str = "logistic", C: float = 1.0):
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df=df)
    model_path = train_model(X_train=X_train, y_train=y_train, model_type=model_type, C=C)
    evaluate_model(model_path=model_path, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    print("=== Run Baseline LogisticRegression ===")
    iris_pipeline(model_type="logistic", C=1.0)

    print("\n=== Run SVM ===")
    iris_pipeline(model_type="svm", C=1.0)

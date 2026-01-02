from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from typing import List


app = FastAPI(title="Iris Prediction API")


class IrisFeatures(BaseModel):
    features: List[float]  # 4 valeurs : sepal_length, sepal_width, petal_length, petal_width


# Chargement du modèle au démarrage (version contrôlée par variable d'environnement)
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
model_path = f"artifacts/model_{MODEL_VERSION}.pkl"
model = joblib.load(model_path)


@app.get("/")
def home():
    return {"message": f"Iris API - Modèle chargé : {MODEL_VERSION}", "accuracy_hint": "v1: ~0.97, v2: 1.0"}


@app.post("/predict")
def predict(data: IrisFeatures):
    prediction = model.predict([data.features])[0]
    class_names = ["setosa", "versicolor", "virginica"]
    return {
        "prediction": class_names[prediction],
        "class_id": int(prediction),
        "model_version": MODEL_VERSION
    }

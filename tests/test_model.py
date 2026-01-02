from sklearn.linear_model import LogisticRegression
import joblib
import os

def test_model_can_predict():
    # Charge un modèle (v1 ou v2, peu importe)
    model_path = "artifacts/model_v1.pkl" if os.path.exists("artifacts/model_v1.pkl") else "artifacts/model_v2.pkl"
    model = joblib.load(model_path)
    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert prediction[0] in [0, 1, 2]  # class_id valide
    print("Test modèle OK")
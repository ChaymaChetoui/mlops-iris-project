import gradio as gr
import joblib
import numpy as np

# Charger le modèle entraîné (le même que dans ton entraînement)
model = joblib.load("artifacts/model.pkl")  # ou le chemin où tu l'as sauvegardé

# Fonction de prédiction
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    # Préparer les données d'entrée
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Prédiction
    prediction = model.predict(features)[0]
    
    # Mapper la classe numérique vers le nom
    class_names = ["setosa", "versicolor", "virginica"]
    predicted_class = class_names[prediction]
    
    return f"Prédiction : **{predicted_class}** 🌸"

# Interface Gradio
interface = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.1, step=0.1, label="Longueur du sépale (cm)"),
        gr.Slider(2.0, 4.5, value=3.5, step=0.1, label="Largeur du sépale (cm)"),
        gr.Slider(0.1, 7.0, value=1.4, step=0.1, label="Longueur du pétale (cm)"),
        gr.Slider(0.1, 2.5, value=0.2, step=0.1, label="Largeur du pétale (cm)"),
    ],
    outputs="text",
    title="Prédicteur Iris 🌷",
    description="Entrez les mesures des fleurs Iris et obtenez la prédiction ! Modèle entraîné avec scikit-learn.",
    theme="default",  # ou "huggingface" pour un look moderne
    examples=[
        [5.1, 3.5, 1.4, 0.2],  # setosa
        [6.4, 3.2, 4.5, 1.5],  # versicolor
        [7.7, 3.8, 6.7, 2.2],  # virginica
    ],
)

# Lancer l'interface
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)  # Important pour EC2
Projet MLOps complet sur le dataset Iris (classification multi-classe) implémentant un workflow de bout en bout avec les outils suivants :

- Git (branches `dev`/`main`, tags v1/v2)
- DVC (versioning des données)
- MLflow (suivi d’expériences)
- ZenML (pipeline MLOps)
- Optuna (optimisation d’hyperparamètres)
- FastAPI + Docker + Docker Compose (déploiement d’API d’inférence)
- GitHub Actions (CI/CD bonus)

## Structure du projet
mlops-iris-project/
├── .github/
│   └── workflows/
│       └── ci.yml                          # CI/CD GitHub Actions (lint, test, build & push image, smoke test)
├── .dvc/                                       # Configuration DVC
├── artifacts/                                  # Modèles sauvegardés
│   ├── model_v1.pkl                            # Baseline LogisticRegression
│   ├── model_v2.pkl                            # Meilleur modèle Optuna (SVM, accuracy 1.0)
│   └── optuna_best_model_final.pkl
├── data/
│   └── iris.csv                                # Dataset Iris (versionné avec DVC)
├── src/
│   ├── app.py                                  # API FastAPI d’inférence (/predict)
│   ├── optuna_study.py                         # Optimisation Optuna + logging MLflow
│   ├── pipeline.py                             # Pipeline ZenML (load → split → train → evaluate)
│   ├── prepare_data.py                         # Préparation initiale des données
│   └── train.py                                # Entraînement avec MLflow (baseline + variations)
├── tests/
│   └── test_model.py                           # Tests unitaires
├── .gitignore
├── Dockerfile                                  # Conteneurisation de l’API
├── docker-compose.yml                          # Lancement API avec switch v1/v2
├── requirements.txt
└── README.md
text## Installation et exécution

### 1. Cloner le repository et récupérer les données

```bash
git clone https://github.com/ChaymaChetoui/mlops-iris-project.git
cd mlops-iris-project
dvc pull                                        # Récupère iris.csv depuis le remote DVC
2. Créer l’environnement virtuel
Bashpython -m venv venv

# Activation
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
3. Suivi d’expériences avec MLflow
Bashmlflow ui
Ouvrir http://localhost:5000
Contenu visible :

Run baseline LogisticRegression
Variations manuelles (différents hyperparamètres C, modèles SVM/Logistic)
Étude Optuna (10 trials) avec child runs
Meilleur modèle loggé (accuracy 1.0 sur test set)

4. Pipeline ZenML
Bashpython src/pipeline.py
zenml up --blocking
Ouvrir http://localhost:8237
Contenu visible :

Pipeline iris_pipeline avec DAG complet :
load_data → split_data → train_model → evaluate_model
Plusieurs runs (LogisticRegression vs SVM)
Artefacts produits (model_path dans artifacts/)

5. Optimisation avec Optuna
Bashpython src/optuna_study.py
mlflow ui
→ Étude de 10 trials, logging des child runs dans MLflow, meilleur modèle sauvegardé.
6. Déploiement de l’API d’inférence
Bashdocker-compose up --build
L’API est disponible sur http://localhost:8000
Tests d’inférence
Bash# Accueil
curl http://localhost:8000/

# Prédiction setosa
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Prédiction versicolor
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [6.3, 3.3, 4.7, 1.6]}'

# Prédiction virginica
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [6.7, 3.0, 5.2, 2.3]}'
Switch de version modèle (v1 → v2 → rollback)
Modifier la variable MODEL_VERSION dans docker-compose.yml :

v1 : modèle baseline LogisticRegression
v2 : meilleur modèle Optuna (SVM, accuracy 1.0)

Relancer :
Bashdocker-compose up --build
Démonstration réalisée :

Chargement et test de v1
Mise à jour vers v2 (meilleur modèle)
Rollback vers v1

7. CI/CD avec GitHub Actions (bonus)
Voir l’onglet Actions sur GitHub.
Workflow .github/workflows/ci.yml :

Lint (flake8)
Tests unitaires (pytest)
Build et push de l’image Docker vers GitHub Container Registry (ghcr.io)
Smoke test planifié quotidien

Pipeline déclenché à chaque push sur dev et main.

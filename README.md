Projet MLOps - Classification Iris

## Description
Projet MLOps complet sur le dataset Iris (classification multi-classe) utilisant :
- Git pour la gestion du code
- DVC pour le versioning des données
- MLflow pour le suivi d'expériences
- ZenML pour les pipelines
- Optuna pour l'optimisation d'hyperparamètres
- Docker / Docker Compose pour la conteneurisation
- FastAPI pour l'API d'inférence
- GitLab CI pour CI/CD (optionnel)

## Structure du projet

mlops-iris-project/
├── data/                # Données (trackées par DVC)
├── src/                 # Code source
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── ...  
 
## Installation et exécution rapide (locale)

1. Cloner le repo
```bash
git clone https://gitlab.com/ton-username/mlops-iris-project.git
cd mlops-iris-project
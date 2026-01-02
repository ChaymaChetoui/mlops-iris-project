FROM python:3.10-slim

WORKDIR /app

# Copie des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code et artefacts
COPY src/ ./src
COPY artifacts/ ./artifacts

# Expose port
EXPOSE 8000

# Commande
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
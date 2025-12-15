FROM python:3.11-slim

# Empêche Python d'écrire des .pyc et bufferiser la sortie
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installer dépendances
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier le code
COPY . /app

# Si ton webapp.py écoute sur 8000 (à adapter sinon)
EXPOSE 8000

# Commande de démarrage
CMD ["python", "webapp.py"]

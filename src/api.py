from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# 1. Initialisation de l'API
app = FastAPI(title="Finance ML Predictor API", description="API ML Engineer pour prédictions boursières")

# 2. Chargement du modèle entraîné au démarrage du serveur
try:
    with open('models/finance_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception("Modèle introuvable. Avez-vous bien lancé model_training.py ?")

# 3. Définition du format de la donnée d'entrée (Typage fort)
class MarketData(BaseModel):
    Daily_Return: float
    SMA_10: float
    SMA_30: float
    Volatility_10: float

@app.get("/")
def read_root():
    return {"message": "API Opérationnelle. Accédez à http://127.0.0.1:8000/docs pour l'interface de test."}

# 4. Point d'accès (Endpoint) pour la prédiction
@app.post("/predict")
def predict_market(data: MarketData):
    # Transformation des données reçues en tableau Pandas pour l'IA
    input_data = pd.DataFrame([data.model_dump()])
    
    # Prédiction de la classe (0 ou 1) et de la probabilité
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1] # Probabilité de la classe 1 (Hausse)
    
    # Réponse renvoyée par l'API
    return {
        "prediction_brute": int(prediction),
        "recommandation": "ACHETER / GARDER" if prediction == 1 else "VENDRE",
        "probabilite_hausse": f"{proba * 100:.2f}%"
    }
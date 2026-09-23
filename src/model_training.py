import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle
import os

def train_model():
    print("1. Chargement des données d'entraînement...")
    df = pd.read_csv('data/processed_market_data.csv', index_col=0)
    
    # 2. Séparation des caractéristiques (X) et de la cible (y)
    # On retire le Ticker et la Target pour ne garder que les données mathématiques
    features = ['Daily_Return', 'SMA_10', 'SMA_30', 'Volatility_10']
    X = df[features]
    y = df['Target']
    
    print(f"2. Séparation Train / Test (80% / 20%)...")
    # On garde 20% des données (non vues par l'IA) pour la tester à la fin
    # shuffle=False est OBLIGATOIRE en finance : on ne mélange pas les dates de l'histoire !
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    
    print(f"   -> Taille d'entraînement (Train) : {len(X_train)} lignes")
    print(f"   -> Taille de test (Test) : {len(X_test)} lignes")
    
    print("\n3. Entraînement du modèle (Random Forest) en cours...")
    # On paramètre le modèle pour éviter le sur-apprentissage (overfitting)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    # C'est ici que l'IA apprend !
    model.fit(X_train, y_train)
    print("   -> Entraînement terminé !")
    
    print("\n4. Évaluation des performances...")
    # On demande au modèle de prédire sur les 20% de données qu'il n'a jamais vues
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probabilité (pour la métrique AUC)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("="*40)
    print(" RÉSULTATS DU MODÈLE ")
    print("="*40)
    print(f"Précision Globale (Accuracy) : {accuracy * 100:.2f}%")
    print(f"Score AUC-ROC                : {auc:.4f}")
    print("\nRapport détaillé :")
    print(classification_report(y_test, y_pred))
    print("="*40)
    
    # 5. Sauvegarde du modèle entraîné pour l'API
    os.makedirs('models', exist_ok=True)
    with open('models/finance_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("\n5. Succès ! Modèle sauvegardé dans : models/finance_model.pkl")

if __name__ == "__main__":
    train_model()
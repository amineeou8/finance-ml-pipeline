import pandas as pd
import warnings
# On ignore certains avertissements sans importance de Pandas
warnings.filterwarnings('ignore')

def preprocess_and_engineer_features():
    print("1. Chargement des données brutes...")
    df = pd.read_csv('data/raw_market_data.csv', index_col=0, parse_dates=True)
    
    # yfinance a généré beaucoup de colonnes. On va identifier la vraie colonne de prix 'Close'
    # S'il y a un multi-index ou des noms étranges, on cherche celle qui contient 'Close'
    close_col = [col for col in df.columns if 'Close' in col][0]
    
    processed_data = []
    tickers = df['Ticker'].unique()
    
    print(f"2. Création des indicateurs mathématiques pour {len(tickers)} entreprises...")
    
    for ticker in tickers:
        # On isole les données d'une seule entreprise
        df_ticker = df[df['Ticker'] == ticker].copy()
        df_ticker.sort_index(inplace=True) # Tri par date obligatoire en finance
        
        # --- FEATURE ENGINEERING ---
        # 1. Rendement journalier (Variation en pourcentage d'un jour à l'autre)
        df_ticker['Daily_Return'] = df_ticker[close_col].pct_change()
        
        # 2. Moyennes Mobiles (Simple Moving Averages) - Lisse la courbe des prix
        df_ticker['SMA_10'] = df_ticker[close_col].rolling(window=10).mean()
        df_ticker['SMA_30'] = df_ticker[close_col].rolling(window=30).mean()
        
        # 3. Volatilité (Écart-type des rendements sur 10 jours)
        df_ticker['Volatility_10'] = df_ticker['Daily_Return'].rolling(window=10).std()
        
        # --- CREATION DE LA CIBLE (TARGET) ---
        # Le modèle doit prédire l'avenir. On regarde donc le prix de DEMAIN (shift(-1)).
        # Si le prix de demain est STRICTEMENT SUPERIEUR au prix d'aujourd'hui = 1 (Achat)
        # Sinon = 0 (Vente/Garder)
        df_ticker['Target'] = (df_ticker[close_col].shift(-1) > df_ticker[close_col]).astype(int)
        
        processed_data.append(df_ticker)
        
    # On rassemble tout dans un seul grand tableau
    final_df = pd.concat(processed_data)
    
    print("3. Nettoyage des valeurs manquantes (Handling missing values)...")
    print(f"   -> Lignes avant nettoyage : {len(final_df)}")
    
    # La création des moyennes mobiles génère des "NaN" (les 30 premiers jours n'ont pas de moyenne sur 30 jours)
    # Le "shift(-1)" génère un NaN pour le tout dernier jour téléchargé (car on ne connaît pas le prix de demain)
    final_df.dropna(inplace=True)
    
    print(f"   -> Lignes après nettoyage : {len(final_df)}")
    
    # On ne garde que les colonnes utiles pour notre modèle IA
    cols_to_keep = ['Ticker', close_col, 'Daily_Return', 'SMA_10', 'SMA_30', 'Volatility_10', 'Target']
    final_df = final_df[cols_to_keep]
    
    # Sauvegarde des données prêtes pour l'entraînement
    filepath = 'data/processed_market_data.csv'
    final_df.to_csv(filepath)
    print(f"4. Succès ! Données propres sauvegardées dans : {filepath}")

if __name__ == "__main__":
    preprocess_and_engineer_features()
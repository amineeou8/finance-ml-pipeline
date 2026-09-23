import yfinance as yf
import pandas as pd
import os

def download_financial_data():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT', 'JNJ', 'PG', 'MA', 'UNH', 'HD']
    
    print("Début du téléchargement global (méthode optimisée)...")
    
    # On télécharge tout en une seule requête (plus rapide)
    df = yf.download(tickers, start="2010-01-01", end="2026-09-01", progress=False)
    
    # L'astuce "Data Engineering" : on aplatit le multi-index (les 76 colonnes)
    # pour obtenir une structure propre : Date | Ticker | Close | Open | etc.
    df = df.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    
    # Sauvegarde propre
    os.makedirs('data', exist_ok=True)
    filepath = 'data/raw_market_data.csv'
    df.to_csv(filepath, index=False)
    
    print("\n" + "="*30)
    print(" RÉSUMÉ DE L'INGESTION CORRIGÉE ")
    print("="*30)
    print(f"Fichier sauvegardé : {filepath}")
    print(f"Nouveau nombre de lignes : {len(df):,}")
    print(f"Nouvelles colonnes propres : {list(df.columns)}")
    print("="*30)

if __name__ == "__main__":
    download_financial_data()
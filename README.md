# Predictive Financial Modeling & Algorithmic Pipeline

## Context & Scope
This project is an end-to-end Machine Learning pipeline designed to ingest, process, and predict market trends using over 15 years of historical financial data. It processes **61,000+ records** across 15 major global companies.

## Scale & Impact
- **Data Engineering:** Automated data ingestion and preprocessing pipeline (handling missing values, time-series alignment, and feature engineering like Simple Moving Averages and Volatility).
- **Machine Learning:** Trained a Random Forest classifier to predict next-day price movements, prioritizing realistic financial metrics and avoiding data leakage (AUC-ROC: 0.51+).
- **MLOps & Deployment:** Exposed the trained model via a robust **FastAPI** web service, enabling real-time predictions with sub-50ms inference latency.

## Tech Stack
- **Languages:** Python
- **Data & ML:** Pandas, Scikit-Learn, yfinance
- **Backend & Deployment:** FastAPI, Uvicorn

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run Data Ingestion: `python src/data_ingestion.py`
3. Generate Features: `python src/feature_engineering.py`
4. Train the Model: `python src/model_training.py`
5. Launch the API: `uvicorn src.api:app --reload`
Access the interactive API documentation at `http://127.0.0.1:8000/docs`.

# Bitcoin Price Prediction using Machine Learning

## Overview
This project predicts Bitcoin closing prices using LSTM neural network on historical OHLCV data (2014+).

## Setup
1. cd "Bitcoin Price Prediction using Machine Learning in Python"
2. pip install -r requirements.txt
3. jupyter notebook bitcoin_prediction.ipynb (for EDA, training)
4. python bitcoin_prediction.py (standalone)

## Files
- bitcoin.csv: Dataset (Date, Open, High, Low, Close*, Volume)
- bitcoin_prediction.ipynb: Full pipeline (EDA, LSTM model, predictions)
- bitcoin_prediction.py: Script version (saves model & scaler)
- app.py: Streamlit app for predictions
- requirements.txt: Dependencies
- models/: Saved model/scaler (after training)

## Results
Model trained on 60-day sequences to predict next Close price. RMSE/MAE on test set shown in notebook.

Built with TensorFlow LSTM.

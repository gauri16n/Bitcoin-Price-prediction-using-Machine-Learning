@echo off
cd /d "%~dp0"
echo Training LSTM model...
python bitcoin_prediction.py
pause

cd "Bitcoin Price Prediction using Machine Learning in Python"
python bitcoin_prediction.py

python -m streamlit run app.py

@echo off
echo Starting Bitcoin Prediction App...
cd /d "%~dp0"
python bitcoin_prediction.py
timeout /t 5 /nobreak
streamlit run app.py
pause

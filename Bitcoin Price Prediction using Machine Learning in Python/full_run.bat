@echo off
cd /d "%~dp0"
if not exist "models" mkdir models
echo Installing dependencies...
python -m pip install -r requirements.txt && echo Dependencies installed successfully! || (echo ERROR: Failed to install dependencies! & pause & exit /b 1)
echo Training LSTM model...
python bitcoin_prediction.py && echo Model trained successfully! || (echo ERROR: Model training failed! & pause & exit /b 1)
echo Model trained successfully!
echo Do you want to launch Streamlit app? (y/n)
set /p launch=Enter choice: 
if /i "%launch%"=="y" (
    echo Launching Streamlit app...
    python -m streamlit run app.py
) else (
    echo Skipped Streamlit launch.
)
pause

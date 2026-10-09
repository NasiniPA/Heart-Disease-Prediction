@echo off
title CardioPredict ML Launcher
echo ======================================================================
echo          CardioPredict ML - Heart Disease Prediction Launcher
echo ======================================================================
echo.

set PYCMD=python
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Neither 'python' nor 'py' was found in your system PATH!
        echo Please install Python 3.9+ from https://www.python.org/
        echo Remember to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    ) else (
        set PYCMD=py
    )
)

echo Using Python interpreter: %PYCMD%
echo.
echo [1/3] Installing/verifying dependencies from requirements.txt...
%PYCMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some dependencies could not be installed automatically.
)

echo.
echo [2/3] Checking if ML model exists...
if not exist "model.pkl" (
    echo Training model on Cleveland dataset (train_model.py)...
    %PYCMD% train_model.py
)

echo.
echo [3/3] Launching Streamlit web application on browser...
echo.
%PYCMD% -m streamlit run app.py
pause

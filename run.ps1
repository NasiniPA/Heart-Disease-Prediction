Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "         CardioPredict ML - Heart Disease Prediction Launcher" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$pythonCmd = "python"
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        $pythonCmd = "py"
    } else {
        Write-Host "[ERROR] Neither 'python' nor 'py' was found on your system." -ForegroundColor Red
        Write-Host "Please install Python 3.9+ from https://www.python.org/ and check 'Add to PATH'." -ForegroundColor Yellow
        pause
        exit 1
    }
}

Write-Host "Using Python: $pythonCmd" -ForegroundColor Green
Write-Host "[1/3] Installing/verifying dependencies..." -ForegroundColor Yellow
& $pythonCmd -m pip install -r requirements.txt

if (-not (Test-Path "model.pkl")) {
    Write-Host "[2/3] Training model on Cleveland dataset..." -ForegroundColor Yellow
    & $pythonCmd train_model.py
}

Write-Host "[3/3] Launching Streamlit web application..." -ForegroundColor Green
& $pythonCmd -m streamlit run app.py

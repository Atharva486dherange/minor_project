@echo off
echo ===============================================================
echo  GigaChad AI Backend — Setup Script (Windows)
echo  100% Free. No API Keys. No Internet Required.
echo ===============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Download from https://python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo.
echo [2/5] Installing Python packages (this may take 5-10 mins on first run)...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo [3/5] Downloading spaCy English model...
python -m spacy download en_core_web_sm

echo.
echo [4/5] Checking Tesseract OCR...
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Tesseract not found. OCR features will be disabled.
    echo           Install from: https://github.com/UB-Mannheim/tesseract/wiki
    echo           Then add to PATH and re-run this script.
) else (
    echo Tesseract found ✓
)

echo.
echo [5/5] Checking FFmpeg (needed for audio/video)...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] FFmpeg not found. Audio/video transcription will not work.
    echo           Install from: https://ffmpeg.org/download.html
    echo           Or run: winget install ffmpeg
) else (
    echo FFmpeg found ✓
)

echo.
echo ===============================================================
echo  Setup Complete!
echo  To start the backend server, run: start_server.bat
echo ===============================================================
pause

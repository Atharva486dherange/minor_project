@echo off
echo ===============================================================
echo  GigaChad AI Backend — Auto-Install and Start
echo ===============================================================
echo.

cd /d "%~dp0"

echo [1/3] Installing required packages...
pip install fastapi uvicorn[standard] python-multipart aiofiles sqlalchemy aiosqlite greenlet pydantic yake pdfplumber Pillow pytesseract httpx python-dotenv

echo.
echo [2/3] Checking if spaCy model is installed...
python -m spacy info en_core_web_sm >nul 2>&1
if errorlevel 1 (
    echo Downloading spaCy English model...
    python -m spacy download en_core_web_sm
) else (
    echo spaCy model already installed.
)

echo.
echo [3/3] Starting server...
echo.
echo  API Docs: http://localhost:8000/docs
echo  Frontend: open index.html in browser
echo.
python main.py
pause

# GigaChad AI — Meeting, Document & Notes Summarizer ⚡

An offline-capable, AI-powered meeting transcription and document summarization workspace built for college projects, lectures, and professional meetings.

---

## 🌟 Key Features

- **Meeting & Audio/Video Summarizer:** Transcribe recordings using OpenAI Whisper (local execution) into timestamped dialogue, speaker turns, action items, and executive summaries.
- **Explain with AI:** Instant student-friendly (ELI5) concept explainer next to each transcript sentence to define technical jargon and why it matters.
- **PDF & Document Summarizer:** Extract and summarize lecture slides, syllabi, research papers, and PDFs without character limits.
- **Handwritten Notes (OCR):** Convert photos of handwritten notes or whiteboards into structured digital text with Tesseract OCR.
- **Interactive AI Chat:** Extractive Q&A to ask direct questions about any uploaded lecture or meeting.
- **100% Free & Private:** No paid API keys required; runs locally on your PC.

---

## 🛠️ Tech Stack

- **Frontend:** Single-page React 18 application, Vanilla CSS Design System with dark and light theme toggle.
- **Backend:** Python FastAPI, Uvicorn, SQLite (SQLAlchemy async).
- **NLP & AI Engine:**
  - Speech-to-Text: OpenAI Whisper
  - Summarization: Transformer-based ranking & Map-Reduce
  - Entity & Action Extraction: spaCy (`en_core_web_sm`)
  - Keyword Analysis: YAKE
  - Document & Image Parsing: `pdfplumber`, `pytesseract`, Pillow

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.9+**
- (Optional for media/OCR) [FFmpeg](https://ffmpeg.org/) and [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Setup Backend
```bash
cd backend
setup.bat
```
Or manually:
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Start Backend Server
```bash
cd backend
start_server.bat
```
* API Server runs at `http://localhost:8000`
* Swagger Interactive Docs: `http://localhost:8000/docs`

### 4. Run Frontend
Simply open `index.html` in your web browser (Chrome, Edge, Firefox, Brave).

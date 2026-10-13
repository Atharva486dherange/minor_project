"""
GigaChad AI — FastAPI Main Server (Lightweight Startup Version)
Heavy AI models (Whisper, BART, etc.) load lazily when first used.
Server starts in ~2 seconds.

Run with: python -m uvicorn main:app --reload --port 8000
     OR:  python main.py
Swagger UI: http://localhost:8000/docs
"""

import os
import uuid
import json
import logging
import aiofiles
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import (
    init_db, get_db,
    Session as DBSession, Transcript, TranscriptSegment,
    Summary, ActionItem
)

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gigachad")

# ── Temp upload directory ──────────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── FastAPI App ────────────────────────────────────────────────────
app = FastAPI(
    title="GigaChad AI Backend",
    description=(
        "100% Free & Open-Source NLP Backend\n"
        "for Meeting & Class Notes Summarizer.\n"
        "No API keys. No internet required. Fully local."
    ),
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("=" * 55)
    logger.info("  GigaChad AI Backend is RUNNING")
    logger.info("  http://localhost:8000")
    logger.info("  API Docs: http://localhost:8000/docs")
    logger.info("=" * 55)


# ── Lazy NLP loader ────────────────────────────────────────────────
# Models only load when first endpoint is called — startup is instant
_nlp_engine = None

def get_nlp():
    global _nlp_engine
    if _nlp_engine is None:
        try:
            from nlp_engine import (
                transcriber, summarizer, action_extractor,
                sentiment_analyzer, qa_engine, keyword_extractor,
                ocr_engine, pdf_extractor
            )
            _nlp_engine = {
                "transcriber":        transcriber,
                "summarizer":         summarizer,
                "action_extractor":   action_extractor,
                "sentiment_analyzer": sentiment_analyzer,
                "qa_engine":          qa_engine,
                "keyword_extractor":  keyword_extractor,
                "ocr_engine":         ocr_engine,
                "pdf_extractor":      pdf_extractor,
            }
            logger.info("NLP Engine loaded ✓")
        except ImportError as e:
            logger.warning(f"NLP Engine not fully available: {e}")
            logger.warning("Install: pip install transformers torch openai-whisper spacy")
            _nlp_engine = {}
    return _nlp_engine


# ══════════════════════════════════════════════════════════════════
#   PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════

class TextInput(BaseModel):
    text: str
    session_id: Optional[str] = None

class QAInput(BaseModel):
    question: str
    context: str

class ExplainInput(BaseModel):
    sentence: str
    context: str

class SentimentInput(BaseModel):
    segments: List[dict]


# ══════════════════════════════════════════════════════════════════
#   HEALTH CHECK
# ══════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    """Server info and endpoint list"""
    return {
        "status":  "running ✓",
        "app":     "GigaChad AI Backend",
        "version": "1.0.0",
        "docs":    "http://localhost:8000/docs",
        "endpoints": {
            "POST /transcribe":    "Audio/Video → Transcript + timestamps",
            "POST /summarize":     "Text → TL;DR bullets + executive summary",
            "POST /actions":       "Text → Action items with owner & deadline",
            "POST /sentiment":     "Segments → Sentiment per segment",
            "POST /chat":          "Question + context → Answer (Ask AI)",
            "POST /explain":       "Sentence → Plain-English explanation",
            "POST /extract-pdf":   "PDF file → Extracted text",
            "POST /ocr":           "Image → Text (handwritten notes)",
            "POST /keywords":      "Text → Keywords for word cloud",
            "POST /full-pipeline": "Any file → Complete NLP analysis",
            "GET  /sessions":      "Session history from database",
        }
    }

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ══════════════════════════════════════════════════════════════════
#   1. TRANSCRIBE
# ══════════════════════════════════════════════════════════════════

@app.post("/transcribe", tags=["NLP"])
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Upload audio/video → Get timestamped transcript with sentiment"""
    nlp = get_nlp()
    if "transcriber" not in nlp:
        raise HTTPException(503, "Whisper not installed. Run: pip install openai-whisper")

    allowed = {".mp3", ".mp4", ".wav", ".webm", ".m4a", ".ogg", ".flac"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported type: {ext}. Allowed: {allowed}")

    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        result    = nlp["transcriber"].transcribe(tmp_path, language=language)
        segments  = nlp["sentiment_analyzer"].analyze(result["segments"])
        session_id = str(uuid.uuid4())

        # Save to DB
        db_session = DBSession(
            id=session_id, title=file.filename,
            type="meeting", status="done",
            completed_at=datetime.utcnow()
        )
        db.add(db_session)
        transcript_row = Transcript(
            session_id=session_id,
            full_text=result["text"],
            language=result["language"],
            word_count=len(result["text"].split()),
        )
        db.add(transcript_row)
        await db.flush()
        for seg in segments:
            db.add(TranscriptSegment(
                transcript_id=transcript_row.id,
                speaker="Unknown",
                start_time=seg.get("start", 0),
                end_time=seg.get("end", 0),
                text=seg["text"],
                sentiment=seg.get("sentiment"),
                sentiment_score=seg.get("sentiment_score"),
            ))
        await db.commit()

        return {
            "session_id": session_id,
            "text":       result["text"],
            "segments":   segments,
            "language":   result["language"],
            "word_count": len(result["text"].split()),
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)  # Privacy: delete after processing


# ══════════════════════════════════════════════════════════════════
#   2. SUMMARIZE
# ══════════════════════════════════════════════════════════════════

@app.post("/summarize", tags=["NLP"])
async def summarize_text(
    body: TextInput,
    db: AsyncSession = Depends(get_db),
):
    """Text → TL;DR bullets + executive summary. Uses BART with Map-Reduce."""
    nlp = get_nlp()
    if "summarizer" not in nlp:
        raise HTTPException(503, "BART not installed. Run: pip install transformers torch")
    if len(body.text.strip()) < 50:
        raise HTTPException(400, "Text too short (need at least 50 chars)")

    result   = nlp["summarizer"].summarize(body.text)
    keywords = nlp["keyword_extractor"].extract(body.text)

    if body.session_id:
        db.add(Summary(
            session_id=body.session_id,
            executive=result["executive"],
            bullet_points=json.dumps(result["bullets"]),
            keywords=json.dumps(keywords),
        ))
        await db.commit()

    return {
        "executive":        result["executive"],
        "bullets":          result["bullets"],
        "anchored_notices": result["anchored_notices"],
        "keywords":         keywords,
        "chunk_count":      result["chunk_count"],
    }


# ══════════════════════════════════════════════════════════════════
#   3. ACTION ITEMS
# ══════════════════════════════════════════════════════════════════

@app.post("/actions", tags=["NLP"])
async def extract_actions(
    body: TextInput,
    db: AsyncSession = Depends(get_db),
):
    """Extract action items, homework, tasks with owner & deadline."""
    nlp = get_nlp()
    if "action_extractor" not in nlp:
        raise HTTPException(503, "spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")

    actions = nlp["action_extractor"].extract(body.text)

    if body.session_id:
        for a in actions:
            db.add(ActionItem(
                session_id=body.session_id,
                task=a["task"],
                assignee=a.get("assignee"),
                deadline=a.get("deadline"),
                priority=a.get("priority", "medium"),
                source_text=a.get("source"),
            ))
        await db.commit()

    return {"actions": actions, "count": len(actions)}


# ══════════════════════════════════════════════════════════════════
#   4. SENTIMENT
# ══════════════════════════════════════════════════════════════════

@app.post("/sentiment", tags=["NLP"])
async def analyze_sentiment(body: SentimentInput):
    """Sentiment analysis per segment. Uses DistilBERT."""
    nlp = get_nlp()
    if "sentiment_analyzer" not in nlp:
        raise HTTPException(503, "transformers not installed")
    return {"segments": nlp["sentiment_analyzer"].analyze(body.segments)}


# ══════════════════════════════════════════════════════════════════
#   5. CHAT / Q&A
# ══════════════════════════════════════════════════════════════════

@app.post("/chat", tags=["NLP"])
async def chat_with_transcript(body: QAInput):
    """Ask AI anything about a transcript. Uses RoBERTa Q&A."""
    nlp = get_nlp()
    if "qa_engine" not in nlp:
        raise HTTPException(503, "transformers not installed")
    if not body.context.strip():
        raise HTTPException(400, "context is required")
    if not body.question.strip():
        raise HTTPException(400, "question is required")
    return nlp["qa_engine"].answer(body.question, body.context)


# ══════════════════════════════════════════════════════════════════
#   6. EXPLAIN THIS POINT
# ══════════════════════════════════════════════════════════════════

@app.post("/explain", tags=["NLP"])
async def explain_point(body: ExplainInput):
    """
    Click any confusing sentence → get plain-English explanation,
    key terms defined, and why it matters.
    """
    nlp = get_nlp()
    if "qa_engine" not in nlp:
        raise HTTPException(503, "transformers not installed")
    if not body.sentence.strip():
        raise HTTPException(400, "sentence is required")
    return nlp["qa_engine"].explain_point(body.sentence, body.context)


# ══════════════════════════════════════════════════════════════════
#   7. EXTRACT PDF
# ══════════════════════════════════════════════════════════════════

@app.post("/extract-pdf", tags=["NLP"])
async def extract_pdf(file: UploadFile = File(...)):
    """Upload PDF → get extracted text, page count, word count."""
    nlp = get_nlp()
    if "pdf_extractor" not in nlp:
        raise HTTPException(503, "pdfplumber not installed")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.pdf")
    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            await f.write(await file.read())
        return nlp["pdf_extractor"].extract(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ══════════════════════════════════════════════════════════════════
#   8. OCR
# ══════════════════════════════════════════════════════════════════

@app.post("/ocr", tags=["NLP"])
async def ocr_image(file: UploadFile = File(...)):
    """Upload photo of handwritten notes/blackboard → extracted text."""
    nlp = get_nlp()
    if "ocr_engine" not in nlp:
        raise HTTPException(503, "pytesseract not installed")

    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported image format: {ext}")

    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            await f.write(await file.read())
        return nlp["ocr_engine"].extract_text(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ══════════════════════════════════════════════════════════════════
#   9. KEYWORDS
# ══════════════════════════════════════════════════════════════════

@app.post("/keywords", tags=["NLP"])
async def extract_keywords(body: TextInput):
    """Text → top keywords for word cloud. Uses YAKE."""
    nlp = get_nlp()
    if "keyword_extractor" not in nlp:
        raise HTTPException(503, "yake not installed")
    return {"keywords": nlp["keyword_extractor"].extract(body.text)}


# ══════════════════════════════════════════════════════════════════
#   10. FULL PIPELINE
# ══════════════════════════════════════════════════════════════════

@app.post("/full-pipeline", tags=["NLP"])
async def full_pipeline(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    One upload → complete analysis:
    Audio/Video: transcribe → summarize → actions → sentiment → keywords
    PDF:         extract → summarize → actions → keywords
    Image:       OCR → summarize → actions → keywords
    """
    nlp = get_nlp()
    ext = os.path.splitext(file.filename)[1].lower()
    audio_exts = {".mp3", ".mp4", ".wav", ".webm", ".m4a", ".ogg"}
    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    session_id = str(uuid.uuid4())

    tmp_path = os.path.join(UPLOAD_DIR, f"{session_id}{ext}")
    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            await f.write(await file.read())

        raw_text, segments, pipeline_type = "", [], "unknown"

        if ext in audio_exts:
            pipeline_type = "audio"
            tr = nlp["transcriber"].transcribe(tmp_path)
            raw_text = tr["text"]
            segments = nlp["sentiment_analyzer"].analyze(tr["segments"])

        elif ext == ".pdf":
            pipeline_type = "pdf"
            raw_text = nlp["pdf_extractor"].extract(tmp_path)["full_text"]

        elif ext in image_exts:
            pipeline_type = "image"
            raw_text = nlp["ocr_engine"].extract_text(tmp_path)["text"]

        else:
            raise HTTPException(400, f"Unsupported file type: {ext}")

        if not raw_text.strip():
            raise HTTPException(422, "Could not extract text from file")

        summary  = nlp["summarizer"].summarize(raw_text)
        actions  = nlp["action_extractor"].extract(raw_text)
        keywords = nlp["keyword_extractor"].extract(raw_text)

        # Save to DB
        db.add(DBSession(id=session_id, title=file.filename,
                         type=pipeline_type, status="done",
                         completed_at=datetime.utcnow()))
        db.add(Summary(session_id=session_id,
                       executive=summary["executive"],
                       bullet_points=json.dumps(summary["bullets"]),
                       keywords=json.dumps(keywords)))
        for a in actions:
            db.add(ActionItem(session_id=session_id, task=a["task"],
                              assignee=a.get("assignee"),
                              deadline=a.get("deadline"),
                              priority=a.get("priority", "medium")))
        await db.commit()

        return {
            "session_id":    session_id,
            "pipeline_type": pipeline_type,
            "file_name":     file.filename,
            "transcript":    {"text": raw_text, "segments": segments},
            "summary": {
                "executive":        summary["executive"],
                "bullets":          summary["bullets"],
                "anchored_notices": summary["anchored_notices"],
            },
            "actions":  actions,
            "keywords": keywords,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ══════════════════════════════════════════════════════════════════
#   DATABASE — Session History (the Vault)
# ══════════════════════════════════════════════════════════════════

@app.get("/sessions", tags=["Database"])
async def list_sessions(limit: int = 20, db: AsyncSession = Depends(get_db)):
    """Get recent sessions from DB (the Vault history in the frontend)"""
    result = await db.execute(
        select(DBSession).order_by(DBSession.created_at.desc()).limit(limit)
    )
    sessions = result.scalars().all()
    return {"sessions": [
        {
            "id":         s.id,
            "title":      s.title,
            "type":       s.type,
            "status":     s.status,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sessions
    ]}


@app.get("/sessions/{session_id}", tags=["Database"])
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get full details of a session (transcript + summary)"""
    r = await db.execute(select(DBSession).where(DBSession.id == session_id))
    s = r.scalar_one_or_none()
    if not s:
        raise HTTPException(404, f"Session not found: {session_id}")

    tr = await db.execute(select(Transcript).where(Transcript.session_id == session_id))
    transcript = tr.scalar_one_or_none()

    sm = await db.execute(select(Summary).where(Summary.session_id == session_id))
    summary = sm.scalar_one_or_none()

    return {
        "session":    {"id": s.id, "title": s.title, "type": s.type},
        "transcript": {"text": transcript.full_text, "word_count": transcript.word_count} if transcript else None,
        "summary": {
            "executive": summary.executive,
            "bullets":   json.loads(summary.bullet_points or "[]"),
            "keywords":  json.loads(summary.keywords or "[]"),
        } if summary else None,
    }


@app.delete("/sessions/{session_id}", tags=["Database"])
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a session and all its data"""
    r = await db.execute(select(DBSession).where(DBSession.id == session_id))
    s = r.scalar_one_or_none()
    if not s:
        raise HTTPException(404, "Session not found")
    await db.delete(s)
    await db.commit()
    return {"deleted": session_id}


# ── Run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

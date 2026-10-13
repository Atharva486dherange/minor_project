"""
GigaChad AI Backend — Database Models & Configuration
Stores sessions, transcripts, action items, and users
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# ── Database Setup ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gigachad.db")
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id          = Column(String, primary_key=True)          # UUID
    name        = Column(String, nullable=False)
    email       = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)          # bcrypt hash, never plain text
    created_at  = Column(DateTime, default=datetime.utcnow)
    sessions    = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    """Represents one processed file (meeting, lecture, PDF, notes)"""
    __tablename__ = "sessions"
    id           = Column(String, primary_key=True)         # UUID
    user_id      = Column(String, ForeignKey("users.id"), nullable=True)
    title        = Column(String, nullable=False)
    type         = Column(String, nullable=False)            # 'meeting' | 'document' | 'notes'
    duration_sec = Column(Float, nullable=True)             # For audio/video
    file_path    = Column(String, nullable=True)            # Local temp path (deleted after processing)
    file_size_mb = Column(Float, nullable=True)
    status       = Column(String, default="pending")        # pending | processing | done | error
    created_at   = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    user         = relationship("User", back_populates="sessions")
    transcript   = relationship("Transcript", back_populates="session", uselist=False, cascade="all, delete-orphan")
    summary      = relationship("Summary", back_populates="session", uselist=False, cascade="all, delete-orphan")
    action_items = relationship("ActionItem", back_populates="session", cascade="all, delete-orphan")


class Transcript(Base):
    """Full transcript broken into speaker turns"""
    __tablename__ = "transcripts"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(String, ForeignKey("sessions.id"), nullable=False)
    full_text    = Column(Text, nullable=False)              # Clean concatenated transcript
    raw_text     = Column(Text, nullable=True)               # Raw whisper output with filler words
    language     = Column(String, default="en")
    word_count   = Column(Integer, default=0)
    created_at   = Column(DateTime, default=datetime.utcnow)

    session      = relationship("Session", back_populates="transcript")
    segments     = relationship("TranscriptSegment", back_populates="transcript", cascade="all, delete-orphan")


class TranscriptSegment(Base):
    """One speaker turn / timestamp line in the transcript"""
    __tablename__ = "transcript_segments"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    transcript_id = Column(Integer, ForeignKey("transcripts.id"), nullable=False)
    speaker      = Column(String, default="Unknown Speaker")
    start_time   = Column(Float, default=0.0)               # Seconds from start
    end_time     = Column(Float, default=0.0)
    text         = Column(Text, nullable=False)
    sentiment    = Column(String, nullable=True)             # 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL'
    sentiment_score = Column(Float, nullable=True)

    transcript   = relationship("Transcript", back_populates="segments")


class Summary(Base):
    """AI-generated summary outputs"""
    __tablename__ = "summaries"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    session_id    = Column(String, ForeignKey("sessions.id"), nullable=False)
    tldr          = Column(Text, nullable=True)              # 3-5 bullet TL;DR
    executive     = Column(Text, nullable=True)              # 1-paragraph executive brief
    bullet_points = Column(Text, nullable=True)              # JSON list of key points
    keywords      = Column(Text, nullable=True)              # JSON list of {word, score}
    risks         = Column(Text, nullable=True)              # JSON list of detected risks
    model_used    = Column(String, default="bart-large-cnn")
    created_at    = Column(DateTime, default=datetime.utcnow)

    session       = relationship("Session", back_populates="summary")


class ActionItem(Base):
    """Extracted action items / homework / tasks"""
    __tablename__ = "action_items"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(String, ForeignKey("sessions.id"), nullable=False)
    task         = Column(Text, nullable=False)
    assignee     = Column(String, nullable=True)             # Extracted person name
    deadline     = Column(String, nullable=True)             # Extracted date string
    priority     = Column(String, default="medium")          # high | medium | low
    status       = Column(String, default="pending")         # pending | done
    source_text  = Column(Text, nullable=True)               # Original sentence it was extracted from

    session      = relationship("Session", back_populates="action_items")


# ── DB Init Helper ────────────────────────────────────────────────
async def init_db():
    """Create all tables on first startup"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields a database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

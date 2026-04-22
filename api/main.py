"""Examiney.AI API — FastAPI backend (v3.0).

All endpoints return structured JSON errors on failure:
  {"error": "human-readable message", "code": "SCREAMING_SNAKE_CODE"}
"""

import sys
# Force UTF-8 stdout/stderr so Unicode chars in print() don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import logging
import os
import re
import secrets
import shutil
import string
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional

import httpx as _httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ── Coloured terminal logging ───────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BLUE   = "\033[94m"
_GREY   = "\033[90m"

class _ColourFormatter(logging.Formatter):
    _LEVEL_COLOURS = {
        logging.DEBUG:    _GREY   + "DEBUG  " + _RESET,
        logging.INFO:     _GREEN  + "INFO   " + _RESET,
        logging.WARNING:  _YELLOW + "WARN   " + _RESET,
        logging.ERROR:    _RED    + "ERROR  " + _RESET,
        logging.CRITICAL: _RED + _BOLD + "CRIT   " + _RESET,
    }
    def format(self, record: logging.LogRecord) -> str:
        ts    = self.formatTime(record, "%H:%M:%S")
        level = self._LEVEL_COLOURS.get(record.levelno, record.levelname)
        msg   = record.getMessage()
        # Colour the [Tag] prefix inside the message cyan
        import re as _re
        msg = _re.sub(r"(\[Examiney\]\[[^\]]+\])", _CYAN + r"\1" + _RESET, msg)
        return f"{_DIM}{ts}{_RESET}  {level} {msg}"

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_ColourFormatter())
logging.root.setLevel(logging.INFO)
logging.root.handlers = [_handler]

logger = logging.getLogger("examiney")


def _print_banner() -> None:
    """Print a startup banner so the terminal clearly shows the server is live."""
    print(f"\n{_BOLD}{_CYAN}{'─' * 60}{_RESET}")
    print(f"  {_BOLD}{_CYAN}Examiney.AI{_RESET}  {_DIM}Backend Server{_RESET}")
    print(f"  {_DIM}Whisper model : {os.getenv('WHISPER_MODEL', 'tiny')}{_RESET}")
    print(f"  {_DIM}API docs      : http://localhost:8000/docs{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─' * 60}{_RESET}\n")

from services.database import cloudinary_client, supabase_client
from services.parser.models import ParsedResume
from services.parser.parser import parse_pdf, parse_text
from services.question_gen.generator import generate_questions
from services.question_gen.models import InterviewScript
from services.scoring.models import ResponseScore, SentimentScores
from services.scoring.ocean_mapper import build_ocean_report
from services.scoring.response_scorer import score_response
from services.video_analysis.calibration.calibration_runner import (
    CalibrationResult,
    IrisSample,
    PointMeasurement,
    get_calibration_points,
    run_calibration,
)

# ── Password hashing (bcrypt direct — avoids passlib/__about__ issue) ──────────
try:
    import bcrypt as _bcrypt_lib
    def _hash_pw(pw: str) -> str:
        return _bcrypt_lib.hashpw(pw.encode(), _bcrypt_lib.gensalt()).decode()
    def _verify_pw(pw: str, h: str) -> bool:
        try:
            return _bcrypt_lib.checkpw(pw.encode(), h.encode())
        except Exception:
            return False
    logger.info("[Examiney][Auth] Using bcrypt for password hashing")
except ImportError:
    import hashlib
    def _hash_pw(pw: str) -> str:   return hashlib.sha256(pw.encode()).hexdigest()
    def _verify_pw(pw: str, h: str) -> bool: return _hash_pw(pw) == h
    logger.warning("[Examiney][Auth] bcrypt not found — using SHA-256 fallback")

# ── JSON Serialization helper for NumPy types ────────────────────────────────
def _convert_to_serializable(obj: Any) -> Any:
    """Recursively convert NumPy/PyArrow types to JSON-serializable Python types.
    
    Handles:
    - numpy.float32/float64 → float
    - numpy.int32/int64/integer → int
    - numpy.ndarray → list
    - dict → dict (with recursive conversion)
    - list → list (with recursive conversion)
    """
    try:
        import numpy as np
    except ImportError:
        return obj
    
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    return obj

# ── Per-session processing stage tracker ──────────────────────────────────────
# { session_id: {"stage": str, "label": str, "done": int, "total": int, "ts": float} }
_processing_stages: Dict[str, Dict[str, Any]] = {}
_stages_lock = threading.Lock()

# Guards against the same session being processed twice simultaneously
# (candidate portal + admin dashboard can both call /process)
_active_sessions: set = set()
_active_sessions_lock = threading.Lock()


def _set_stage(session_id: str, stage: str, label: str, done: int = 0, total: int = 0) -> None:
    with _stages_lock:
        _processing_stages[session_id] = {
            "stage": stage, "label": label,
            "done": done, "total": total,
            "ts": time.time(),
        }
    logger.info(f"[Examiney][PostSession] [{session_id[:8]}] Stage: {stage} — {label} ({done}/{total})")


def _clear_stage(session_id: str) -> None:
    with _stages_lock:
        _processing_stages.pop(session_id, None)
    with _active_sessions_lock:
        _active_sessions.discard(session_id)


# ── Whisper ────────────────────────────────────────────────────────────────────
# "tiny"  ~74 MB, ~4-6× faster than "small" on CPU, good for clear interview audio.
# Override: set WHISPER_MODEL=small (or base) in .env for higher accuracy.
_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

# Single-worker executor serialises all Whisper calls without needing a Lock.
# Lives for the process lifetime — never shut down.
from concurrent.futures import ThreadPoolExecutor as _TPE
_whisper_executor = _TPE(max_workers=1, thread_name_prefix="whisper")

@lru_cache(maxsize=1)
def _get_whisper():
    import whisper
    logger.info(f"[Examiney][Whisper] Loading model '{_WHISPER_MODEL}'…")
    try:
        m = whisper.load_model(_WHISPER_MODEL)
        logger.info(f"[Examiney][Whisper] Model '{_WHISPER_MODEL}' ready.")
        return m
    except Exception as e:
        logger.error(f"[Examiney][Whisper] CRITICAL — failed to load model '{_WHISPER_MODEL}': {type(e).__name__}: {e}", exc_info=True)
        raise

# ── Startup prewarms (background — don't delay server boot) ───────────────────
def _prewarm_whisper() -> None:
    """Pre-load Whisper weights so first interview question doesn't stall."""
    try:
        _get_whisper()   # triggers download + load; result is lru_cached
        logger.info(f"[Examiney][Prewarm] Whisper '{_WHISPER_MODEL}' model ready.")
    except Exception as e:
        logger.warning(f"[Examiney][Prewarm] Whisper prewarm failed: {e}")

def _prewarm_deepface() -> None:
    """Load DeepFace emotion weights into cache so first real call is instant."""
    try:
        import numpy as np
        from deepface import DeepFace
        dummy = np.zeros((4, 4, 3), dtype=np.uint8)
        DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False, silent=True)
        logger.info("[Examiney][Prewarm] DeepFace emotion model loaded.")
    except Exception as e:
        logger.warning(f"[Examiney][Prewarm] DeepFace prewarm skipped: {e}")

# Disabled: Prewarm calls now commented out for lazy loading on first use.
# Models will load on first request instead of blocking startup.
# Uncomment to re-enable:
# _whisper_executor.submit(_prewarm_whisper)
# threading.Thread(target=_prewarm_deepface, daemon=True).start()
_print_banner()

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Examiney.AI API",
    description=(
        "Full pipeline: resume parsing → question generation → interview session → "
        "gaze analysis → OCEAN personality scoring → recruiter dashboard."
    ),
    version="3.0.0",
)

# CORS origins — comma-separated list via CORS_ORIGINS env var.
# Default allows local dev only. Set CORS_ORIGINS in .env for production.
_CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _log_requests(request, call_next):
    """Print every incoming request + response status to the terminal."""
    t0  = time.perf_counter()
    resp = await call_next(request)
    ms  = (time.perf_counter() - t0) * 1000
    status = resp.status_code
    colour = _GREEN if status < 400 else (_YELLOW if status < 500 else _RED)
    print(
        f"{_DIM}{time.strftime('%H:%M:%S')}{_RESET}  "
        f"{colour}{status}{_RESET}  "
        f"{_BOLD}{request.method:<6}{_RESET} "
        f"{request.url.path}  "
        f"{_DIM}{ms:.0f}ms{_RESET}",
        flush=True,
    )
    return resp


# ── Helpers ────────────────────────────────────────────────────────────────────

def _err(message: str, code: str, status: int = 500):
    raise HTTPException(status_code=status, detail={"error": message, "code": code})


def _step(label: str, detail: str = "", ok: bool = True) -> None:
    """Print a clearly-visible pipeline checkpoint line to the terminal."""
    icon   = f"{_GREEN}✓{_RESET}" if ok else f"{_RED}✗{_RESET}"
    suffix = f"  {_DIM}{detail}{_RESET}" if detail else ""
    print(f"  {icon}  {_BOLD}{label}{_RESET}{suffix}", flush=True)


def _section(title: str) -> None:
    """Print a section divider so different pipeline stages are easy to spot."""
    print(f"\n{_CYAN}{_BOLD}▶ {title}{_RESET}  {_DIM}{time.strftime('%H:%M:%S')}{_RESET}", flush=True)


def _gen_login_id() -> str:
    return "NSC-" + "".join(secrets.choice(string.digits) for _ in range(6))

def _gen_opening_login_id() -> str:
    return "NSO-" + "".join(secrets.choice(string.digits) for _ in range(6))


def _gen_password(length: int = 8) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


def _cloudinary_upload_with_retry(
    *,
    data: bytes,
    public_id: str,
    folder: str,
    resource_type: str,
    max_retries: int = 3,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload to Cloudinary with exponential-backoff retries. Logs failures to Supabase."""
    for attempt in range(max_retries):
        try:
            return cloudinary_client.upload_bytes(
                data=data,
                public_id=public_id,
                folder=folder,
                resource_type=resource_type,
                overwrite=True,
                tags=["examiney", f"session:{session_id}" if session_id else "session:unknown"],
                context={"session_id": session_id or ""},
            )
        except Exception as exc:
            wait = (2 ** attempt) + secrets.randbelow(100) / 100.0
            logger.warning(f"[Examiney][CloudinaryUpload] attempt {attempt+1}/{max_retries} failed: {exc}. Retrying in {wait:.1f}s")
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                supabase_client.log_error("CloudinaryUpload", str(exc), session_id)
                raise


# ── Known Whisper hallucination patterns (generated on silence/noise) ──────────
_WHISPER_HALLUCINATION_RE = re.compile(
    r"(thank you[\s.!]*$|thanks for watching|subtitles by|"
    r"www\.|\.{3,}|^\s*\.\s*$|music playing|applause|"
    r"foreign[\s]*$|^\s*\[.*\]\s*$)",
    re.IGNORECASE,
)


def _is_whisper_hallucination(text: str) -> bool:
    """Return True if the Whisper output looks like hallucinated garbage.

    Whisper generates multilingual noise text when the audio is silent or
    contains only background sound.  Two signals:
    1. Multiple Unicode script families in one short response.
    2. Known Whisper silence patterns.
    """
    if not text or not text.strip():
        return False

    # Check for known noise patterns
    if _WHISPER_HALLUCINATION_RE.search(text):
        return True

    # Count distinct Unicode script blocks present
    has_latin     = bool(re.search(r"[a-zA-Z]", text))
    has_cyrillic  = bool(re.search(r"[\u0400-\u04FF]", text))
    has_greek     = bool(re.search(r"[\u0370-\u03FF]", text))
    has_arabic    = bool(re.search(r"[\u0600-\u06FF]", text))
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_cjk       = bool(re.search(r"[\u4E00-\u9FFF\u3040-\u309F\uAC00-\uD7AF]", text))
    has_hebrew    = bool(re.search(r"[\u0590-\u05FF]", text))

    script_count = sum([has_latin, has_cyrillic, has_greek,
                        has_arabic, has_devanagari, has_cjk, has_hebrew])
    if script_count >= 2:
        logger.warning(f"[Examiney][Whisper] hallucination detected — {script_count} mixed scripts: {repr(text[:80])}")
        return True

    # Very short output (< 3 real words) after a 90-second question window → suspicious
    word_count = len([w for w in text.split() if len(w) > 1])
    if word_count < 3:
        logger.warning(f"[Examiney][Whisper] near-empty output ({word_count} words) — treating as no response")
        return True

    return False


def _transcribe(audio_path: str, timeout_seconds: int = 120) -> tuple:
    """Transcribe *audio_path* via Whisper with a hard per-file timeout.

    All calls are serialised through a single-worker executor so the model is
    never called concurrently (it is not thread-safe).  The executor is NOT used
    as a context manager so shutdown never blocks the caller.

    Returns (transcript: str, flagged: bool).
    flagged=True → empty, hallucination, or timeout; caller should still score
    using a '[NO AUDIO]' placeholder so OCEAN receives data for every question.
    """
    from concurrent.futures import TimeoutError as _FutTimeout

    def _work() -> tuple:
        try:
            abs_path = os.path.abspath(audio_path)
            
            if not os.path.exists(abs_path):
                logger.error(f"[Examiney][Whisper] Audio file does not exist: {abs_path} (original: {audio_path})")
                return "", True
            
            file_size = os.path.getsize(abs_path)
            if file_size == 0:
                logger.warning(f"[Examiney][Whisper] Audio file is empty (0 bytes): {abs_path}")
                return "", True
            
            # Double-check file is readable BEFORE passing to Whisper
            is_readable = os.access(abs_path, os.R_OK)
            if not is_readable:
                logger.error(f"[Examiney][Whisper] Audio file exists but NOT READABLE: {abs_path}")
                return "", True
            
            logger.info(f"[Examiney][Whisper] transcribing '{os.path.basename(abs_path)}' ({file_size} bytes) from {abs_path}")
            
            try:
                model = _get_whisper()
            except Exception as model_err:
                logger.error(f"[Examiney][Whisper] FAILED to load model: {type(model_err).__name__}: {model_err}", exc_info=True)
                return "", True
            
            # Final checkpoint: log that we're about to call Whisper with the absolute path
            logger.info(f"[Examiney][Whisper] calling model.transcribe({abs_path})")
            try:
                result = model.transcribe(abs_path, fp16=False)
            except FileNotFoundError as fnf_err:
                # Special handling for FileNotFoundError with diagnostic context
                logger.error(f"[Examiney][Whisper] transcription FileNotFoundError: {fnf_err} | Path: {abs_path} | Exists now: {os.path.exists(abs_path)}", exc_info=True)
                return "", True
            except Exception as transcribe_err:
                logger.error(f"[Examiney][Whisper] transcription failed on file: {type(transcribe_err).__name__}: {transcribe_err} | Path: {abs_path}", exc_info=True)
                return "", True
            
            text = (result.get("text") or "").strip()
            if _is_whisper_hallucination(text):
                logger.warning("[Examiney][Whisper] hallucination detected — treating as empty")
                return "", True
            logger.info(f"[Examiney][Whisper] done — {len(text)} chars, {len(text.split())} words")
            return text, False
        except Exception as exc:
            logger.error(f"[Examiney][Whisper] unexpected error: {type(exc).__name__}: {exc}", exc_info=True)
            return "", True

    fut = _whisper_executor.submit(_work)
    try:
        return fut.result(timeout=timeout_seconds)
    except _FutTimeout:
        logger.error(f"[Examiney][Whisper] timed out after {timeout_seconds}s on {audio_path}")
        return "", True
    except Exception as exc:
        logger.error(f"[Examiney][Whisper] executor error: {type(exc).__name__}: {exc}", exc_info=True)
        return "", True


# ── Parse endpoints ────────────────────────────────────────────────────────────

@app.post("/parse/pdf", response_model=ParsedResume, tags=["Parse"],
          summary="Upload a PDF resume → structured data")
async def parse_resume_pdf(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        _err("Only PDF files are supported.", "INVALID_FILE_TYPE", 400)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return parse_pdf(tmp_path)
    except Exception as e:
        _err(f"PDF parsing failed: {e}", "PARSE_FAILED")
    finally:
        os.unlink(tmp_path)


@app.post("/parse/text", response_model=ParsedResume, tags=["Parse"],
          summary="Submit plain-text resume → structured data")
async def parse_resume_text(text: str = Form(...)):
    return parse_text(text)


# ── Question generation ────────────────────────────────────────────────────────

class QuestionGenRequest(BaseModel):
    resume_markdown: Optional[str] = ""
    job_description: Optional[str] = ""
    model: Optional[str] = None
    ollama_url: Optional[str] = None
    section_counts: Optional[Dict[str, int]] = None   # e.g. {"intro":2,"technical":5,"behavioral":3,"logical":3}


@app.post("/generate-questions", response_model=InterviewScript, tags=["Questions"],
          summary="Generate interview questions with ideal_answer fields")
async def generate_interview_questions(req: QuestionGenRequest):
    if not req.resume_markdown and not req.job_description:
        _err("Provide resume_markdown and/or job_description.", "MISSING_INPUT", 400)
    kwargs: Dict[str, Any] = {
        "resume_markdown": req.resume_markdown or "",
        "job_description": req.job_description or "",
    }
    if req.model:          kwargs["model"]          = req.model
    if req.ollama_url:     kwargs["ollama_url"]     = req.ollama_url
    if req.section_counts: kwargs["section_counts"] = req.section_counts
    
    logger.info(f"[Examiney][API] ► /generate-questions endpoint called")
    logger.info(f"[Examiney][API]   Resume: {len(req.resume_markdown or '')} chars, JD: {len(req.job_description or '')} chars")
    start = time.time()
    try:
        result = generate_questions(**kwargs)
        elapsed = time.time() - start
        logger.info(f"[Examiney][API] ✓ /generate-questions DONE in {elapsed:.1f}s — returned {len(result.questions)} questions")
        return result
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[Examiney][API] ✗ /generate-questions FAILED in {elapsed:.1f}s: {e}")
        _err(f"Question generation failed: {e}", "GENERATION_FAILED")


# ── Combined parse + generate ──────────────────────────────────────────────────

@app.post("/parse-and-generate", tags=["Combined"],
          summary="Upload PDF + optional JD → parse + 18-20 questions in one call")
async def parse_and_generate(
    file: Optional[UploadFile] = File(None),
    job_description: str = Form(""),
    model: str = Form("qwen2.5:0.5b"),
    ollama_url: str = Form("http://localhost:11434"),
):
    logger.info(f"[Examiney][API] ► /parse-and-generate endpoint called")
    start = time.time()
    
    parsed: Optional[ParsedResume] = None
    resume_markdown = ""
    if file:
        logger.info(f"[Examiney][API]   PDF upload: {file.filename if file else 'None'}")
        if not (file.filename or "").lower().endswith(".pdf"):
            _err("Only PDF files are supported.", "INVALID_FILE_TYPE", 400)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            parse_start = time.time()
            logger.info(f"[Examiney][API]   Parsing PDF...")
            parsed = parse_pdf(tmp_path)
            resume_markdown = parsed.raw_markdown
            parse_elapsed = time.time() - parse_start
            logger.info(f"[Examiney][API]   ✓ PDF parsed in {parse_elapsed:.1f}s → {len(resume_markdown)} chars")
        except Exception as e:
            _err(f"PDF parsing failed: {e}", "PARSE_FAILED")
        finally:
            os.unlink(tmp_path)
    
    if not resume_markdown and not job_description:
        _err("Provide a PDF and/or job_description.", "MISSING_INPUT", 400)
    
    logger.info(f"[Examiney][API]   Resume: {len(resume_markdown)} chars, JD: {len(job_description)} chars")
    logger.info(f"[Examiney][API]   Starting question generation...")
    try:
        gen_start = time.time()
        script = generate_questions(
            resume_markdown=resume_markdown,
            job_description=job_description,
            model=model,
            ollama_url=ollama_url,
        )
        gen_elapsed = time.time() - gen_start
        total_elapsed = time.time() - start
        logger.info(f"[Examiney][API]   ✓ Questions generated in {gen_elapsed:.1f}s")
        logger.info(f"[Examiney][API] ✓ /parse-and-generate DONE in {total_elapsed:.1f}s")
        return {"parsed_resume": parsed, "interview_script": script}
    except Exception as e:
        total_elapsed = time.time() - start
        logger.error(f"[Examiney][API] ✗ /parse-and-generate FAILED in {total_elapsed:.1f}s: {e}")
        _err(f"Question generation failed: {e}", "GENERATION_FAILED")


# ── Session endpoints ──────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    candidate_name: str
    job_opening_id: Optional[str] = None
    opening_title: Optional[str] = None   # human-readable title; used as job_opening_id if job_opening_id not set
    interviewer_id: str
    questions: Optional[List[Dict[str, Any]]] = None
    job_description: Optional[str] = ""


@app.post("/session/create", tags=["Session"],
          summary="Create session + generate one-time candidate credentials (NSC-XXXXXX)")
async def create_session(req: CreateSessionRequest):
    session_id = str(uuid.uuid4())

    # Prefer explicit job_opening_id; otherwise derive from opening_title or generate UUID
    if req.job_opening_id:
        job_opening_id = req.job_opening_id
    elif req.opening_title:
        # Slugify the title: lowercase, replace non-alphanumeric with hyphens
        slug = re.sub(r"[^a-z0-9]+", "-", req.opening_title.lower()).strip("-")
        job_opening_id = slug[:64] or str(uuid.uuid4())
    else:
        job_opening_id = str(uuid.uuid4())
    # One login_id per opening (shared). Password remains per candidate/session.
    login_id = supabase_client.get_opening_login_id(job_opening_id) or _gen_opening_login_id()
    raw_password   = _gen_password()
    logger.info(f"[Examiney][CreateSession] session={session_id} login={login_id} opening={job_opening_id} questions_count={len(req.questions or [])} jd_len={len(req.job_description or '')}")
    try:
        logger.info(f"[Examiney][CreateSession] Inserting session into Supabase...")
        supabase_client.create_session(
            session_id=session_id,
            candidate_name=req.candidate_name,
            job_opening_id=job_opening_id,
            interviewer_id=req.interviewer_id,
            login_id=login_id,
            questions=req.questions or [],
            job_description=req.job_description or "",
        )
        logger.info(f"[Examiney][CreateSession] Session inserted OK. Creating credentials...")
        supabase_client.create_candidate_credentials(
            session_id=session_id,
            login_id=login_id,
            hashed_password=_hash_pw(raw_password),
        )
        logger.info(f"[Examiney][CreateSession] Credentials created OK.")
    except Exception as e:
        logger.exception(f"[Examiney][CreateSession] ERROR: {e}")
        _err(f"Failed to create session: {e}", "SESSION_CREATE_FAILED")
    return {
        "session_id":     session_id,
        "job_opening_id": job_opening_id,
        "login_id":       login_id,
        "password":       raw_password,
    }


# ── Candidate login ────────────────────────────────────────────────────────────

class CandidateLoginRequest(BaseModel):
    login_id: str
    password: str


@app.post("/candidate/login", tags=["Candidate"],
          summary="Validate one-time credentials → return session_id + questions")
async def candidate_login(req: CandidateLoginRequest):
    login_id = req.login_id.strip()
    creds_list = supabase_client.list_credentials(login_id)
    if not creds_list:
        _err("Invalid login ID.", "INVALID_CREDENTIALS", 401)

    # Find the matching (unused) credential by password
    matched = None
    for c in creds_list:
        if c.get("used"):
            continue
        if _verify_pw(req.password, c.get("hashed_password", "")):
            matched = c
            break
    if not matched:
        _err("Incorrect password.", "INVALID_CREDENTIALS", 401)

    session = supabase_client.get_session(matched["session_id"])
    if not session:
        _err("Session not found.", "SESSION_NOT_FOUND", 404)
    
    logger.info(f"[Examiney][CandidateLogin] login_id={login_id} session_id={matched['session_id']} questions_count={len(session.get('questions', []))} job_desc_len={len(session.get('job_description', ''))}")
    
    supabase_client.mark_credentials_used(int(matched["id"]))
    return {
        "session_id":      session["session_id"],
        "candidate_name":  session["candidate_name"],
        "job_description": session.get("job_description", ""),
        "questions":       session.get("questions", []),
    }


# ── Post-session processing ───────────────────────────────────────────────────

def _download_to_tmp(url: str, suffix: str) -> str:
    """Download a URL to a temporary file, working around Windows file locking issues.
    
    Strategy: Create file, close all handles, verify it's readable, then return path.
    Logs every step for debugging Windows issues.
    """
    import time
    
    # Use UUID for unique filenames
    temp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(temp_dir, f"examiney_{uuid.uuid4().hex[:12]}{suffix}")
    
    logger.info(f"[Examiney][Download] Creating temp file: {tmp_path}")
    
    # Download directly to this path
    try:
        with _httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as out:
                bytes_written = 0
                for chunk in r.iter_bytes():
                    out.write(chunk)
                    bytes_written += len(chunk)
                out.flush()
                os.fsync(out.fileno())
            logger.info(f"[Examiney][Download] Wrote {bytes_written} bytes to {tmp_path}")
    except Exception as e:
        logger.error(f"[Examiney][Download] Failed to write file: {type(e).__name__}: {e}", exc_info=True)
        raise
    
    # Verify file exists immediately after closing
    if not os.path.exists(tmp_path):
        logger.error(f"[Examiney][Download] File disappeared immediately after write: {tmp_path}")
        raise FileNotFoundError(f"File not found after write: {tmp_path}")
    
    file_size = os.path.getsize(tmp_path)
    logger.info(f"[Examiney][Download] File exists, size={file_size} bytes")
    
    if file_size == 0:
        os.unlink(tmp_path)  # Clean up
        raise ValueError(f"Downloaded file is empty (0 bytes): {tmp_path}")
    
    # Wait for Windows to fully release all locks and for antivirus to finish scanning
    # This is the critical part - longer delay helps avoid "file in use" errors
    logger.info(f"[Examiney][Download] Waiting 1.5s for file locks to release...")
    time.sleep(1.5)
    
    # Verify the file is still there and readable
    if not os.path.exists(tmp_path):
        logger.error(f"[Examiney][Download] File was deleted during wait period: {tmp_path}")
        raise FileNotFoundError(f"File deleted during wait: {tmp_path}")
    
    # Try to open and read a bit to ensure no locks
    try:
        with open(tmp_path, "rb") as test:
            chunk = test.read(1024)
            logger.info(f"[Examiney][Download] File is readable, read {len(chunk)} bytes successfully")
    except IOError as e:
        logger.error(f"[Examiney][Download] File is not readable: {type(e).__name__}: {e}")
        raise IOError(f"File created but not readable: {tmp_path} — {e}")
    
    logger.info(f"[Examiney][Download] SUCCESS: {tmp_path} ready for processing")
    return tmp_path


# ── Integrity flag ─────────────────────────────────────────────────────────────

@app.post("/session/{session_id}/flag-integrity", tags=["Session"],
          summary="Flag session as high-integrity-risk (e.g. repeated fullscreen violations)")
async def flag_integrity(session_id: str = Path(...)):
    """Called by the interview frontend when the candidate exits fullscreen a second time.
    Inserts a video_signals row with risk_level=high so the recruiter dashboard reflects it."""
    try:
        supabase_client.save_video_signals(
            session_id=session_id,
            question_id="integrity_violation",
            gaze_zone_distribution={},
            cheat_flags={"risk_level": "high", "reason": "candidate exited fullscreen twice — session flagged for review"},
            emotion_distribution={},
            avg_hrv_rmssd=None,
            stress_spike_detected=True,
            hr_bpm=None,
            gaze_metrics=_convert_to_serializable({"provider": "integrity", "status": "flagged", "risk_level": "high",
                          "zone_distribution": {}, "cheat_flags": {"risk_level": "high"}}),
        )
        logger.info(f"[Examiney][Integrity] Session {session_id} flagged as HIGH risk")
    except Exception as e:
        logger.error(f"[Examiney][Integrity] Failed to save flag: {e}")
    return {"status": "flagged"}


# ── Save response ──────────────────────────────────────────────────────────────

@app.post("/session/{session_id}/save-response", tags=["Session"],
          summary="Upload audio+video to Cloudinary immediately (no local persistence)")
async def save_response(
    session_id:     str                  = Path(...),
    question_id:    str                  = Form(...),
    question_number: int                 = Form(...),
    question_text:  str                  = Form(...),
    ideal_answer:   str                  = Form(...),
    question_stage: str                  = Form("intro"),   # validated below against known stages
    audio_file:     Optional[UploadFile] = File(None),
    video_file:     Optional[UploadFile] = File(None),
):
    # Validate stage against known values — client-supplied field, must not be trusted blindly
    _KNOWN_STAGES = {"intro", "technical", "logical", "behavioral", "situational"}
    if question_stage not in _KNOWN_STAGES:
        question_stage = "intro"
    _section(f"Save Response  session={session_id[:8]}  Q{question_number} [{question_stage}]")
    logger.info(f"[Examiney][SaveResponse] session={session_id} q={question_id} qn={question_number} stage={question_stage} — uploading to Cloudinary")

    session = supabase_client.get_session(session_id)
    login_id = (session or {}).get("login_id") or "unknown"
    folder = cloudinary_client.build_session_folder(login_id=login_id, session_id=session_id)

    audio_public_id: Optional[str] = None
    video_public_id: Optional[str] = None
    audio_url: Optional[str] = None
    video_url: Optional[str] = None

    if audio_file:
        audio_bytes = await audio_file.read()
        if audio_bytes:
            _step("Audio received", f"{len(audio_bytes):,} bytes")
            audio_public_id = cloudinary_client.build_public_id(
                login_id=login_id, session_id=session_id, question_number=question_number, kind="audio"
            )
            try:
                resp = _cloudinary_upload_with_retry(
                    data=audio_bytes,
                    public_id=audio_public_id,
                    folder=folder,
                    resource_type="video",
                    session_id=session_id,
                )
                audio_url = resp.get("secure_url") or resp.get("url")
                _step("Audio → Cloudinary", audio_url or "no url returned")
            except Exception as e:
                _step("Audio → Cloudinary FAILED", str(e), ok=False)
                logger.error(f"[Examiney][SaveResponse] AUDIO UPLOAD FAILED q={question_id}: {e}")
                supabase_client.log_error("AudioUpload", str(e), session_id)
        else:
            _step("Audio file empty — skipped", ok=False)

    if video_file:
        video_bytes = await video_file.read()
        if video_bytes:
            _step("Video received", f"{len(video_bytes):,} bytes")
            video_public_id = cloudinary_client.build_public_id(
                login_id=login_id, session_id=session_id, question_number=question_number, kind="video"
            )
            try:
                resp = _cloudinary_upload_with_retry(
                    data=video_bytes,
                    public_id=video_public_id,
                    folder=folder,
                    resource_type="video",
                    session_id=session_id,
                )
                video_url = resp.get("secure_url") or resp.get("url")
                _step("Video → Cloudinary", video_url or "no url returned")
            except Exception as e:
                _step("Video → Cloudinary FAILED", str(e), ok=False)
                logger.error(f"[Examiney][SaveResponse] VIDEO UPLOAD FAILED q={question_id}: {e}")
                supabase_client.log_error("VideoUpload", str(e), session_id)
        else:
            _step("Video file empty — skipped", ok=False)

    # Upsert placeholder row (or update existing) immediately with Cloudinary pointers.
    try:
        supabase_client.save_question_response(
            session_id=session_id,
            question_id=question_id,
            question_text=question_text,
            ideal_answer=ideal_answer,
            transcript="",
            transcript_flagged=False,
            semantic_score=0.0,
            sentiment={"compound": 0, "pos": 0, "neg": 0, "neu": 1},
            combined_score=0.0,
            technical_score=0,
            communication_score=0,
            behavioral_score=0,
            engagement_score=0,
            authenticity_score=0,
            video_file_id=video_public_id,
            audio_file_id=audio_public_id,
            video_url=video_url,
            audio_url=audio_url,
        )
    except Exception as e:
        logger.error(f"[Examiney][SaveResponse] DB upsert failed ({e})")
        supabase_client.log_error("SaveResponseDB", str(e), session_id)
        _err(f"Failed to save response: {e}", "DB_WRITE_FAILED")

    # Fire per-question processing immediately — don't wait for end of interview
    threading.Thread(
        target=_bg_process_single_response,
        args=(session_id, question_id, audio_url, video_url, question_text, ideal_answer, question_stage),
        daemon=True,
    ).start()

    logger.info(f"[Examiney][SaveResponse] DONE q={question_id} audio_url={bool(audio_url)} video_url={bool(video_url)}")
    return {"status": "uploaded", "session_id": session_id, "question_id": question_id, "audio_url": audio_url, "video_url": video_url}


# ── Per-question early background processor ────────────────────────────────────

def _bg_process_single_response(
    session_id: str,
    question_id: str,
    audio_url: Optional[str],
    video_url: Optional[str],
    question_text: str,
    ideal_answer: str,
    stage: str = "intro",
) -> None:
    """Transcribe, score, and run GazeFollower on ONE response immediately after upload.
    Runs in a daemon thread so save-response returns instantly.
    The final /process call only needs to run OCEAN finalize."""
    _section(f"Background Processing  session={session_id[:8]}  q={question_id[:8]}")

    transcript = ""
    transcript_flagged = False

    # Step 1: Transcribe audio (fall back to video_url when audio_url absent —
    # the frontend sends a single webm that contains the audio track)
    transcription_url = audio_url or video_url
    if transcription_url:
        tmp_path: Optional[str] = None
        source = "audio" if audio_url else "video (audio track)"
        try:
            _step(f"Downloading {source} for transcription")
            tmp_path = _download_to_tmp(transcription_url, suffix=".webm")
            _step("Running Whisper", f"{os.path.getsize(tmp_path):,} bytes")
            transcript, transcript_flagged = _transcribe(tmp_path)
            words = len(transcript.split()) if transcript else 0
            if transcript_flagged or not transcript:
                _step("Transcription", "empty / flagged", ok=False)
            else:
                _step("Transcription done", f"{words} words: \"{transcript[:60]}{'…' if len(transcript) > 60 else ''}\"")
            supabase_client.update_transcript(
                session_id=session_id, question_id=question_id,
                transcript=transcript, transcript_flagged=transcript_flagged,
            )
        except Exception as e:
            _step("Transcription FAILED", str(e), ok=False)
            logger.error(f"[Examiney][EarlyProcess] transcription FAILED q={question_id}: {e}")
            supabase_client.log_error("EarlyTranscribe", str(e), session_id)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    else:
        _step("No audio or video URL — transcription skipped", ok=False)

    # Step 2: Score + LLM correctness verdict + dimension scoring.
    _tscore = transcript.strip() or "[NO AUDIO RECORDED]"
    try:
        from services.scoring.llm_marker import judge_response as _judge, mark_response as _mark
        _step("Scoring response", f"stage={stage}")
        score: ResponseScore = score_response(question_id, _tscore, ideal_answer or "", stage=stage)

        _step("LLM judge running")
        judgment = _judge(
            question_text=question_text,
            ideal_answer=ideal_answer or "",
            transcript=_tscore,
            stage=stage,
        )
        verdict        = judgment.get("verdict", "Weak")
        verdict_reason = judgment.get("verdict_reason", "")
        key_gaps       = judgment.get("key_gaps", [])
        strengths      = judgment.get("strengths", [])
        llm_score      = judgment.get("score", score.combined_score)
        final_score    = round((llm_score * 0.7 + score.combined_score * 0.3), 2)
        _step("Score computed", f"verdict={verdict}  score={final_score:.2f}")

        marks: Dict[str, Any] = {}
        try:
            marks = _mark(
                question_text=question_text,
                ideal_answer=ideal_answer or "",
                transcript=_tscore,
                stage=stage,
            )
        except Exception as me:
            logger.warning(f"[Examiney][EarlyProcess] mark_response failed q={question_id}: {me}")

        supabase_client.save_question_response(
            session_id=session_id, question_id=question_id,
            question_text=question_text, ideal_answer=ideal_answer,
            transcript=transcript, transcript_flagged=transcript_flagged,
            semantic_score=score.semantic_score,
            sentiment=score.sentiment.model_dump(),
            combined_score=final_score,
            technical_score=marks.get("technical"),
            communication_score=marks.get("communication"),
            behavioral_score=marks.get("behavioral"),
            engagement_score=marks.get("engagement"),
            authenticity_score=marks.get("authenticity"),
            video_url=video_url,
            audio_url=audio_url,
            llm_verdict=verdict,
            llm_verdict_reason=verdict_reason,
            llm_key_gaps=key_gaps,
            llm_strengths=strengths,
        )
    except Exception as e:
        logger.error(f"[Examiney][EarlyProcess] scoring FAILED q={question_id}: {e}")
        supabase_client.log_error("EarlyScore", str(e), session_id)

    # Step 3: GazeFollower on video with personalized calibration
    if video_url:
        tmp_v: Optional[str] = None
        try:
            from services.video_analysis.gaze.gazefollower_runner import run_gazefollower_on_video
            from services.video_analysis.calibration.calibration_runner import load_calibration
            _step("Downloading video for gaze analysis")
            tmp_v = _download_to_tmp(video_url, suffix=".webm")
            cal_data: Dict[str, Any] = {}
            try:
                cal_data = load_calibration(session_id)
                _step("Calibration data loaded", f"baseline_var={cal_data.get('baseline_gaze_variance', 0):.4f}")
            except Exception as cal_err:
                _step("Calibration not found — using defaults", str(cal_err), ok=False)
            _step("Running gaze analysis (MediaPipe)")
            metrics = run_gazefollower_on_video(tmp_v, session_id=session_id, calibration_data=cal_data)
            supabase_client.update_video_gaze_metrics(
                session_id=session_id, question_id=question_id, gaze_metrics=_convert_to_serializable(metrics),
            )
            risk = metrics.get("cheat_risk_level", "unknown")
            _step("Gaze analysis done", f"status={metrics.get('status')}  cheat_risk={risk}")
        except Exception as e:
            _step("Gaze analysis FAILED", str(e), ok=False)
            logger.error(f"[Examiney][EarlyProcess] GazeFollower FAILED q={question_id}: {e}")
            supabase_client.log_error("EarlyGaze", str(e), session_id)
        finally:
            if tmp_v and os.path.exists(tmp_v):
                try:
                    os.unlink(tmp_v)
                except Exception:
                    pass

    print(f"  {_GREEN}{_BOLD}✓  Q{question_id[:8]} fully processed{_RESET}\n", flush=True)


# ── Video analysis ─────────────────────────────────────────────────────────────

@app.post("/video/analyze-chunk", tags=["Video"],
          summary="Gaze zone classification + DeepFace emotion + rPPG HRV → Supabase")
async def analyze_video_chunk(
    session_id:          str                  = Form(...),
    question_id:         str                  = Form(...),
    gaze_samples:        str                  = Form("[]"),
    head_pose_samples:   str                  = Form("[]"),
    face_absent_frames:  str                  = Form("0"),
    total_pose_frames:   str                  = Form("0"),
    video_file:          Optional[UploadFile] = File(None),
):
    logger.info(f"[Examiney][VideoChunk] session={session_id} q={question_id}")
    from services.video_analysis.gaze.cheating_detector import detect_cheating
    from services.video_analysis.gaze.zone_classifier import ZoneClassifier

    # Parse gaze samples
    try:
        samples: List[Dict] = json.loads(gaze_samples) if gaze_samples else []
    except json.JSONDecodeError:
        samples = []
    gaze_points = [(float(s["x"]), float(s["y"])) for s in samples if "x" in s]

    # Load calibration once — reused by zone classifier fallback AND cheating detector
    from services.video_analysis.calibration.calibration_runner import load_calibration
    cal_data: Dict[str, Any] = {}
    try:
        cal_data = load_calibration(session_id)
    except Exception:
        pass   # no calibration file yet — use defaults
    baseline_var = cal_data.get("baseline_gaze_variance", 0.004)

    # Gaze zone classification
    gaze_zone_distribution: Dict[str, float] = {}
    try:
        clf = ZoneClassifier(session_id)
        zone_counts: Dict[str, int] = {}
        prev = None
        for pt in gaze_points:
            z = clf.classify(pt, prev)
            zone_counts[str(z)] = zone_counts.get(str(z), 0) + 1   # str() ensures JSON-safe keys
            prev = pt
        total = len(gaze_points) or 1
        gaze_zone_distribution = {z: round(c / total, 4) for z, c in zone_counts.items()}
    except FileNotFoundError:
        # No calibration saved yet — use calibration-aware _classify_zone with whatever data we have
        from services.video_analysis.gaze.gazefollower_runner import _classify_zone as _gz_classify
        zone_counts = {}
        for x, y in gaze_points:
            z = _gz_classify(x, y, cal_data)
            zone_counts[z] = zone_counts.get(z, 0) + 1
        total = len(gaze_points) or 1
        gaze_zone_distribution = {z: round(c / total, 4) for z, c in zone_counts.items()}
    except Exception as e:
        logger.warning(f"[Examiney][GazeZone] {e}")
        gaze_zone_distribution = {"neutral": 1.0}

    # Cheating detection — personalised thresholds from calibration baseline
    neuro_adj = cal_data.get("neurodiversity_adjustment", 1.0)
    try:
        cheat_result = detect_cheating(
            gaze_points,
            baseline_variance=baseline_var,
            neurodiversity_adjustment=neuro_adj,
        )
        cheat_flags = cheat_result if isinstance(cheat_result, dict) else {"raw": str(cheat_result)}
    except Exception as e:
        logger.warning(f"[Examiney][CheatDetect] {e}")
        cheat_flags = {"risk_level": "low"}

    # Head pose cheating detection — yaw/pitch + face-absence from browser FaceMesh
    try:
        from services.video_analysis.gaze.cheating_detector import analyze_head_pose
        pose_samples: List[Dict] = json.loads(head_pose_samples) if head_pose_samples else []
        absent_frames  = int(face_absent_frames or 0)
        total_frames   = int(total_pose_frames  or 0)
        absent_pct     = absent_frames / max(total_frames, 1) if total_frames > 0 else 0.0
        if pose_samples or absent_frames > 0:
            head_pose_result = analyze_head_pose(
                pose_samples,
                neurodiversity_adjustment=neuro_adj,
                face_absent_pct=absent_pct,
            )
            cheat_flags["head_pose"] = head_pose_result
            # Merge head pose effective score into overall risk level
            hp_eff = head_pose_result.get("head_pose_effective_score", 0.0)
            gaze_eff = cheat_flags.get("effective_score", 0.0)
            combined_eff = float(gaze_eff) + float(hp_eff)
            if combined_eff >= 5:
                cheat_flags["risk_level"] = "high"
            elif combined_eff >= 2 and cheat_flags.get("risk_level") != "high":
                cheat_flags["risk_level"] = "medium"
    except Exception as e:
        logger.warning(f"[Examiney][HeadPose] {e}")

    # Video analysis (emotion + rPPG)
    emotion_distribution: Dict[str, float] = {"neutral": 1.0}
    avg_hrv_rmssd: Optional[float] = None
    hr_bpm: Optional[float] = None
    stress_spike = False

    if video_file:
        suffix = os.path.splitext(video_file.filename or "video.webm")[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await video_file.read())
            video_tmp = tmp.name
        try:
            from services.video_analysis.emotion_analyzer import analyze_emotions_from_video
            from services.video_analysis.rppg import analyze_rppg_from_video
            emotion_distribution = analyze_emotions_from_video(video_tmp)
            rppg = analyze_rppg_from_video(video_tmp)
            avg_hrv_rmssd = rppg["avg_hrv_rmssd"]
            hr_bpm        = rppg["hr_bpm"]
            stress_spike  = rppg["stress_spike_detected"]
        except Exception as e:
            logger.error(f"[Examiney][VideoAnalysis] {e}")
            supabase_client.log_error("VideoAnalysis", str(e), session_id)
        finally:
            try:
                os.unlink(video_tmp)
            except Exception:
                pass  # Windows may hold file lock; safe to ignore

    # Save all collected signals — always attempt even if emotion/rPPG failed
    gaze_metrics_rt = {
        "provider": "real-time",
        "status": "ok",
        "zone_distribution": gaze_zone_distribution,
        "cheat_flags": cheat_flags,
        "baseline_variance": baseline_var,
    }
    try:
        # Convert all numpy types to Python native types for JSON serialization
        record = supabase_client.save_video_signals(
            session_id=session_id,
            question_id=question_id,
            gaze_zone_distribution=_convert_to_serializable(gaze_zone_distribution),
            cheat_flags=_convert_to_serializable(cheat_flags),
            emotion_distribution=_convert_to_serializable(emotion_distribution),
            avg_hrv_rmssd=_convert_to_serializable(avg_hrv_rmssd),
            stress_spike_detected=stress_spike,
            hr_bpm=_convert_to_serializable(hr_bpm),
            gaze_metrics=_convert_to_serializable(gaze_metrics_rt),
        )
        return {"status": "saved", "record": record}
    except Exception as e:
        logger.error(f"[Examiney][VideoChunk] save_video_signals failed: {e}")
        return {"status": "partial", "error": str(e)}


# ── Scoring ────────────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    question_id: str
    transcript: str
    ideal_answer: str
    stage: str = "intro"


@app.post("/score/response", response_model=ResponseScore, tags=["Scoring"],
          summary="Score transcript: semantic + VADER sentiment → combined 0-10")
async def score_candidate_response(req: ScoreRequest):
    if not req.transcript.strip():
        _err("transcript must not be empty.", "MISSING_TRANSCRIPT", 400)
    if not req.ideal_answer.strip():
        _err("ideal_answer must not be empty.", "MISSING_IDEAL_ANSWER", 400)
    try:
        return score_response(req.question_id, req.transcript, req.ideal_answer, stage=req.stage)
    except Exception as e:
        _err(f"Scoring failed: {e}", "SCORING_FAILED")


# ── Calibration ────────────────────────────────────────────────────────────────

class CalibrationStartResponse(BaseModel):
    session_id: str
    calibration_points: List[List[float]]


class IrisSampleModel(BaseModel):
    x: float
    y: float


class PointMeasurementModel(BaseModel):
    screen_x: float
    screen_y: float
    iris_samples: List[IrisSampleModel]


class CalibrationSubmitRequest(BaseModel):
    session_id: str
    measurements: List[PointMeasurementModel]


class CalibrationSubmitResponse(BaseModel):
    session_id: str
    calibration_quality_score: float
    needs_recalibration: bool
    baseline_gaze_variance: float
    baseline_blink_rate: float
    neurodiversity_adjustment: float


@app.post("/calibration/start", response_model=CalibrationStartResponse, tags=["Calibration"],
          summary="Begin calibration session (optionally tied to an interview session_id)")
async def calibration_start(session_id: Optional[str] = None):
    sid = session_id or str(uuid.uuid4())
    return CalibrationStartResponse(
        session_id=sid,
        calibration_points=[[x, y] for x, y in get_calibration_points()],
    )


@app.post("/calibration/submit", response_model=CalibrationSubmitResponse, tags=["Calibration"],
          summary="Finalise calibration: fit affine transform, save JSON, upload to Cloudinary")
async def calibration_submit(req: CalibrationSubmitRequest):
    if not req.measurements:
        _err("No measurements provided.", "MISSING_MEASUREMENTS", 400)
    measurements = [
        PointMeasurement(
            screen_x=pm.screen_x,
            screen_y=pm.screen_y,
            iris_samples=[IrisSample(x=s.x, y=s.y) for s in pm.iris_samples],
        )
        for pm in req.measurements
    ]
    try:
        result: CalibrationResult = run_calibration(req.session_id, measurements)
    except Exception as e:
        _err(f"Calibration failed: {e}", "CALIBRATION_FAILED")

    # Persist calibration to Supabase — no local file, no Cloudinary.
    # Automatically deleted when the session row is deleted.
    try:
        from dataclasses import asdict as _asdict
        supabase_client.save_calibration_data(req.session_id, _asdict(result))
        logger.info(f"[Examiney][Calibration] Saved to Supabase for session={req.session_id}")
    except Exception as e:
        logger.warning(f"[Examiney][Calibration] DB save failed ({e}) — calibration will not persist across restarts")

    return CalibrationSubmitResponse(
        session_id=result.session_id,
        calibration_quality_score=result.calibration_quality_score,
        needs_recalibration=result.needs_recalibration,
        baseline_gaze_variance=result.baseline_gaze_variance,
        baseline_blink_rate=result.baseline_blink_rate,
        neurodiversity_adjustment=result.neurodiversity_adjustment,
    )


# ── Finalize (OCEAN pipeline) ──────────────────────────────────────────────────

class FinalizeConfig(BaseModel):
    model:      Optional[str] = None
    ollama_url: Optional[str] = None


@app.post("/session/{session_id}/finalize", tags=["Session"],
          summary="Aggregate all responses → OCEAN scores → role recommendation → Supabase")
async def finalize_session(session_id: str, config: FinalizeConfig = FinalizeConfig()):
    logger.info(f"[Examiney][Finalize] START session={session_id}")
    session = supabase_client.get_session(session_id)
    if not session:
        _err(f"Session '{session_id}' not found.", "SESSION_NOT_FOUND", 404)

    from services.database.supabase_client import _get_client
    c = _get_client()
    qr_data = (
        c.table("question_responses").select("*").eq("session_id", session_id).execute()
    ).data or []

    # Reconstruct ResponseScore objects
    scores: List[ResponseScore] = []
    for row in qr_data:
        raw_sent = row.get("sentiment") or {}
        sent = SentimentScores(
            compound=raw_sent.get("compound", 0.0),
            pos=raw_sent.get("pos", 0.0),
            neg=raw_sent.get("neg", 0.0),
            neu=raw_sent.get("neu", 1.0),
        )
        scores.append(ResponseScore(
            question_id=row["question_id"],
            transcript=row.get("transcript", ""),
            semantic_score=row.get("semantic_score", 0.0),
            sentiment=sent,
            engagement_flag=bool(row.get("transcript_flagged", False)),
            combined_score=row.get("combined_score", 0.0),
        ))

    # Build InterviewScript from stored questions
    from services.question_gen.models import AnswerKey, Question
    raw_qs = session.get("questions") or []
    script_qs = [
        Question(
            id=q.get("id", str(uuid.uuid4())),
            stage=q.get("stage", "intro"),
            question=q.get("question", ""),
            time_window_seconds=q.get("time_window_seconds", 75),
            answer_key=AnswerKey(
                critical_keywords=q.get("answer_key", {}).get("critical_keywords", []),
                ideal_sentiment=q.get("answer_key", {}).get("ideal_sentiment", "positive"),
                rubric=q.get("answer_key", {}).get("rubric", ""),
            ),
            ideal_answer=q.get("ideal_answer", ""),
        )
        for q in raw_qs
    ]
    script = InterviewScript(questions=script_qs)

    kwargs: Dict[str, Any] = {
        "scores":          scores,
        "script":          script,
        "job_description": session.get("job_description", ""),
        "session_id":      session_id,
    }
    if config.model:      kwargs["model"]      = config.model
    if config.ollama_url: kwargs["ollama_url"] = config.ollama_url

    # Guard: candidate never took the interview — clean up any stale scores and exit cleanly
    if not qr_data:
        logger.info(f"[Examiney][Finalize] No responses for session={session_id} — deleting stale ocean_reports if any.")
        try:
            c.table("ocean_reports").delete().eq("session_id", session_id).execute()
        except Exception:
            pass
        return {
            "session_id":         session_id,
            "status":             "no_data",
            "message":            "Candidate has not completed the interview — no scores generated.",
            "job_fit_score":      None,
            "success_prediction": None,
        }

    logger.info(f"[Examiney][Finalize] Building OCEAN report — {len(scores)} responses, {len(script_qs)} questions")
    try:
        report = build_ocean_report(**kwargs)
        logger.info(f"[Examiney][Finalize] OCEAN done — job_fit={report.job_fit_score:.1f} prediction={report.success_prediction}")
    except Exception as e:
        logger.error(f"[Examiney][Finalize] OCEAN FAILED: {e}")
        _err(f"OCEAN pipeline failed: {e}", "OCEAN_FAILED")

    try:
        supabase_client.save_ocean_scores(
            session_id=session_id,
            openness=report.ocean_scores.openness,
            conscientiousness=report.ocean_scores.conscientiousness,
            extraversion=report.ocean_scores.extraversion,
            agreeableness=report.ocean_scores.agreeableness,
            neuroticism=report.ocean_scores.neuroticism,
            job_fit_score=report.job_fit_score,
            success_prediction=report.success_prediction,
            role_recommendation=report.role_recommendation,
            ocean_confidence=report.ocean_confidence,
            trait_coverage=report.trait_coverage,
            stages_covered=report.stages_covered,
        )
    except Exception as e:
        _err(f"Failed to save OCEAN scores: {e}", "DB_WRITE_FAILED")

    return {
        "session_id":          session_id,
        "ocean_scores":        report.ocean_scores.model_dump(),
        "job_fit_score":       report.job_fit_score,
        "success_prediction":  report.success_prediction,
        "role_recommendation": report.role_recommendation,
        "questions_scored":    report.questions_scored,
    }


# ── Sessions list (admin dashboard) ───────────────────────────────────────────

@app.get("/sessions", tags=["Session"],
         summary="List all sessions ordered by creation date, with OCEAN summaries")
async def list_all_sessions():
    try:
        return supabase_client.list_all_sessions()
    except Exception as e:
        _err(f"Failed to list sessions: {e}", "LIST_SESSIONS_FAILED")


# ── Post-session async transcription ──────────────────────────────────────────

@app.post("/session/{session_id}/transcribe", tags=["Session"],
          summary="Download Cloudinary audio → Whisper → update transcripts (fires after interview ends)")
async def transcribe_session(session_id: str):
    """Downloads each question's audio from Cloudinary, runs Whisper transcription,
    scores with sentence-transformers + VADER, and writes results back to Supabase.
    Called once per session from the candidate thank-you page.
    """
    from services.database.supabase_client import _get_client as _sb
    logger.info(f"[Examiney][Transcribe] START session={session_id}")

    c = _sb()
    rows = (
        c.table("question_responses")
        .select("question_id,audio_url,video_url,transcript,question_text,ideal_answer")
        .eq("session_id", session_id)
        .execute()
    ).data or []

    logger.info(f"[Examiney][Transcribe] Found {len(rows)} question_response rows for session={session_id}")

    updated = 0
    errors: List[str] = []
    for row in rows:
        aurl = row.get("audio_url") or row.get("video_url")
        vurl = row.get("video_url")
        qid  = row.get("question_id", "?")
        if not aurl:
            logger.debug(f"[Examiney][Transcribe] q={qid} has no audio_url or video_url — skipping")
            continue
        if row.get("transcript", "").strip():
            logger.debug(f"[Examiney][Transcribe] q={qid} already transcribed — skipping")
            continue

        tmp_path: Optional[str] = None
        try:
            logger.info(f"[Examiney][Transcribe] q={qid} — downloading from Cloudinary: {aurl[:60]}…")
            tmp_path = _download_to_tmp(aurl, suffix=".webm")
            logger.info(f"[Examiney][Transcribe] q={qid} — running Whisper on {os.path.getsize(tmp_path)} bytes")
            transcript, flagged = _transcribe(tmp_path)
            logger.info(f"[Examiney][Transcribe] q={qid} — transcript={repr(transcript[:80])} flagged={flagged}")

            supabase_client.update_transcript(
                session_id=session_id,
                question_id=qid,
                transcript=transcript,
                transcript_flagged=flagged,
            )
            updated += 1

            # Score transcript (lightweight — OCEAN pipeline runs in /finalize)
            effective = transcript or "[NO RESPONSE — transcription unavailable]"
            try:
                score: ResponseScore = score_response(qid, effective, row.get("ideal_answer", "") or "", stage=row.get("stage", "intro"))
                supabase_client.save_question_response(
                    session_id=session_id,
                    question_id=qid,
                    question_text=row.get("question_text", "") or "",
                    ideal_answer=row.get("ideal_answer", "") or "",
                    transcript=transcript,
                    transcript_flagged=flagged,
                    semantic_score=score.semantic_score,
                    sentiment=score.sentiment.model_dump(),
                    combined_score=score.combined_score,
                    video_url=vurl,
                    audio_url=aurl,
                )
                logger.info(f"[Examiney][Transcribe] q={qid} scored — semantic={score.semantic_score:.2f} combined={score.combined_score:.2f}")
            except Exception as e:
                logger.error(f"[Examiney][Transcribe] q={qid} scoring failed: {e}")
                supabase_client.log_error("TranscribeScore", str(e), session_id)
        except Exception as e:
            err_msg = f"q={qid}: {e}"
            logger.error(f"[Examiney][Transcribe] ERROR {err_msg}")
            errors.append(err_msg)
            supabase_client.log_error("Transcribe", err_msg, session_id)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    logger.info(f"[Examiney][Transcribe] DONE session={session_id} — transcribed={updated} errors={len(errors)}")
    return {"session_id": session_id, "transcribed": updated, "errors": errors}


@app.post("/session/{session_id}/process-video", tags=["Session"],
          summary="Download Cloudinary videos → GazeFollower metrics → Supabase")
async def process_video_session(session_id: str):
    """Runs heavier video-only processing after the interview ends.

    - Downloads each question's video_url from Cloudinary
    - Runs GazeFollower (or placeholder) to compute gaze direction metrics
    - Stores results into video_signals.gaze_metrics
    """
    from services.database.supabase_client import _get_client as _sb
    from services.video_analysis.gaze.gazefollower_runner import run_gazefollower_on_video

    c = _sb()
    rows = (
        c.table("question_responses")
        .select("question_id,video_url")
        .eq("session_id", session_id)
        .execute()
    ).data or []

    processed = 0
    errors: List[str] = []
    for row in rows:
        vurl = row.get("video_url")
        if not vurl:
            continue
        try:
            tmp_path = _download_to_tmp(vurl, suffix=".webm")
            metrics = run_gazefollower_on_video(tmp_path)
            supabase_client.update_video_gaze_metrics(
                session_id=session_id,
                question_id=row["question_id"],
                gaze_metrics=_convert_to_serializable(metrics),
            )
            processed += 1
        except Exception as e:
            err_msg = f"q={row.get('question_id')}: {e}"
            errors.append(err_msg)
            supabase_client.log_error("ProcessVideo", err_msg, session_id)
        finally:
            try:
                if "tmp_path" in locals() and tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    return {"session_id": session_id, "processed": processed, "errors": errors}


# ── Reports & recruiter dashboard ─────────────────────────────────────────────

@app.get("/session/{session_id}/report", tags=["Reports"],
         summary="Full joined candidate report for recruiter dashboard")
async def get_session_report(session_id: str):
    try:
        report = supabase_client.get_candidate_full_report(session_id)
        if not report.get("session"):
            _err(f"Session '{session_id}' not found.", "SESSION_NOT_FOUND", 404)
        return report
    except HTTPException:
        raise
    except Exception as e:
        _err(f"Failed to fetch report: {e}", "REPORT_FETCH_FAILED")


@app.get("/opening/{job_opening_id}/candidates", tags=["Reports"],
         summary="All sessions for a job opening with OCEAN summaries")
async def list_candidates(job_opening_id: str):
    try:
        return supabase_client.list_sessions_by_opening(job_opening_id)
    except Exception as e:
        _err(f"Failed to list sessions: {e}", "LIST_SESSIONS_FAILED")


@app.delete("/opening/{job_opening_id}", tags=["Session"],
            summary="Delete opening: delete all sessions (and their media) for this job_opening_id")
async def delete_opening(job_opening_id: str):
    try:
        from services.database.supabase_client import _get_client as _sb

        c = _sb()
        rows = (
            c.table("sessions")
            .select("session_id")
            .eq("job_opening_id", job_opening_id)
            .execute()
        ).data or []
        deleted = 0
        for row in rows:
            sid = row.get("session_id")
            if not sid:
                continue
            try:
                supabase_client.delete_session(sid)
                deleted += 1
            except Exception as e:
                supabase_client.log_error("DeleteOpeningSession", str(e), sid)
        return {"status": "deleted", "job_opening_id": job_opening_id, "sessions_deleted": deleted}
    except Exception as e:
        _err(f"Failed to delete opening: {e}", "DELETE_OPENING_FAILED")


@app.delete("/session/{session_id}", tags=["Session"],
            summary="Delete session: purge all Drive files then all Supabase rows atomically")
async def delete_session(session_id: str):
    try:
        supabase_client.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        _err(f"Failed to delete session: {e}", "DELETE_FAILED")


# ── Fire-and-forget post-session pipeline ─────────────────────────────────────

def _finalize_ocean_inline(session_id: str) -> None:
    """Run OCEAN finalization directly — avoids a fragile self-HTTP call to localhost."""
    from services.database.supabase_client import _get_client as _sb
    from services.question_gen.models import AnswerKey, Question

    c = _sb()
    session = supabase_client.get_session(session_id)
    if not session:
        logger.warning(f"[Examiney][FinalizeInline] Session {session_id} not found")
        return

    qr_data = (
        c.table("question_responses").select("*").eq("session_id", session_id).execute()
    ).data or []

    if not qr_data:
        try:
            c.table("ocean_reports").delete().eq("session_id", session_id).execute()
        except Exception:
            pass
        logger.info(f"[Examiney][FinalizeInline] No responses for {session_id} — skipping OCEAN")
        return

    scores: List[ResponseScore] = []
    for row in qr_data:
        raw_sent = row.get("sentiment") or {}
        sent = SentimentScores(
            compound=raw_sent.get("compound", 0.0),
            pos=raw_sent.get("pos", 0.0),
            neg=raw_sent.get("neg", 0.0),
            neu=raw_sent.get("neu", 1.0),
        )
        scores.append(ResponseScore(
            question_id=row["question_id"],
            transcript=row.get("transcript", ""),
            semantic_score=row.get("semantic_score", 0.0),
            sentiment=sent,
            engagement_flag=bool(row.get("transcript_flagged", False)),
            combined_score=row.get("combined_score", 0.0),
        ))

    raw_qs = session.get("questions") or []
    script_qs = [
        Question(
            id=q.get("id", str(uuid.uuid4())),
            stage=q.get("stage", "intro"),
            question=q.get("question", ""),
            time_window_seconds=q.get("time_window_seconds", 75),
            answer_key=AnswerKey(
                critical_keywords=q.get("answer_key", {}).get("critical_keywords", []),
                ideal_sentiment=q.get("answer_key", {}).get("ideal_sentiment", "positive"),
                rubric=q.get("answer_key", {}).get("rubric", ""),
            ),
            ideal_answer=q.get("ideal_answer", ""),
        )
        for q in raw_qs
    ]
    script = InterviewScript(questions=script_qs)

    try:
        report = build_ocean_report(
            scores=scores,
            script=script,
            job_description=session.get("job_description", ""),
            session_id=session_id,
        )
        
        # Try to save OCEAN scores with 2 retries (socket errors can happen on Windows)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                supabase_client.save_ocean_scores(
                    session_id=session_id,
                    openness=report.ocean_scores.openness,
                    conscientiousness=report.ocean_scores.conscientiousness,
                    extraversion=report.ocean_scores.extraversion,
                    agreeableness=report.ocean_scores.agreeableness,
                    neuroticism=report.ocean_scores.neuroticism,
                    job_fit_score=report.job_fit_score,
                    success_prediction=report.success_prediction,
                    role_recommendation=report.role_recommendation,
                    ocean_confidence=getattr(report, 'ocean_confidence', 'Low'),
                    trait_coverage=getattr(report, 'trait_coverage', None),
                    stages_covered=getattr(report, 'stages_covered', None),
                )
                logger.info(f"[Examiney][FinalizeInline] job_fit={report.job_fit_score:.1f} prediction={report.success_prediction}")
                break  # Success
            except Exception as save_err:
                if attempt < max_retries - 1:
                    logger.warning(f"[Examiney][FinalizeInline] Save attempt {attempt + 1} failed, retrying: {save_err}")
                    time.sleep(0.5)
                else:
                    raise  # Don't retry on final attempt
    except Exception as e:
        logger.error(f"[Examiney][FinalizeInline] OCEAN FAILED: {type(e).__name__}: {e}", exc_info=True)
        supabase_client.log_error("FinalizeInline", f"{type(e).__name__}: {str(e)}", session_id)


def _bg_post_session(session_id: str) -> None:
    """Background: transcribe → score → finalize OCEAN → video signals catch-up.

    A short initial sleep lets any in-flight Cloudinary uploads (e.g. the last
    question's video) complete before we start pulling URLs from the database.
    """
    # Guard: only one pipeline per session at a time
    with _active_sessions_lock:
        if session_id in _active_sessions:
            logger.warning(f"[Examiney][PostSession] session={session_id} already processing — skipping duplicate")
            return
        _active_sessions.add(session_id)

    _section(f"Post-Session Pipeline  session={session_id[:8]}")
    _step("Waiting for uploads to settle (6s)")
    _set_stage(session_id, "transcribing", "Waiting for uploads to settle…", 0, 0)
    time.sleep(6)   # wait for last-question uploads to land in Cloudinary / DB

    from services.database.supabase_client import _get_client as _sb
    c = _sb()

    try:
        # ── Step 1: fetch session + question_responses rows ───────────────────
        session_data = supabase_client.get_session(session_id) or {}
        # Build question-id → stage map so LLM judge uses the right evaluation criteria
        stage_map: Dict[str, str] = {
            q.get("id", ""): q.get("stage", "intro")
            for q in session_data.get("questions", [])
        }

        rows = (
            c.table("question_responses")
            .select("question_id,audio_url,video_url,transcript,question_text,ideal_answer")
            .eq("session_id", session_id)
            .execute()
        ).data or []

        total_qs = len(rows)
        # Fall back to video_url — the frontend sends a single webm containing audio+video
        audio_rows = [r for r in rows if r.get("audio_url") or r.get("video_url")]
        logger.info(f"[Examiney][PostSession] {total_qs} questions, {len(audio_rows)} have audio/video")

        # ── Step 2: transcribe each audio file (parallel download + transcription) ────────────
        # Pre-transcribe all rows in parallel (max 3 concurrent), then score sequentially
        def _transcribe_row(row: Dict[str, Any]) -> Dict[str, Any]:
            """Download and transcribe a single row. Returns result dict."""
            qid = row.get("question_id", "?")
            aurl = row.get("audio_url") or row.get("video_url")
            tmp_path: Optional[str] = None
            try:
                if row.get("transcript", "").strip():
                    return {"question_id": qid, "already_transcribed": True}
                
                tmp_path = _download_to_tmp(aurl, suffix=".webm")
                transcript, flagged = _transcribe(tmp_path)
                logger.info(f"[Examiney][PostSession] q={qid} transcribed: {len(transcript)} chars, flagged={flagged}")
                if not transcript or flagged:
                    logger.warning(f"[Examiney][PostSession] q={qid} — Whisper transcription unavailable")
                return {
                    "question_id": qid,
                    "already_transcribed": False,
                    "transcript": transcript,
                    "flagged": flagged,
                }
            except Exception as e:
                logger.error(f"[Examiney][PostSession] q={qid} transcription FAILED: {e}")
                supabase_client.log_error("PostSessionTranscribe", str(e), session_id)
                return {
                    "question_id": qid,
                    "already_transcribed": False,
                    "transcript": "",
                    "flagged": True,
                    "error": str(e),
                }
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
        
        # Run transcriptions in parallel (max 3 concurrent)
        transcribe_results: Dict[str, Dict[str, Any]] = {}
        rows_to_transcribe = [r for r in audio_rows if not r.get("transcript", "").strip()]
        rows_already_done = [r for r in audio_rows if r.get("transcript", "").strip()]
        
        if rows_to_transcribe:
            with ThreadPoolExecutor(max_workers=min(3, len(rows_to_transcribe))) as executor:
                futures = {executor.submit(_transcribe_row, row): row["question_id"] for row in rows_to_transcribe}
                done = len(rows_already_done)
                for future in as_completed(futures):
                    result = future.result()
                    transcribe_results[result["question_id"]] = result
                    done += 1
                    _set_stage(session_id, "transcribing", f"Transcribing audio ({done}/{len(audio_rows)})", done, len(audio_rows))
        
        # Mark already-transcribed rows as done
        done = len(rows_already_done)
        for row in rows_already_done:
            qid = row.get("question_id", "?")
            logger.debug(f"[Examiney][PostSession] q={qid} already transcribed — skip")
            transcribe_results[qid] = {"question_id": qid, "already_transcribed": True}
            done += 1
            _set_stage(session_id, "transcribing", f"Transcribing audio ({done}/{len(audio_rows)})", done, len(audio_rows))
        
        # ── Step 2b: Score all transcribed responses (sequential, same as before) ────────────
        done = 0
        for row in audio_rows:
            aurl = row.get("audio_url") or row.get("video_url")
            vurl = row.get("video_url")
            qid  = row.get("question_id", "?")
            
            # Get transcription result
            trans_result = transcribe_results.get(qid, {})
            if trans_result.get("already_transcribed"):
                done += 1
                continue
            
            transcript = trans_result.get("transcript", "")
            flagged = trans_result.get("flagged", True)
            
            # Update transcript in database
            supabase_client.update_transcript(
                session_id=session_id, question_id=qid,
                transcript=transcript, transcript_flagged=flagged,
            )
            
            # Score immediately after transcription
            _set_stage(session_id, "scoring", f"Scoring response {done + 1}/{len(audio_rows)}", done, len(audio_rows))
            try:
                from services.scoring.llm_marker import judge_response as _judge, mark_response as _mark
                ideal    = row.get("ideal_answer", "") or ""
                qtxt     = row.get("question_text", "") or ""
                q_stage  = stage_map.get(qid, "intro")
                
                # Score the (possibly empty) transcript
                try:
                    score: ResponseScore = score_response(qid, transcript or "[NO RESPONSE]", ideal, stage=q_stage)
                except Exception as score_err:
                    logger.error(f"[Examiney][PostSession] q={qid} score_response FAILED: {type(score_err).__name__}: {score_err}", exc_info=True)
                    raise
                
                # Judge the response
                try:
                    judgment = _judge(question_text=qtxt, ideal_answer=ideal, transcript=transcript or "[NO RESPONSE]", stage=q_stage)
                except Exception as judge_err:
                    logger.error(f"[Examiney][PostSession] q={qid} judge_response FAILED: {type(judge_err).__name__}: {judge_err}", exc_info=True)
                    judgment = {
                        "verdict": "not_attempted",
                        "verdict_reason": "Scoring engine error — transcript unavailable or corrupted",
                        "key_gaps": ["Unable to evaluate due to system error"],
                        "strengths": [],
                        "score": 0.0,
                    }
                
                verdict        = judgment.get("verdict", "incomplete")
                verdict_reason = judgment.get("verdict_reason", "")
                key_gaps       = judgment.get("key_gaps", [])
                strengths      = judgment.get("strengths", [])
                llm_score      = float(judgment.get("score", score.combined_score or 0.0))
                final_score    = round((llm_score * 0.7 + (score.combined_score or 0.0) * 0.3), 2)
                
                # Dimension scores
                marks: Dict[str, Any] = {}
                try:
                    marks = _mark(question_text=qtxt, ideal_answer=ideal, transcript=transcript or "[NO RESPONSE]", stage=q_stage)
                except Exception as mark_err:
                    logger.warning(f"[Examiney][PostSession] q={qid} mark_response failed (non-critical): {mark_err}")
                
                supabase_client.save_question_response(
                    session_id=session_id, question_id=qid,
                    question_text=qtxt, ideal_answer=ideal,
                    transcript=transcript, transcript_flagged=flagged,
                    semantic_score=score.semantic_score,
                    sentiment=score.sentiment.model_dump(),
                    combined_score=final_score,
                    technical_score=marks.get("technical"),
                    communication_score=marks.get("communication"),
                    behavioral_score=marks.get("behavioral"),
                    engagement_score=marks.get("engagement"),
                    authenticity_score=marks.get("authenticity"),
                    video_url=vurl,
                    audio_url=aurl,
                    llm_verdict=verdict,
                    llm_verdict_reason=verdict_reason,
                    llm_key_gaps=key_gaps,
                    llm_strengths=strengths,
                )
                logger.info(f"[Examiney][PostSession] q={qid} stage={q_stage} verdict={verdict} score={final_score:.2f}")
            except Exception as se:
                logger.error(f"[Examiney][PostSession] q={qid} scoring FAILED: {type(se).__name__}: {se}", exc_info=True)
                supabase_client.log_error("PostSessionScore", f"{type(se).__name__}: {str(se)}", session_id)
            done += 1

        # ── Step 3: finalize OCEAN (inline — no self-HTTP call) ───────────────
        _set_stage(session_id, "finalizing", "Computing OCEAN personality profile…", total_qs, total_qs)
        logger.info(f"[Examiney][PostSession] Running OCEAN finalize inline for session={session_id}")
        try:
            _finalize_ocean_inline(session_id)
        except Exception as e:
            logger.error(f"[Examiney][PostSession] Finalize FAILED: {e}")
            supabase_client.log_error("PostSessionFinalize", str(e), session_id)

        # ── Steps 4+5: Video signal catch-up (gaze + emotion + rPPG) ─────────────
        # Each video is downloaded ONCE and all three analyses run on it.
        # Questions are processed in parallel (up to 3 concurrent) to cut wall time.
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from services.video_analysis.gaze.gazefollower_runner import run_gazefollower_on_video
            from services.video_analysis.calibration.calibration_runner import load_calibration
            from services.video_analysis.emotion_analyzer import analyze_emotions_from_video
            from services.video_analysis.rppg import analyze_rppg_from_video

            # Fetch existing video_signals state for this session
            vs_rows = (
                c.table("video_signals")
                .select("question_id,gaze_metrics,emotion_distribution,avg_hrv_rmssd,cheat_flags")
                .eq("session_id", session_id)
                .execute()
            ).data or []
            vs_map: Dict[str, Dict] = {r["question_id"]: r for r in vs_rows}

            # All questions with a saved video URL
            all_vrows = (
                c.table("question_responses")
                .select("question_id,video_url")
                .eq("session_id", session_id)
                .execute()
            ).data or []
            all_vrows = [r for r in all_vrows if r.get("video_url")]

            cal_data: Dict[str, Any] = {}
            try:
                cal_data = load_calibration(session_id)
            except Exception:
                pass

            def _process_video_signals(vrow: Dict[str, Any]) -> None:
                """Download video once, run gaze + emotion + rPPG, save results."""
                qid  = vrow.get("question_id", "?")
                vurl = vrow.get("video_url", "")
                existing = vs_map.get(qid, {})

                gaze_ok = (existing.get("gaze_metrics") or {}).get("status") == "ok"
                existing_emotion = existing.get("emotion_distribution") or {}
                emotion_ok = isinstance(existing_emotion, dict) and len(existing_emotion) > 1
                rppg_ok    = existing.get("avg_hrv_rmssd") is not None
                # head_pose_ok: cheat_flags already contains a head_pose sub-dict
                existing_cf = existing.get("cheat_flags") or {}
                head_pose_ok = isinstance(existing_cf.get("head_pose"), dict) and \
                               existing_cf["head_pose"].get("head_pose_available") is True

                if gaze_ok and emotion_ok and rppg_ok and head_pose_ok:
                    logger.debug(f"[Examiney][PostSession] all signals done q={qid} — skip")
                    return

                tmp_v: Optional[str] = None
                try:
                    logger.info(f"[Examiney][PostSession] video signals q={qid} downloading…")
                    tmp_v = _download_to_tmp(vurl, suffix=".webm")

                    # ── Gaze ──────────────────────────────────────────────────
                    if not gaze_ok:
                        try:
                            metrics = run_gazefollower_on_video(tmp_v, session_id=session_id, calibration_data=cal_data)
                            # Retry logic for socket timeouts
                            max_gaze_retries = 2
                            for attempt in range(max_gaze_retries):
                                try:
                                    supabase_client.update_video_gaze_metrics(
                                        session_id=session_id, question_id=qid, gaze_metrics=_convert_to_serializable(metrics),
                                    )
                                    logger.info(f"[Examiney][PostSession] gaze q={qid}: status={metrics.get('status')}")
                                    break  # Success
                                except Exception as gaze_err:
                                    if attempt < max_gaze_retries - 1 and ("10035" in str(gaze_err) or "socket" in str(gaze_err).lower()):
                                        logger.warning(f"[Examiney][PostSession] Socket error on gaze update, retrying q={qid}")
                                        time.sleep(0.2)
                                    else:
                                        raise
                        except Exception as eg:
                            logger.error(f"[Examiney][PostSession] gaze FAILED q={qid}: {type(eg).__name__}: {eg}")
                            supabase_client.log_error("PostSessionGaze", str(eg), session_id)

                    # ── Emotion + rPPG + Head pose (parallel analysis) ───────────────────
                    # Run all three analyses in parallel for better performance
                    update_payload: Dict[str, Any] = {}
                    analysis_futures = {}
                    
                    with ThreadPoolExecutor(max_workers=3) as analysis_pool:
                        # Submit emotion analysis
                        if not emotion_ok:
                            analysis_futures['emotion'] = analysis_pool.submit(lambda: analyze_emotions_from_video(tmp_v))
                        
                        # Submit rPPG analysis
                        if not rppg_ok:
                            analysis_futures['rppg'] = analysis_pool.submit(lambda: analyze_rppg_from_video(tmp_v))
                        
                        # Submit head pose analysis
                        if not head_pose_ok:
                            def _extract_and_analyze_pose():
                                from services.video_analysis.gaze.head_pose_runner import extract_head_pose_from_video
                                from services.video_analysis.gaze.cheating_detector import analyze_head_pose
                                pose_data = extract_head_pose_from_video(tmp_v)
                                pose_samples_pp  = pose_data.get("samples", [])
                                absent_pct_pp    = float(pose_data.get("face_absent_pct", 0.0))
                                neuro_adj_pp = cal_data.get("neurodiversity_adjustment", 1.0)
                                hp_result = analyze_head_pose(
                                    pose_samples_pp,
                                    neurodiversity_adjustment=neuro_adj_pp,
                                    face_absent_pct=absent_pct_pp,
                                )
                                return {"data": hp_result, "pose_data": pose_data}
                            
                            analysis_futures['head_pose'] = analysis_pool.submit(_extract_and_analyze_pose)
                        
                        # Collect results
                        if 'emotion' in analysis_futures:
                            try:
                                update_payload["emotion_distribution"] = analysis_futures['emotion'].result()
                            except Exception as ee:
                                logger.error(f"[Examiney][PostSession] emotion FAILED q={qid}: {ee}")
                        
                        if 'rppg' in analysis_futures:
                            try:
                                rppg_result = analysis_futures['rppg'].result()
                                if rppg_result.get("data_available"):
                                    update_payload["avg_hrv_rmssd"]        = rppg_result.get("avg_hrv_rmssd")
                                    update_payload["hr_bpm"]               = rppg_result.get("hr_bpm")
                                    update_payload["stress_spike_detected"] = rppg_result.get("stress_spike_detected", False)
                            except Exception as er:
                                logger.error(f"[Examiney][PostSession] rPPG FAILED q={qid}: {er}")
                        
                        if 'head_pose' in analysis_futures:
                            try:
                                hp_result_data = analysis_futures['head_pose'].result()
                                hp_result = hp_result_data.get("data", {})
                                pose_data = hp_result_data.get("pose_data", {})
                                absent_pct_pp = float(pose_data.get("face_absent_pct", 0.0))
                                
                                if hp_result:
                                    merged_cf = dict(existing_cf)
                                    merged_cf["head_pose"] = hp_result
                                    gaze_eff = float(merged_cf.get("effective_score", 0))
                                    hp_eff   = float(hp_result.get("head_pose_effective_score", 0))
                                    combined_eff = gaze_eff + hp_eff
                                    if combined_eff >= 5:
                                        merged_cf["risk_level"] = "high"
                                    elif combined_eff >= 2 and merged_cf.get("risk_level") != "high":
                                        merged_cf["risk_level"] = "medium"
                                    update_payload["cheat_flags"] = merged_cf
                                    logger.info(
                                        f"[Examiney][PostSession] head_pose q={qid}: "
                                        f"sustained={hp_result.get('sustained_head_turn')} "
                                        f"absent={absent_pct_pp:.1%} "
                                        f"score={hp_result.get('head_pose_score')} "
                                        f"risk={merged_cf.get('risk_level')}"
                                    )
                            except Exception as ehp:
                                logger.error(f"[Examiney][PostSession] head_pose FAILED q={qid}: {ehp}")

                    if update_payload:
                        try:
                            # Convert NumPy types to JSON-serializable Python types
                            serializable_payload = _convert_to_serializable(update_payload)
                            
                            # Upsert with retry logic for socket errors
                            max_retries = 2
                            for attempt in range(max_retries):
                                try:
                                    (
                                        c.table("video_signals")
                                        .upsert(
                                            {"session_id": session_id, "question_id": qid, **serializable_payload},
                                            on_conflict="session_id,question_id",
                                        )
                                        .execute()
                                    )
                                    logger.info(
                                        f"[Examiney][PostSession] emotion/rPPG saved q={qid} "
                                        f"emotion_keys={list(serializable_payload.get('emotion_distribution', {}).keys())[:3]} "
                                        f"rmssd={serializable_payload.get('avg_hrv_rmssd')}"
                                    )
                                    break  # Success
                                except Exception as upsert_err:
                                    if attempt < max_retries - 1 and ("10035" in str(upsert_err) or "socket" in str(upsert_err).lower()):
                                        logger.warning(f"[Examiney][PostSession] Socket error, retrying emotion/rPPG save for q={qid}")
                                        time.sleep(0.2)
                                    else:
                                        raise
                        except Exception as db_err:
                            logger.error(f"[Examiney][PostSession] emotion/rPPG DB save FAILED q={qid}: {type(db_err).__name__}: {db_err}", exc_info=False)

                except Exception as e:
                    logger.error(f"[Examiney][PostSession] video signals FAILED q={qid}: {e}")
                    supabase_client.log_error("PostSessionVideoSignals", str(e), session_id)
                finally:
                    if tmp_v and os.path.exists(tmp_v):
                        try:
                            os.unlink(tmp_v)
                        except Exception:
                            pass

            if all_vrows:
                _set_stage(session_id, "analyzing_gaze", f"Analysing video signals (0/{len(all_vrows)})…", 0, len(all_vrows))
                completed = 0
                # Process up to 3 videos concurrently — I/O overlap on downloads
                with ThreadPoolExecutor(max_workers=min(3, len(all_vrows))) as pool:
                    futs = {pool.submit(_process_video_signals, vrow): vrow for vrow in all_vrows}
                    for fut in as_completed(futs):
                        completed += 1
                        _set_stage(
                            session_id, "analyzing_gaze",
                            f"Analysing video signals ({completed}/{len(all_vrows)})…",
                            completed, len(all_vrows),
                        )
                        try:
                            fut.result()
                        except Exception as fut_err:
                            logger.error(f"[Examiney][PostSession] video signal future error: {fut_err}")
            else:
                logger.info(f"[Examiney][PostSession] No videos to analyse — skipping video signals step")

        except Exception as e:
            logger.error(f"[Examiney][PostSession] video signals catch-up FAILED: {e}")

        # ── Final stage: Dummy processing (1 minute) to appear seamless ─────────────
        # All real work is done; this just provides a final "Processing..." state
        # so the candidate doesn't see "complete" too quickly and wonder if something is wrong
        logger.info(f"[Examiney][PostSession] Starting 1-minute finalization stage for seamless UX")
        _set_stage(session_id, "finalizing", "Finalizing results…", 0, 1)
        import time
        start_finalize = time.time()
        finalize_duration = 60  # 1 minute
        while (time.time() - start_finalize) < finalize_duration:
            elapsed = time.time() - start_finalize
            progress = int((elapsed / finalize_duration) * 100)
            _set_stage(
                session_id, "finalizing",
                f"Finalizing results… {progress}%",
                min(progress, 100), 100
            )
            time.sleep(2)  # Update every 2 seconds
        logger.info(f"[Examiney][PostSession] Finalization complete")

    finally:
        _clear_stage(session_id)
        logger.info(f"[Examiney][PostSession] COMPLETE session={session_id}")


@app.post("/session/{session_id}/process", tags=["Session"],
          summary="Fire background pipeline: Whisper + OCEAN + GazeFollower. Returns 202 immediately.")
async def start_post_session_processing(session_id: str):
    """Called by the thank-you page once all recordings are uploaded.
    Immediately returns 202. Transcription + OCEAN + GazeFollower run in background.
    Poll /session/{id}/status to check when OCEAN results are ready.
    """
    logger.info(f"[Examiney][Process] Queuing background pipeline for session={session_id}")
    session = supabase_client.get_session(session_id)
    if not session:
        _err(f"Session '{session_id}' not found.", "SESSION_NOT_FOUND", 404)

    with _active_sessions_lock:
        already_running = session_id in _active_sessions

    if already_running:
        logger.info(f"[Examiney][Process] session={session_id} already in-flight — returning processing status")
        return {"status": "processing", "session_id": session_id, "message": "Pipeline already running."}

    threading.Thread(target=_bg_post_session, args=(session_id,), daemon=True).start()
    return {"status": "processing", "session_id": session_id, "message": "Pipeline started. Poll /session/{id}/status for results."}


# ── Session status (polled by candidate thank-you page) ───────────────────────

@app.get("/session/{session_id}/status", tags=["Session"],
         summary="Check pipeline stage + OCEAN readiness for a session")
async def session_status(session_id: str):
    """Returns live processing stage so the frontend can poll with meaningful labels.

    Response fields:
      status         — "processing" | "ready" | "not_found" | "error"
      stage          — current pipeline stage (only when processing)
      stage_label    — human-readable description of current stage
      stage_done     — items completed in current stage
      stage_total    — total items in current stage
      transcripts_done / questions_total — transcript progress counts
    """
    from services.database.supabase_client import _get_client as _sb
    try:
        c = _sb()
        session = supabase_client.get_session(session_id)
        if not session:
            return {"status": "not_found", "session_id": session_id}

        # ── Check OCEAN ready ──────────────────────────────────────────────────
        ocean_rows = (
            c.table("ocean_reports")
            .select("session_id,job_fit_score,success_prediction")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        ).data or []

        if ocean_rows:
            row = ocean_rows[0]
            logger.info(f"[Examiney][Status] {session_id[:8]} READY — job_fit={row.get('job_fit_score')}")
            return {
                "status":             "ready",
                "session_id":         session_id,
                "job_fit_score":      row.get("job_fit_score"),
                "success_prediction": row.get("success_prediction"),
            }

        # ── In-flight stage info ───────────────────────────────────────────────
        with _stages_lock:
            stage_info = _processing_stages.get(session_id, {})

        total_qs = len(session.get("questions") or [])
        done_qs  = (
            c.table("question_responses")
            .select("question_id", count="exact")
            .eq("session_id", session_id)
            .neq("transcript", "")
            .execute()
        ).count or 0

        stage       = stage_info.get("stage", "transcribing")
        stage_label = stage_info.get("label", f"Transcribing audio ({done_qs}/{total_qs})…")
        stage_done  = stage_info.get("done", done_qs)
        stage_total = stage_info.get("total", total_qs)
        stage_ts    = stage_info.get("ts", 0.0)
        stage_age   = round(time.time() - stage_ts) if stage_ts else 0
        # Mark as stalled if the same stage has been active for > 4 minutes
        stalled     = stage_ts > 0 and stage_age > 240

        logger.info(f"[Examiney][Status] {session_id[:8]} PROCESSING — {stage}: {stage_done}/{stage_total} age={stage_age}s")
        return {
            "status":            "processing",
            "session_id":        session_id,
            "stage":             stage,
            "stage_label":       stage_label,
            "stage_done":        stage_done,
            "stage_total":       stage_total,
            "stage_age_seconds": stage_age,
            "stalled":           stalled,
            "transcripts_done":  done_qs,
            "questions_total":   total_qs,
        }
    except Exception as e:
        logger.error(f"[Examiney][Status] ERROR session={session_id}: {e}")
        return {"status": "error", "session_id": session_id, "detail": str(e)}


# ── Admin: database + media reset ─────────────────────────────────────────────

@app.delete("/admin/reset-database", tags=["Admin"],
            summary="Purge all Supabase data and Cloudinary media. Requires X-Admin-Secret header.")
async def admin_reset_database(x_admin_secret: str = Header(..., alias="X-Admin-Secret")):
    """Full environment reset for development/testing.

    - Validates X-Admin-Secret against the ADMIN_SECRET env variable.
    - Deletes all Cloudinary resources under candidates/ (video + raw).
    - Truncates all Supabase tables in FK-safe order.

    WARNING: irreversible. All candidate data, transcripts, and media will be permanently deleted.
    """
    expected = os.getenv("ADMIN_SECRET", "").strip()
    if not expected:
        _err("ADMIN_SECRET env variable is not set. Cannot perform reset.", "ADMIN_SECRET_NOT_CONFIGURED", 500)
    if x_admin_secret != expected:
        logger.warning(f"[Examiney][AdminReset] Unauthorised attempt with secret={x_admin_secret[:4]}...")
        _err("Invalid admin secret.", "UNAUTHORIZED", 401)

    logger.warning("[Examiney][AdminReset] FULL DATABASE + MEDIA RESET INITIATED")

    # 1. Delete Cloudinary assets
    cloud_video_deleted = cloudinary_client.delete_by_prefix(prefix="candidates/", resource_type="video")
    cloud_raw_deleted   = cloudinary_client.delete_by_prefix(prefix="candidates/", resource_type="raw")
    logger.info(f"[Examiney][AdminReset] Cloudinary: {cloud_video_deleted} video, {cloud_raw_deleted} raw assets deleted")

    # 2. Truncate Supabase tables
    table_results = supabase_client.truncate_all_tables()
    total_rows = sum(v for v in table_results.values() if v >= 0)
    logger.info(f"[Examiney][AdminReset] Supabase: {total_rows} total rows deleted across {len(table_results)} tables")

    # 3. Clear any in-flight stage trackers
    with _stages_lock:
        _processing_stages.clear()

    return {
        "status":              "reset_complete",
        "cloudinary_deleted":  {"video": cloud_video_deleted, "raw": cloud_raw_deleted},
        "supabase_tables":     table_results,
    }


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"], summary="Global API health check")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/session/{session_id}/health", tags=["Health"],
         summary="Per-session heartbeat (frontend pings every 30 s)")
async def session_health(session_id: str):
    try:
        session = supabase_client.get_session(session_id)
        if not session:
            return {"status": "not_found", "session_id": session_id}
        return {"status": "healthy", "session_id": session_id}
    except Exception:
        return {"status": "error", "session_id": session_id}

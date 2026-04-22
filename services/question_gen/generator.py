import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from .models import AnswerKey, InterviewScript, Question
from .prompts import SYSTEM_PROMPT, build_batch_prompt

logger = logging.getLogger(__name__)

OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

# ── Hosted API keys (set at least one for fast generation) ──────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEY2 = os.getenv("GEMINI_API_KEY2", "")
# Primary model — override with GEMINI_MODEL env var.
# Free-tier safe choices (Google AI Studio): gemini-2.0-flash-lite, gemini-1.5-flash, gemini-2.0-flash
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
# Fallback chain tried in order when the primary model returns 404/429/503
_GEMINI_FALLBACK_MODELS = ["gemini-1.5-flash", "gemini-1.5-flash-8b"]
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Groq: free tier, ~800 tok/s, OpenAI-compatible.
# Recommended models: "llama-3.3-70b-versatile" (quality) or "qwen-qwq-32b" (Qwen)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

# Default batch sizes — used when no section_counts are provided
_DEFAULT_BATCHES = [
    ("intro",       3),
    ("technical",   4),
    ("technical",   3),
    ("behavioral",  4),
    ("logical",     4),
]

_BATCH_MAX = 4   # max questions per LLM call


def _build_batches(section_counts: dict) -> list:
    """Build (stage, count) tuples from a section_counts dict, splitting large sections."""
    batches = []
    for stage, total in section_counts.items():
        remaining = max(1, int(total))
        while remaining > 0:
            chunk = min(remaining, _BATCH_MAX)
            batches.append((stage, chunk))
            remaining -= chunk
    return batches


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers the model sometimes adds."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(text: str) -> dict:
    """Try direct parse; fall back to balanced-brace extraction of the first {...} block.

    The old greedy regex r"\\{.*\\}" (re.DOTALL) captured from the first '{' to the
    LAST '}' in the text, which breaks whenever the model emits trailing prose after
    the JSON object.  The balanced-brace scanner below finds the first complete object.
    """
    text = _strip_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Balanced-brace scanner: find the first syntactically complete {...}
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
            if depth == 1:
                start = i
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = -1  # keep scanning for the next candidate
    raise ValueError(f"No JSON object found in response: {text[:200]}")


def _default_question(stage: str, idx: int, resume_snippet: str = "", jd_snippet: str = "") -> Question:
    """Fallback question used when a batch fails after retries."""
    context = (resume_snippet or jd_snippet or "the candidate's background")[:80]
    return Question(
        id=f"q{idx}",
        stage=stage,  # type: ignore[arg-type]
        question=f"Tell me about your experience relevant to {context}.",
        time_window_seconds={"intro": 60, "technical": 90, "behavioral": 90, "logical": 60}.get(stage, 75),
        ideal_answer="Describe specific examples with measurable outcomes.",
        answer_key=AnswerKey(
            critical_keywords=["experience", "outcome"],
            ideal_sentiment="confident",
            rubric="1=vague, 5=adequate, 10=specific with outcomes",
        ),
    )


def _call_ollama(
    system: str,
    user: str,
    model: str,
    ollama_url: str,
    retries: int = 3,
) -> dict:
    chat_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        # No "format":"json" — 0.5b model returns 500 with it on longer prompts
        "options": {"temperature": 0.8, "num_predict": 1024},
    }
    # Fallback payload for older Ollama builds that don't support /api/chat
    gen_payload = {
        "model":  model,
        "prompt": f"{system}\n\n{user}",
        "stream": False,
        "options": {"temperature": 0.8, "num_predict": 1024},
    }
    for attempt in range(retries):
        try:
            attempt_start = time.time()
            logger.info(f"[Examiney][Generator]     Ollama attempt {attempt + 1}: POST to {ollama_url}/api/chat...")
            with httpx.Client(timeout=120.0) as client:
                r = client.post(f"{ollama_url}/api/chat", json=chat_payload)
                if r.status_code == 404:
                    logger.info(f"[Examiney][Generator]     /api/chat returned 404, trying /api/generate...")
                    r = client.post(f"{ollama_url}/api/generate", json=gen_payload)
                r.raise_for_status()
                data = r.json()
                content = data.get("message", {}).get("content") or data.get("response", "")
                result = _extract_json(content)
                elapsed = time.time() - attempt_start
                logger.info(f"[Examiney][Generator]     Ollama HTTP completed in {elapsed:.2f}s (attempt {attempt + 1})")
                return result
        except Exception as e:
            elapsed = time.time() - attempt_start
            if attempt < retries - 1:
                logger.warning(f"[Examiney][Generator]     Ollama attempt {attempt + 1} failed in {elapsed:.2f}s: {e}, retrying...")
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Ollama call failed after {retries} attempts: {e}") from e
    raise RuntimeError("Unreachable")


def _call_gemini(system: str, user: str, retries: int = 3, api_key: str = "") -> dict:
    """Call Gemini — free tier, ~1-3s per call.

    Tries GEMINI_MODEL first, then falls back through _GEMINI_FALLBACK_MODELS
    on 404 (model not found) or 429 (rate limit exhausted).
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": 2048},
    }
    models_to_try = [GEMINI_MODEL] + [m for m in _GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    last_error: Exception = RuntimeError("Gemini: no models tried")

    for model_name in models_to_try:
        url = f"{_GEMINI_BASE_URL}/{model_name}:generateContent"
        for attempt in range(retries):
            attempt_start = time.time()
            try:
                # Allow slower responses before falling back to defaults
                with httpx.Client(timeout=45.0) as client:
                    r = client.post(url, params={"key": key}, json=payload)
                    if r.status_code in (404, 400):
                        # Model not available — skip retries, try next model
                        err_body = r.text[:200]
                        logger.warning(
                            f"[Examiney][Generator]     Gemini model '{model_name}' returned {r.status_code}: "
                            f"{err_body} — trying next model"
                        )
                        last_error = RuntimeError(f"Gemini {r.status_code} for model {model_name}: {err_body}")
                        break  # break retry loop → next model
                    if r.status_code == 429:
                        logger.warning(f"[Examiney][Generator]     Gemini model '{model_name}' rate-limited (429) — trying next model")
                        last_error = RuntimeError(f"Gemini 429 rate limit on model {model_name}")
                        break
                    r.raise_for_status()
                    content = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    result = _extract_json(content)
                    elapsed = time.time() - attempt_start
                    logger.info(
                        f"[Examiney][Generator]     Gemini '{model_name}' completed in {elapsed:.2f}s "
                        f"(attempt {attempt + 1})"
                    )
                    return result
            except Exception as e:
                elapsed = time.time() - attempt_start
                last_error = e
                if attempt < retries - 1:
                    logger.warning(
                        f"[Examiney][Generator]     Gemini '{model_name}' attempt {attempt + 1} "
                        f"failed in {elapsed:.2f}s: {e}, retrying…"
                    )
                    time.sleep(2 ** attempt)

    raise RuntimeError(f"Gemini call failed on all models: {last_error}") from last_error


_GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"


def _call_groq(system: str, user: str, retries: int = 3) -> dict:
    """Call Groq API — free tier, ~800 tok/s, OpenAI-compatible.
    Falls back from GROQ_MODEL to llama-3.3-70b-versatile on 400 errors.
    Get a free key at: https://console.groq.com
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    # Try the configured model first; on 400 (unsupported/bad model) use the reliable fallback
    models_to_try = [GROQ_MODEL]
    if GROQ_MODEL != _GROQ_FALLBACK_MODEL:
        models_to_try.append(_GROQ_FALLBACK_MODEL)

    last_error: Exception = RuntimeError("Groq: no models tried")
    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": 0.8,
            "max_tokens": 2048,
        }
        model_failed = False
        for attempt in range(retries):
            attempt_start = time.time()
            try:
                logger.info(f"[Examiney][Generator]     Groq attempt {attempt + 1}: POST to {GROQ_URL} (model: {model_name})...")
                # Slightly higher timeout to reduce premature fallbacks
                with httpx.Client(timeout=45.0) as client:
                    r = client.post(
                        GROQ_URL,
                        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                        json=payload,
                    )
                    if r.status_code == 400:
                        # Model itself is unsupported — skip retries, try next model
                        logger.warning(f"[Examiney][Generator]     Groq model {model_name} returned 400 — trying fallback model...")
                        last_error = RuntimeError(f"Groq 400 for model {model_name}")
                        model_failed = True
                        break
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    result = _extract_json(content)
                    elapsed = time.time() - attempt_start
                    logger.info(f"[Examiney][Generator]     Groq HTTP completed in {elapsed:.2f}s (model: {model_name}, attempt {attempt + 1})")
                    return result
            except Exception as e:
                elapsed = time.time() - attempt_start
                last_error = e
                if attempt < retries - 1:
                    logger.warning(f"[Examiney][Generator]     Groq attempt {attempt + 1} failed in {elapsed:.2f}s: {e}, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    model_failed = True
        if not model_failed:
            break  # shouldn't reach here if return fired, but safety guard

    raise RuntimeError(f"Groq call failed on all models: {last_error}") from last_error


def _call_llm(system: str, user: str, model: str, ollama_url: str) -> dict:
    """Priority: Gemini Flash → Groq → Ollama (local fallback).
    Set GEMINI_API_KEY or GROQ_API_KEY in .env for fast hosted inference.
    """
    call_start = time.time()
    logger.info(f"[Examiney][Generator] → LLM call started (Gemini {GEMINI_MODEL}→key2→Groq→Ollama priority)")
    
    if GEMINI_API_KEY:
        try:
            logger.info(f"[Examiney][Generator]   Trying Gemini Flash (key1)...")
            result = _call_gemini(system, user, api_key=GEMINI_API_KEY)
            elapsed = time.time() - call_start
            logger.info(f"[Examiney][Generator]   ✓ Gemini key1 SUCCESS in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - call_start
            logger.warning(f"[Examiney][Generator]   ✗ Gemini key1 failed in {elapsed:.2f}s ({e}), trying key2...")

    if GEMINI_API_KEY2:
        try:
            logger.info(f"[Examiney][Generator]   Trying Gemini Flash (key2)...")
            result = _call_gemini(system, user, api_key=GEMINI_API_KEY2)
            elapsed = time.time() - call_start
            logger.info(f"[Examiney][Generator]   ✓ Gemini key2 SUCCESS in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - call_start
            logger.warning(f"[Examiney][Generator]   ✗ Gemini key2 failed in {elapsed:.2f}s ({e}), trying Groq...")

    if GROQ_API_KEY:
        try:
            logger.info(f"[Examiney][Generator]   Trying Groq...")
            result = _call_groq(system, user)
            elapsed = time.time() - call_start
            logger.info(f"[Examiney][Generator]   ✓ Groq SUCCESS in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - call_start
            logger.warning(f"[Examiney][Generator]   ✗ Groq failed in {elapsed:.2f}s ({e}), falling back to Ollama")
    
    logger.info(f"[Examiney][Generator]   Trying Ollama (this may be slow)...")
    result = _call_ollama(system, user, model, ollama_url)
    elapsed = time.time() - call_start
    logger.info(f"[Examiney][Generator]   ✓ Ollama SUCCESS in {elapsed:.2f}s")
    return result


_STAGE_ALIASES = {
    "introduction": "intro",
    "opening":      "intro",
    "tech":         "technical",
    "logic":        "logical",
    "logic_reasoning": "logical",
    "reasoning":    "logical",
    "behaviour":    "behavioral",
    "behaviour_based": "behavioral",
    "behavioural":  "behavioral",
    "situational":  "situational",
}


def _normalize_stage(raw_stage: str, fallback: str) -> str:
    """Map model-returned stage strings to valid enum values."""
    s = (raw_stage or "").strip().lower()
    return _STAGE_ALIASES.get(s, s if s in {"intro", "technical", "behavioral", "logical", "situational"} else fallback)


def _coerce_question(raw: dict, stage: str, idx: int) -> Question:
    """Build a Question from raw dict, filling in missing fields with safe defaults."""
    ak_raw = raw.get("answer_key") or {}
    answer_key = AnswerKey(
        critical_keywords=ak_raw.get("critical_keywords") or [],
        ideal_sentiment=ak_raw.get("ideal_sentiment") or "confident",
        rubric=ak_raw.get("rubric") or "1=poor, 5=adequate, 10=excellent",
    )
    raw_stage = _normalize_stage(raw.get("stage") or "", stage)
    return Question(
        id=raw.get("id") or f"q{idx}",
        stage=raw_stage,  # type: ignore[arg-type]
        question=raw.get("question") or raw.get("text") or f"Question {idx}",
        time_window_seconds=int(raw.get("time_window_seconds") or
                                {"intro": 60, "technical": 90, "behavioral": 90, "logical": 60}.get(stage, 75)),
        ideal_answer=raw.get("ideal_answer") or "",
        answer_key=answer_key,
    )


def _process_batch_result(
    data: dict,
    stage: str,
    count: int,
    q_start: int,
    resume_markdown: str,
    job_description: str,
) -> list:
    """Parse one batch result dict into a list of Questions, padding with fallbacks."""
    raws = data.get("questions") or []
    questions = []
    local_idx = q_start
    added = 0
    for raw in raws[:count]:
        if isinstance(raw, dict):
            questions.append(_coerce_question(raw, stage, local_idx))
            local_idx += 1
            added += 1
    for _ in range(count - added):
        questions.append(_default_question(stage, local_idx, resume_markdown, job_description))
        local_idx += 1
    return questions


def generate_questions(
    resume_markdown: str = "",
    job_description: str = "",
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL,
    section_counts=None,  # Optional[Dict[str, int]]
) -> InterviewScript:
    """Generate interview questions — all batches run in parallel for maximum speed.

    If *section_counts* is provided (e.g. {"intro": 2, "technical": 5, "behavioral": 3}),
    it overrides the default batch configuration.

    Speed tiers (fastest → slowest):
      1. Gemini 2.0 Flash  — set GEMINI_API_KEY (~1-3s/batch, free 1500 req/day)
      2. Groq              — set GROQ_API_KEY  (~0.5-2s/batch, free 14400 req/day)
                             set GROQ_MODEL=qwen-qwq-32b for hosted Qwen
      3. Ollama local      — fallback, slow without GPU
    """
    logger.info("[Examiney][Generator] ✓ START question generation")
    logger.info(f"[Examiney][Generator]   Resume: {len(resume_markdown)} chars, JD: {len(job_description)} chars")
    logger.info(f"[Examiney][Generator]   Using model: {model}, Ollama URL: {ollama_url}")

    batches = _build_batches(section_counts) if section_counts else _DEFAULT_BATCHES
    logger.info(f"[Examiney][Generator] ✓ Building {len(batches)} batches: {batches}")

    # Pre-assign q_idx offsets so all batches can be built before any runs
    tasks: list = []  # (batch_index, stage, count, q_start, prompt)
    q_idx = 1
    logger.info("[Examiney][Generator] ✓ Building prompts...")
    for i, (stage, count) in enumerate(batches):
        logger.debug(f"[Examiney][Generator]   Batch {i}: {stage}x{count} starting at q{q_idx}")
        prompt = build_batch_prompt(
            stage=stage,
            count=count,
            id_start=q_idx,
            resume_snippet=resume_markdown,
            job_snippet=job_description,
        )
        tasks.append((i, stage, count, q_idx, prompt))
        q_idx += count  # advance by expected count (fallbacks fill any gaps)
    logger.info(f"[Examiney][Generator] ✓ All prompts built. Total tasks: {len(tasks)}")

    results: dict = {}  # batch_index -> list[Question]

    # Run all batches concurrently — stagger submissions by 0.4s each to avoid
    # Gemini/Groq rate-limit 429s when all batches fire at the exact same moment.
    logger.info(f"[Examiney][Generator] ✓ Starting ThreadPoolExecutor with {min(len(tasks), 5)} workers...")
    start_time = time.time()

    def _staggered_call(batch_idx: int, sys_prompt: str, usr_prompt: str, mdl: str, url: str) -> dict:
        if batch_idx > 0:
            time.sleep(batch_idx * 0.4)  # 0s, 0.4s, 0.8s, 1.2s … per batch
        return _call_llm(sys_prompt, usr_prompt, mdl, url)

    with ThreadPoolExecutor(max_workers=min(len(tasks), 5)) as executor:
        logger.info(f"[Examiney][Generator] ✓ Submitting {len(tasks)} LLM calls (staggered)...")
        future_map = {
            executor.submit(_staggered_call, i, SYSTEM_PROMPT, prompt, model, ollama_url): (i, stage, count, q_start)
            for i, stage, count, q_start, prompt in tasks
        }
        logger.info(f"[Examiney][Generator] ✓ All {len(future_map)} futures submitted. Waiting for results...")
        completed_count = 0
        for future in as_completed(future_map):
            i, stage, count, q_start = future_map[future]
            completed_count += 1
            elapsed = time.time() - start_time
            logger.info(f"[Examiney][Generator] ⏳ [{completed_count}/{len(future_map)}] Batch {i} ({stage}x{count}) completed in {elapsed:.1f}s total")
            try:
                data = future.result()
                results[i] = _process_batch_result(data, stage, count, q_start, resume_markdown, job_description)
                logger.info(f"[Examiney][Generator]   ✓ Batch {i} processed: {len(results[i])} questions")
            except Exception as e:
                logger.error(f"[Examiney][Generator]   ✗ batch {stage}x{count} failed: {e} — using fallbacks")
                results[i] = [
                    _default_question(stage, q_start + j, resume_markdown, job_description)
                    for j in range(count)
                ]

    # Reconstruct in original batch order
    all_questions = []
    for i in range(len(tasks)):
        all_questions.extend(results[i])

    # Ensure distinct question wording (timeouts sometimes return duplicates).
    seen_text: set[str] = set()
    duplicate_counter = 0
    for idx, q in enumerate(all_questions):
        key = (q.question or "").strip().lower()
        if key in seen_text:
            duplicate_counter += 1
            replacement = _default_question(q.stage, idx + 1, resume_markdown, job_description)
            replacement.question = f"{replacement.question} (share a different example #{duplicate_counter})"
            replacement.id = q.id  # preserve ordering/id
            replacement.stage = q.stage
            all_questions[idx] = replacement
            key = replacement.question.strip().lower()
        seen_text.add(key)

    # ── CRITICAL FIX: Enforce exact expected question count ──────────────────
    # Calculate how many questions were expected from batches
    expected_count = sum(count for _, count in batches)
    
    # If we have more questions than expected, trim to exact count
    if len(all_questions) > expected_count:
        logger.info(f"[Examiney][Generator] Trimming from {len(all_questions)} to {expected_count} questions (excess removed)")
        all_questions = all_questions[:expected_count]
    # If we have fewer questions than expected, pad with fallbacks
    elif len(all_questions) < expected_count:
        logger.warning(f"[Examiney][Generator] Padding from {len(all_questions)} to {expected_count} questions (fallbacks added)")
        while len(all_questions) < expected_count:
            idx = len(all_questions) + 1
            # Use the expected stage for this position if possible
            stage = "intro"
            for i, (s, c) in enumerate(batches):
                batch_start = sum(count for _, count in batches[:i])
                batch_end = batch_start + c
                if batch_start <= len(all_questions) < batch_end:
                    stage = s
                    break
            fallback_q = _default_question(stage, idx, resume_markdown, job_description)
            all_questions.append(fallback_q)

    total_time = time.time() - start_time
    logger.info(f"[Examiney][Generator] ✓ DONE! Generated {len(all_questions)} questions in {total_time:.1f}s (expected {expected_count})")
    return InterviewScript(
        job_title=None,
        candidate_name=None,
        questions=all_questions,
    )

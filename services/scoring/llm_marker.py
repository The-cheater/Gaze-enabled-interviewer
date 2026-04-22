"""LLM-based per-dimension response marker and correctness judge.

Two entry points:
  judge_response()  — PRIMARY: determines if answer is correct / partially_correct /
                      can_be_better / incorrect / not_attempted.
  mark_response()   — SECONDARY: scores 5 dimensions + OCEAN signals.

Both share a single _call_llm_judge() path that tries Gemini (primary) then
Ollama (fallback).  A module-level threading semaphore caps concurrent Gemini
requests at 2, preventing 429 rate-limit bursts when 19 questions are scored
in parallel.
"""

import json
import logging
import os
import re
import threading
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_OLLAMA_URL: str  = os.getenv("OLLAMA_URL",    "http://localhost:11434")
_MODEL: str       = os.getenv("OLLAMA_MODEL",  "qwen2.5:0.5b")
_GEMINI_KEY: str  = os.getenv("GEMINI_API_KEY",  "")
_GEMINI_KEY2: str = os.getenv("GEMINI_API_KEY2", "")
_GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
# gemini-1.5-flash and gemini-1.5-flash-8b were deprecated and return 404.
_GEMINI_FALLBACKS = ["gemini-2.0-flash"]
_GEMINI_BASE      = "https://generativelanguage.googleapis.com/v1beta/models"

# Limit concurrent Gemini calls to 2 across all parallel scoring threads.
# Free-tier Gemini Flash allows 15 RPM; with 19 questions × 2 calls each,
# firing all at once causes a burst of 429s.
_GEMINI_SEMAPHORE = threading.Semaphore(2)

# ── Verdict constants ─────────────────────────────────────────────────────────
VERDICT_CORRECT           = "correct"
VERDICT_PARTIALLY_CORRECT = "partially_correct"
VERDICT_CAN_BE_BETTER     = "can_be_better"
VERDICT_INCORRECT         = "incorrect"
VERDICT_NOT_ATTEMPTED     = "not_attempted"

VERDICT_SCORE: Dict[str, float] = {
    VERDICT_CORRECT:           9.5,
    VERDICT_PARTIALLY_CORRECT: 6.5,
    VERDICT_CAN_BE_BETTER:     3.5,
    VERDICT_INCORRECT:         1.0,
    VERDICT_NOT_ATTEMPTED:     0.0,
}

# ── Stage-specific evaluation criteria ───────────────────────────────────────
_STAGE_CRITERIA = {
    "intro": (
        "Evaluate for: clarity of self-introduction, motivation fit for the role, "
        "communication fluency, and whether the candidate makes a confident professional impression. "
        "Do NOT penalise for technical gaps at this stage."
    ),
    "technical": (
        "Evaluate for: factual accuracy against the ideal answer, depth of technical knowledge, "
        "use of correct terminology, and ability to explain concepts clearly. "
        "Be strict — vague or generic answers that lack specifics must be marked 'can_be_better' or lower."
    ),
    "logical": (
        "Evaluate for: structured step-by-step reasoning, correct problem decomposition, "
        "and whether the candidate reaches a sound conclusion. "
        "Penalise guessing or jumping to conclusions without justification."
    ),
    "behavioral": (
        "Evaluate for: use of the STAR method (Situation, Task, Action, Result). "
        "Answers that lack a concrete example or real past experience must be 'can_be_better' or lower. "
        "Generic responses like 'I would...' instead of 'I did...' are a red flag."
    ),
    "situational": (
        "Evaluate for: decision-making framework, consideration of trade-offs, "
        "and practicality of the proposed approach. "
        "Penalise answers that are overly theoretical or fail to consider real constraints."
    ),
}

_DEFAULT_CRITERIA = "Evaluate for factual accuracy, depth, and relevance to the question."


# ── Combined schema (judge + mark in one call) ────────────────────────────────

_COMBINED_SCHEMA = (
    '{"verdict":"correct|partially_correct|can_be_better|incorrect|not_attempted",'
    '"verdict_reason":"one sentence",'
    '"key_gaps":["gap1"],'
    '"strengths":["strength1"],'
    '"technical":7,"communication":8,"behavioral":6,"engagement":7,"authenticity":8,'
    '"ocean_signals":{"openness":0.7,"conscientiousness":0.6,"extraversion":0.5,'
    '"agreeableness":0.8,"neuroticism":0.3}}'
)

_COMBINED_SYSTEM = (
    "You are a strict and precise interview assessor. "
    "Evaluate the candidate's answer and reply with ONLY valid JSON — no prose, no markdown."
)


def _combined_prompt(question_text: str, ideal_answer: str, transcript: str, stage: str) -> str:
    stage_criteria = _STAGE_CRITERIA.get(stage, _DEFAULT_CRITERIA)
    verdict_guide = (
        "Verdict definitions (apply strictly):\n"
        "  correct           — addresses ALL key points accurately and specifically\n"
        "  partially_correct — right direction but misses at least one important point\n"
        "  can_be_better     — correct in spirit but too vague, shallow, or lacking evidence\n"
        "  incorrect         — factually wrong, off-topic, or contradicts the ideal answer\n"
        "  not_attempted     — candidate said nothing relevant\n"
    )
    return (
        f"Question type: {stage}\n"
        f"Stage evaluation criteria: {stage_criteria}\n\n"
        f"Question: {question_text}\n\n"
        f"Ideal answer: {ideal_answer[:500]}\n\n"
        f"Candidate response: {transcript}\n\n"
        f"{verdict_guide}\n"
        "Also score these 5 dimensions (0-10 each, be strict):\n"
        "  technical — factual accuracy and depth\n"
        "  communication — clarity, structure, vocabulary\n"
        "  behavioral — use of examples / STAR method\n"
        "  engagement — enthusiasm and energy\n"
        "  authenticity — genuine, natural delivery\n\n"
        "Extract OCEAN personality signals (0.0-1.0 each):\n"
        "  openness, conscientiousness, extraversion, agreeableness, neuroticism\n\n"
        f"Reply with ONLY this JSON (no other text):\n{_COMBINED_SCHEMA}"
    )


def _split_combined(raw: dict) -> tuple:
    """Split combined LLM response into (judgment_dict, marks_dict)."""
    verdict = raw.get("verdict", VERDICT_INCORRECT)
    if verdict not in VERDICT_SCORE:
        verdict = VERDICT_INCORRECT

    verdict_reason = _clean_markdown(str(raw.get("verdict_reason", "")).strip()[:400])
    key_gaps  = [_clean_markdown(str(g)) for g in raw.get("key_gaps",  [])[:5]]
    strengths = [_clean_markdown(str(s)) for s in raw.get("strengths", [])[:5]]

    judgment = {
        "verdict":        verdict,
        "verdict_reason": verdict_reason,
        "score":          VERDICT_SCORE[verdict],
        "key_gaps":       key_gaps,
        "strengths":      strengths,
    }

    marks = _normalise(raw)
    return judgment, marks


# ── Public API ────────────────────────────────────────────────────────────────

def judge_response(
    question_text: str,
    ideal_answer: str,
    transcript: str,
    stage: str,
) -> Dict:
    """Determine whether the candidate's answer is correct.

    Returns:
      {verdict, verdict_reason, score, key_gaps, strengths}

    Side-effect: stores the dimension marks in _combined_cache so that a
    subsequent call to mark_response() with the same question+transcript
    reuses the same LLM result without a second network call.
    """
    if not transcript or not transcript.strip():
        return {
            "verdict":        VERDICT_NOT_ATTEMPTED,
            "verdict_reason": "No response was given.",
            "score":          0.0,
            "key_gaps":       ["Complete response required"],
            "strengths":      [],
        }

    prompt = _combined_prompt(question_text, ideal_answer, transcript, stage)
    raw = _call_llm_judge(_COMBINED_SYSTEM, prompt)
    if raw is None:
        return _heuristic_verdict(transcript, ideal_answer)

    judgment, marks = _split_combined(raw)
    # Cache marks so mark_response() doesn't need a second LLM call
    _combined_cache[_cache_key(question_text, transcript)] = marks
    return judgment


def mark_response(
    question_text: str,
    ideal_answer: str,
    transcript: str,
    stage: str,
    model: str = _MODEL,
    ollama_url: str = _OLLAMA_URL,
) -> Dict:
    """Score a transcript on 5 dimensions and extract raw OCEAN signals.

    NOTE: When called after judge_response() in the same pipeline step,
    the LLM result is already cached via _last_combined_cache so no
    second network call is made.

    Returns:
        {technical, communication, behavioral, engagement, authenticity, ocean_signals}
    """
    # Check thread-local cache set by judge_response
    cached = _combined_cache.get(_cache_key(question_text, transcript))
    if cached is not None:
        return cached

    prompt = _combined_prompt(question_text, ideal_answer, transcript, stage)
    raw = _call_llm_judge(_COMBINED_SYSTEM, prompt, ollama_url=ollama_url, ollama_model=model)
    if raw is None:
        return _heuristic_dimension_scores(transcript, ideal_answer, stage)

    _, marks = _split_combined(raw)
    return marks


# ── Combined result cache (avoids double LLM call per question) ───────────────
# Keyed by (question_text[:100], transcript[:100]) — cheap in-process dict.
# Cleared between sessions by the process lifecycle.

_combined_cache: Dict[str, dict] = {}


def _cache_key(question_text: str, transcript: str) -> str:
    return f"{question_text[:80]}||{transcript[:80]}"


# ── Internal LLM call ─────────────────────────────────────────────────────────

def _call_llm_judge(
    system_msg: str,
    prompt: str,
    max_tokens: int = 600,
    temperature: float = 0.1,
    ollama_url: str = _OLLAMA_URL,
    ollama_model: str = _MODEL,
) -> Optional[dict]:
    """Try Gemini (semaphore-guarded) then Ollama. Returns parsed dict or None."""
    result = _call_gemini_llm(system_msg, prompt, max_tokens, temperature)
    if result is not None:
        return result

    # Ollama /api/chat
    try:
        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        with httpx.Client(timeout=45.0) as client:
            r = client.post(f"{ollama_url}/api/chat", json=payload)
            if r.status_code == 200:
                return json.loads(r.json()["message"]["content"])
    except Exception as e:
        logger.debug(f"[Examiney][LLMJudge] Ollama chat failed: {e}")

    # Ollama /api/generate
    try:
        payload = {
            "model":  ollama_model,
            "prompt": f"{system_msg}\n\n{prompt}",
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        with httpx.Client(timeout=45.0) as client:
            r = client.post(f"{ollama_url}/api/generate", json=payload)
            r.raise_for_status()
            return json.loads(r.json()["response"])
    except Exception as e:
        logger.warning(f"[Examiney][LLMJudge] All LLM backends failed: {e}")

    return None


def _call_gemini_llm(
    system_msg: str, prompt: str, max_tokens: int = 600, temperature: float = 0.1
) -> Optional[dict]:
    """Call Gemini with semaphore, key1→key2, model fallback. Returns parsed dict or None."""
    keys = [k for k in (_GEMINI_KEY, _GEMINI_KEY2) if k]
    if not keys:
        return None

    models = [_GEMINI_MODEL] + [m for m in _GEMINI_FALLBACKS if m != _GEMINI_MODEL]
    payload = {
        "system_instruction": {"parts": [{"text": system_msg}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }

    with _GEMINI_SEMAPHORE:   # at most 2 threads call Gemini at once
        for key in keys:
            for model in models:
                url = f"{_GEMINI_BASE}/{model}:generateContent"
                try:
                    with httpx.Client(timeout=25.0) as client:
                        r = client.post(url, params={"key": key}, json=payload)
                        if r.status_code == 429:
                            logger.debug(f"[Examiney][LLMJudge] Gemini {model} 429 — rate limited")
                            continue
                        if r.status_code in (400, 404):
                            logger.debug(f"[Examiney][LLMJudge] Gemini {model} {r.status_code} — skipping")
                            continue
                        r.raise_for_status()
                        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                        return _extract_json_robust(text)
                except Exception as e:
                    logger.debug(f"[Examiney][LLMJudge] Gemini {model} failed: {e}")

    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_markdown(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"__(.*?)__",     r"\1", text)
    text = re.sub(r"_(.*?)_",       r"\1", text)
    text = re.sub(r"`(.*?)`",       r"\1", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_json_robust(text: str) -> dict:
    text = text.strip()
    text_clean = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text_clean = re.sub(r"\s*```\s*$", "", text_clean).strip()
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass
    depth = 0
    start = -1
    for i, ch in enumerate(text_clean):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(text_clean[start:i + 1])
                except json.JSONDecodeError:
                    start = -1
    raise ValueError(f"No valid JSON found in: {text[:200]}")


def _clamp(val, lo: float, hi: float) -> float:
    try:
        return float(min(hi, max(lo, float(val))))
    except (TypeError, ValueError):
        return (lo + hi) / 2.0


def _normalise(data: dict) -> dict:
    sig = data.get("ocean_signals", {})
    return {
        "technical":     _clamp(data.get("technical", 0),     0, 10),
        "communication": _clamp(data.get("communication", 0), 0, 10),
        "behavioral":    _clamp(data.get("behavioral", 0),    0, 10),
        "engagement":    _clamp(data.get("engagement", 0),    0, 10),
        "authenticity":  _clamp(data.get("authenticity", 0),  0, 10),
        "ocean_signals": {
            "openness":          _clamp(sig.get("openness", 0.0),          0, 1),
            "conscientiousness": _clamp(sig.get("conscientiousness", 0.0), 0, 1),
            "extraversion":      _clamp(sig.get("extraversion", 0.0),      0, 1),
            "agreeableness":     _clamp(sig.get("agreeableness", 0.0),     0, 1),
            "neuroticism":       _clamp(sig.get("neuroticism", 0.0),       0, 1),
        },
    }


def _heuristic_verdict(transcript: str, ideal_answer: str) -> Dict:
    """Fallback when LLM is unavailable — keyword overlap as rough proxy."""
    t_words = set(transcript.lower().split())
    i_words = set(w for w in ideal_answer.lower().split() if len(w) > 4)
    word_count = len(transcript.split())

    if not i_words:
        return {
            "verdict":        VERDICT_NOT_ATTEMPTED,
            "verdict_reason": "No ideal answer provided for comparison.",
            "score":          VERDICT_SCORE[VERDICT_NOT_ATTEMPTED],
            "key_gaps":       [],
            "strengths":      [],
        }

    overlap = len(t_words & i_words) / len(i_words)
    has_depth = word_count >= 50

    if overlap >= 0.60 and has_depth:
        verdict = VERDICT_CORRECT
        reason = f"Strong overlap ({overlap:.0%}) with sufficient depth ({word_count} words)."
    elif overlap >= 0.45 and has_depth:
        verdict = VERDICT_PARTIALLY_CORRECT
        reason = f"Good overlap ({overlap:.0%}) but could be more comprehensive."
    elif overlap >= 0.20 and word_count >= 20:
        verdict = VERDICT_CAN_BE_BETTER
        reason = f"Basic understanding present ({overlap:.0%}) but lacks depth ({word_count} words)."
    elif overlap >= 0.10:
        verdict = VERDICT_INCORRECT
        reason = f"Limited overlap ({overlap:.0%}) — response misses key points."
    elif word_count >= 5:
        verdict = VERDICT_INCORRECT
        reason = "Response off-topic or not addressing the question sufficiently."
    else:
        verdict = VERDICT_NOT_ATTEMPTED
        reason = "Response too short or empty."

    return {
        "verdict":        verdict,
        "verdict_reason": reason + " (LLM judge unavailable — heuristic fallback.)",
        "score":          VERDICT_SCORE[verdict],
        "key_gaps":       [],
        "strengths":      [],
    }


def _heuristic_dimension_scores(transcript: str, ideal_answer: str, stage: str = "") -> dict:
    """Fallback dimension scoring when LLM is unavailable."""
    word_count = len([w for w in transcript.split() if len(w) > 1])
    t_words = {w.lower() for w in transcript.split() if len(w) > 2}
    i_words = {w.lower() for w in ideal_answer.split() if len(w) > 2}

    overlap = len(t_words & i_words) / len(i_words) if i_words else 0.5

    if stage == "intro":
        min_words, target_words = 20, 60
    elif stage == "behavioral":
        min_words, target_words = 60, 150
    else:
        min_words, target_words = 40, 100

    if word_count < min_words:
        depth = 0.0
    elif word_count >= target_words:
        depth = 1.0
    else:
        depth = (word_count - min_words) / (target_words - min_words)

    base_score = max(0.0, min(10.0, (overlap * 0.4 + depth * 0.6) * 10))

    return {
        "technical":     round(max(0.0, min(10.0, base_score * (0.9 + overlap * 0.1))), 1),
        "communication": round(max(0.0, min(10.0, base_score * (0.95 + depth * 0.05))), 1),
        "behavioral":    round(max(0.0, min(10.0, base_score if stage == "behavioral" else base_score * 0.8)), 1),
        "engagement":    round(max(0.0, min(10.0, 4.0 + depth * 3.0)), 1),
        "authenticity":  round(max(0.0, min(10.0, base_score * 0.9)), 1),
        "ocean_signals": {
            "openness":          round(0.5 + overlap * 0.3, 1),
            "conscientiousness": round(0.5 + depth * 0.35, 1),
            "extraversion":      round(0.5 + depth * 0.3, 1),
            "agreeableness":     round(min(0.9, 0.6 + word_count / 300.0), 1),
            "neuroticism":       round(max(0.1, 0.4 - depth * 0.2), 1),
        },
    }

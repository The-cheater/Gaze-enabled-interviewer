import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Union

from .models import ParsedResume, Experience, Project

logger = logging.getLogger(__name__)

# Module-level cache for parsed resumes (hash -> ParsedResume)
_PARSE_CACHE: Dict[str, ParsedResume] = {}


@lru_cache(maxsize=1)
def _get_converter():
    """Load DocumentConverter once — it initialises heavy ML models on first call."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError("Install docling: pip install docling")
    return DocumentConverter()


def parse_pdf(file_path: Union[str, Path]) -> ParsedResume:
    """Parse a PDF resume using IBM Docling, with hash-based caching."""
    try:
        # Compute hash of file content for caching
        with open(file_path, 'rb') as f:
            file_content = f.read()
            file_hash = hashlib.md5(file_content).hexdigest()
        
        # Check cache first
        if file_hash in _PARSE_CACHE:
            logger.info(f"[Parser] Using cached parse for {file_path} (hash: {file_hash[:8]})")
            return _PARSE_CACHE[file_hash]
    except Exception as e:
        logger.warning(f"[Parser] Cache lookup failed (non-critical): {e}")
        file_hash = None
    
    # Parse using DocumentConverter
    converter = _get_converter()
    result = converter.convert(str(file_path))
    markdown = result.document.export_to_markdown()
    parsed = _extract_from_markdown(markdown)
    
    # Cache the result
    if file_hash:
        try:
            _PARSE_CACHE[file_hash] = parsed
            logger.info(f"[Parser] Cached parse for {file_path} (hash: {file_hash[:8]})")
        except Exception as e:
            logger.warning(f"[Parser] Cache storage failed (non-critical): {e}")
    
    return parsed


def parse_text(text: str) -> ParsedResume:
    """Parse plain text/markdown resume."""
    return _extract_from_markdown(text)


def _extract_from_markdown(markdown: str) -> ParsedResume:
    resume = ParsedResume(raw_markdown=markdown)

    # Email — supports multi-part TLDs (.co.uk, .com.au, etc.)
    m = re.search(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b", markdown)
    if m:
        resume.email = m.group()

    # Phone — flexible international format (E.164, UK, IN, EU, US).
    # Anchored to word/line boundaries to avoid matching IP addresses (192.168.x.x)
    # and dates (2024-01-15).  Requires 7-15 digits total (ITU-T E.164 limits).
    # Negative lookahead on 4-digit year followed by date separator to skip dates.
    _PHONE_RE = re.compile(
        r"(?<!\d)"                            # not preceded by digit (avoids mid-number)
        r"(?!\d{4}[-/]\d{2}[-/]\d{2})"       # not a date like 2024-01-15
        r"(?:\+\d{1,3}[\s\-.]?)?"            # optional country code (+44, +1, +91…)
        r"(?:\(?\d{2,4}\)?[\s\-.]?)?"        # optional area code  (0207, (415)…)
        r"\d{3,5}[\s\-.]?\d{3,5}"            # main body           (555 1234, 98765…)
        r"(?:[\s\-.]?\d{2,4})?"              # optional extension
        r"(?!\d)",                            # not followed by digit (avoids IP octets)
    )
    for pm in _PHONE_RE.finditer(markdown):
        digits = re.sub(r"\D", "", pm.group())
        if 7 <= len(digits) <= 15:
            resume.phone = pm.group().strip()
            break

    # Section-header keywords to exclude from name detection
    _SECTION_HEADERS = {
        "education", "experience", "skills", "projects", "summary", "objective",
        "work", "history", "certifications", "publications", "awards", "languages",
        "references", "contact", "profile", "about", "technologies", "interests",
    }

    # Name: first non-empty line that looks like a person's name (not a section header)
    for line in markdown.splitlines():
        clean = line.strip().lstrip("#").strip()
        words = clean.split()
        if (
            clean
            and not re.search(r"[@|:\d]", clean)        # no email/phone/number chars
            and 1 < len(words) <= 6                      # 2-6 words (skip single-word headers)
            and clean.lower() not in _SECTION_HEADERS    # not a known section header
            and not any(w.lower() in _SECTION_HEADERS for w in words)
        ):
            resume.name = clean
            break

    # Skills section
    skills_match = re.search(
        r"(?:skills?|technologies?|tech\s*stack)[:\s]*\n+(.*?)(?=\n## |\Z)",
        markdown,
        re.IGNORECASE | re.DOTALL,
    )
    if skills_match:
        raw = skills_match.group(1)
        items = re.split(r"[,\|•\n\-\*]+", raw)
        resume.skills = [s.strip() for s in items if s.strip() and len(s.strip()) > 1]

    # Use \n## (exactly 2 hashes) as section boundary to avoid matching ### sub-headings
    _SECTION_END = r"(?=\n## |\Z)"

    # Education section (simple line extraction)
    edu_match = re.search(
        r"(?:education)[:\s]*\n+(.*?)" + _SECTION_END,
        markdown,
        re.IGNORECASE | re.DOTALL,
    )
    if edu_match:
        lines = [l.strip() for l in edu_match.group(1).splitlines() if l.strip()]
        resume.education = lines[:10]

    # Experience section
    exp_match = re.search(
        r"(?:experience|work\s*history)[:\s]*\n+(.*?)" + _SECTION_END,
        markdown,
        re.IGNORECASE | re.DOTALL,
    )
    if exp_match:
        resume.experience = _parse_experience_block(exp_match.group(1))

    # Projects section
    proj_match = re.search(
        r"(?:projects?)[:\s]*\n+(.*?)" + _SECTION_END,
        markdown,
        re.IGNORECASE | re.DOTALL,
    )
    if proj_match:
        resume.projects = _parse_projects_block(proj_match.group(1))

    return resume


def _parse_experience_block(text: str) -> list[Experience]:
    experiences = []
    # Prepend \n so the first ### is also caught by the split pattern
    blocks = re.split(r"\n#{2,3}\s+|\n\*\*", "\n" + text)
    for block in blocks:
        if not block.strip():
            continue
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        title_line = lines[0].lstrip("#").strip().rstrip("*")
        company, title, duration = "", title_line, ""

        # "Role — Company (Date)" pattern
        m = re.match(r"(.+?)\s*[—\-–]+\s*(.+?)(?:\s*\((.+?)\))?$", title_line)
        if m:
            title = m.group(1).strip()
            company = m.group(2).strip()
            duration = m.group(3) or ""

        desc = " ".join(lines[1:])
        experiences.append(Experience(title=title, company=company, duration=duration, description=desc))

    return experiences[:10]


def _parse_projects_block(text: str) -> list[Project]:
    projects = []
    blocks = re.split(r"\n#{2,3}\s+|\n\*\*", "\n" + text)
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        name = lines[0].lstrip("#").strip().rstrip("*")
        desc = ""
        techs = []

        # Look for "Technologies:" line
        for line in lines[1:]:
            tech_m = re.match(r"[Tt]echnolog(?:y|ies)[:\s]+(.+)", line)
            if tech_m:
                techs = [t.strip() for t in re.split(r"[,\|]", tech_m.group(1))]
            else:
                desc = line if not desc else desc

        projects.append(Project(name=name, description=desc, technologies=techs))

    return projects[:10]

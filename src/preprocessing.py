"""Transcript cleaning: participant speech, markers, QC checks."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, is_process2
from src.data_loading import load_process2_ctd_path

logger = logging.getLogger(__name__)

# Common interviewer / system prefixes in clinical transcripts
_INTERVIEWER_PREFIXES = (
    r"^inv\s*:\s*",
    r"^int\s*:\s*",
    r"^examiner\s*:\s*",
    r"^researcher\s*:\s*",
    r"^speaker\s*0\s*:\s*",
)

# Time / duration markers like "(3 seconds)" — thesis-agnostic noise for NLP stats
_TIME_PAREN = re.compile(r"\(\s*\d+(?:\.\d+)?\s*(?:second|seconds|sec|s)\s*\)", re.I)
# Disfluency markers from CHAT/CLAN style (optional stripping of bare tags)
_TAG_NOISE = re.compile(r"\[[^\]]*\]|\+\.\.\.|\+\/\/|\+\s*\"|<[^>]+>")

_FILLERS = frozenset({"um", "uh", "er", "well", "like"})


def extract_participant_lines(raw: str, participant_prefixes: tuple[str, ...] = ("Pat:", "PAT:", "Participant:", "PAR:")) -> str:
    """
    Keep lines that look like participant speech; drop interviewer-only lines.

    For block transcripts without line prefixes, returns cleaned full text.
    """
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if any(re.match(p, low) for p in _INTERVIEWER_PREFIXES):
            continue
        # If transcript uses explicit participant tags, keep only those lines
        if any(stripped.startswith(p) for p in participant_prefixes):
            for pref in participant_prefixes:
                if stripped.startswith(pref):
                    stripped = stripped[len(pref) :].strip()
                    break
            lines.append(stripped)
        elif not re.match(r"^[A-Za-z]{2,4}:\s", stripped):
            # No short "XX:" speaker tag — likely continuous participant text
            lines.append(stripped)
        # else: drop lines like "Inv: ..." or unknown tagged speakers
    if not lines:
        # Fallback: entire raw after removing obvious interviewer headers
        return _strip_interviewer_blocks(raw)
    return "\n".join(lines)


def _strip_interviewer_blocks(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        low = line.strip().lower()
        if any(re.match(p, low) for p in _INTERVIEWER_PREFIXES):
            continue
        out_lines.append(line.strip())
    return "\n".join(out_lines).strip()


def clean_transcript(text: str) -> str:
    """Normalize whitespace, casing, and remove timing/noise markers; keep fillers."""
    text = _TAG_NOISE.sub(" ", text)
    text = _TIME_PAREN.sub(" ", text)
    text = text.replace("’", "'")
    # Remove parenthetical stage directions that are purely nonverbal cues
    text = re.sub(
        r"\(\s*(?:tuts?|laughs?|sighs?|coughs?|clears throat)\s*\)", " ", text, flags=re.I
    )
    text = re.sub(r"[^\w\s'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize_words(cleaned: str) -> list[str]:
    """Simple word tokenizer after clean_transcript (lowercase tokens)."""
    if not cleaned:
        return []
    return [w for w in cleaned.split() if w]


def count_fillers(tokens: list[str]) -> int:
    return sum(1 for t in tokens if t in _FILLERS)


def load_ctd_process2(participant_id: str, data_dir: Path | None = None) -> str | None:
    """Load Cookie Theft Description (``*__CTD.txt``) for PROCESS-2 (participant-only text)."""
    path = load_process2_ctd_path(participant_id, data_dir)
    if path is None:
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    return raw if raw.strip() else None


def load_ctd_transcript(participant_id: str, data_dir: Path | None = None) -> str | None:
    """Dataset-aware CTD loader (PROCESS-2 folder layout or legacy Archive)."""
    if is_process2():
        return load_ctd_process2(participant_id, data_dir)
    return load_ctd_from_archive(participant_id, data_dir)


def load_ctd_from_archive(participant_id: str, data_dir: Path | None = None) -> str | None:
    """
    Load Cookie Theft description transcript for Process-rec-XXX from Archive layout.

    Returns None if file missing or empty.
    """
    root = data_dir or DATA_DIR
    archive = root / "Archive" / participant_id
    if not archive.is_dir():
        return None
    matches = list(archive.glob("*__CTD.txt")) + list(archive.glob("*CTD*.txt"))
    if not matches:
        return None
    path = sorted(matches, key=lambda p: len(p.name))[0]
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return None
    return raw


def _df_train_transcript_map(data_dir: Path) -> dict[str, str]:
    """Optional aggregated CTD text from df_train.csv (used when Archive CTD is empty)."""
    p = data_dir / "df_train.csv"
    if not p.is_file():
        return {}
    df = pd.read_csv(p)
    if "Record-ID" not in df.columns or "Transcript_CTD" not in df.columns:
        return {}
    return {
        str(r["Record-ID"]): str(r["Transcript_CTD"])
        for _, r in df.iterrows()
        if pd.notna(r.get("Transcript_CTD"))
    }


def build_cleaned_corpus(
    participant_ids: list[str],
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build one row per participant: participant_id, cleaned_transcript, word_count, qc_flags.
    """
    rows = []
    root = data_dir or DATA_DIR
    train_map = _df_train_transcript_map(root)
    use_process2 = is_process2()
    for pid in participant_ids:
        raw = load_ctd_transcript(pid, root)
        if (raw is None or not str(raw).strip()) and pid in train_map:
            raw = train_map[pid]
            logger.info("Using df_train.csv CTD fallback for %s", pid)
        if raw is None:
            loc = root / pid if use_process2 else root / "Archive"
            logger.warning("Missing CTD transcript for %s under %s", pid, loc)
            extracted = ""
        elif use_process2:
            # PROCESS-2: participant-only text; defensive clean only
            extracted = str(raw)
        else:
            extracted = extract_participant_lines(str(raw))
        cleaned = clean_transcript(extracted)
        tokens = tokenize_words(cleaned)
        wc = len(tokens)
        rows.append(
            {
                "participant_id": pid,
                "cleaned_transcript": cleaned,
                "word_count": wc,
                "qc_empty": wc == 0,
                "qc_short": wc < 20,
            }
        )
    return pd.DataFrame(rows)


def run_quality_checks(df: pd.DataFrame) -> None:
    """Raise on hard failures; warn on soft QC flags."""
    if df["participant_id"].duplicated().any():
        raise ValueError("Duplicated participant_id in cleaned corpus.")
    if df["qc_empty"].any():
        bad = df.loc[df["qc_empty"], "participant_id"].tolist()
        raise ValueError(f"Empty transcripts after cleaning for: {bad[:10]}...")
    short = df.loc[df["qc_short"], "participant_id"].tolist()
    if short:
        logger.warning(
            "Short transcripts (<20 words) — inspect manually: %s", short[:15]
        )

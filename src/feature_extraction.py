"""Linguistic feature extraction: TTR, fillers, clausal length, POS ratios.

The resubmission models the 13 features listed in ``config.yaml``. The
``build_feature_matrix`` function at the bottom is dataset-agnostic: give it any
table with raw transcript text and it returns exactly the configured feature
columns. ``semantic_coherence`` is intentionally not computed (it was constant
0.0 in the thesis environment and is excluded from the feature set), so no SBERT
dependency is exercised on the active path.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, SPACY_MODEL
from src.preprocessing import clean_transcript

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"[.!?]+")


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    try:
        return spacy.load(SPACY_MODEL)
    except OSError as e:
        raise OSError(
            f"spaCy model '{SPACY_MODEL}' not installed. Run:\n"
            f"  python -m spacy download {SPACY_MODEL}\n"
        ) from e


def type_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def filler_features(tokens: list[str]) -> tuple[int, float]:
    fillers = {"um", "uh", "er", "well", "like"}
    c = sum(1 for t in tokens if t in fillers)
    ratio = c / len(tokens) if tokens else 0.0
    return c, ratio


def mean_clause_length_spacy(text: str) -> float:
    """
    Mean estimated clause length (tokens) using spaCy POS/segmentation.

    Per sentence, approximate finite predicates with POS tags associated with
    tensed verbs/modals, divide sentence length by max(1, n_predicates), then
    average across sentences. This is a pragmatic proxy for clausal complexity.
    """
    if not text.strip():
        return 0.0
    nlp = _nlp()
    doc = nlp(text)
    lengths: list[float] = []
    for sent in doc.sents:
        toks = [t for t in sent if not t.is_space and not t.is_punct]
        if not toks:
            continue
        finite = [t for t in toks if t.tag_ in ("VBD", "VBZ", "VBP", "MD")]
        n_clause = max(1, len(finite))
        lengths.append(len(toks) / n_clause)
    return float(np.mean(lengths)) if lengths else 0.0


def content_density_ratio(text: str) -> float:
    """Content words (NOUN, VERB, ADJ, ADV) / total non-punctuation tokens."""
    if not text.strip():
        return 0.0
    nlp = _nlp()
    doc = nlp(text)
    content = 0
    total = 0
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        total += 1
        if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV"):
            content += 1
    return float(content / total) if total else 0.0


def pos_ratios(text: str) -> dict[str, float]:
    """Noun/verb/adj/adv/pronoun/determiner ratios over content-like tokens."""
    if not text.strip():
        return {
            "noun_ratio": 0.0,
            "verb_ratio": 0.0,
            "adjective_ratio": 0.0,
            "adverb_ratio": 0.0,
            "pronoun_ratio": 0.0,
            "determiner_ratio": 0.0,
        }
    nlp = _nlp()
    doc = nlp(text)
    counts = {k: 0 for k in ("NOUN", "VERB", "ADJ", "ADV", "PRON", "DET")}
    total = 0
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        total += 1
        if t.pos_ in counts:
            counts[t.pos_] += 1
    if total == 0:
        return {f"{k.lower()}_ratio": 0.0 for k in counts}
    return {
        "noun_ratio": counts["NOUN"] / total,
        "verb_ratio": counts["VERB"] / total,
        "adjective_ratio": counts["ADJ"] / total,
        "adverb_ratio": counts["ADV"] / total,
        "pronoun_ratio": counts["PRON"] / total,
        "determiner_ratio": counts["DET"] / total,
    }


# -----------------------------------------------------------------------------
# Dataset-agnostic feature matrix (resubmission pipeline)
# -----------------------------------------------------------------------------
def _word_count(cleaned: str) -> int:
    return len([w for w in cleaned.split() if w])


def _sentence_count(raw: str) -> int:
    t = str(raw).strip()
    if not t:
        return 0
    return len([p for p in _SENTENCE_SPLIT.split(t) if p.strip()])


def compute_all_features(raw_text: str) -> dict[str, float]:
    """Compute every supported linguistic feature from a single raw transcript.

    Returns a superset dict; callers select the configured FEATURE_COLUMNS.
    ``semantic_coherence`` is deliberately excluded (constant 0.0, dropped).
    """
    cleaned = clean_transcript(str(raw_text))
    tokens = [w for w in cleaned.split() if w]
    fc, fr = filler_features(tokens)
    pos = pos_ratios(cleaned)
    return {
        "word_count": float(_word_count(cleaned)),
        "sentence_count": float(_sentence_count(raw_text)),
        "type_token_ratio": type_token_ratio(tokens),
        "filler_count": float(fc),
        "filler_ratio": fr,
        "mean_clause_length": mean_clause_length_spacy(cleaned),
        "content_density": content_density_ratio(cleaned),
        **pos,
    }


def build_feature_matrix(
    table: pd.DataFrame,
    *,
    text_column: str,
    id_column: str,
    label_column: str,
    split_column: str,
    feature_names: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Build the modelling feature matrix from any table of raw transcripts.

    ``table`` must contain the id/label/split columns plus a raw transcript text
    column. Returns a DataFrame with id/label/split + one column per configured
    feature, in feature order. No imputation is performed; missing features (if a
    transcript cannot be parsed) surface as NaN for the audit to catch.
    """
    rows: list[dict[str, float]] = []
    for _, r in table.iterrows():
        feats = compute_all_features(r[text_column])
        row: dict[str, float] = {
            id_column: r[id_column],
            label_column: r[label_column],
            split_column: str(r[split_column]).strip().lower(),
        }
        for name in feature_names:
            row[name] = float(feats.get(name, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)

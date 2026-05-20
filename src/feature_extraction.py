"""Linguistic feature extraction: TTR, fillers, clausal length, SBERT coherence, POS ratios."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from src.config import FEATURE_COLUMNS, SBERT_MODEL_NAME, SPACY_MODEL

logger = logging.getLogger(__name__)


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


@lru_cache(maxsize=1)
def _sbert():
    """
    Load SBERT model for semantic coherence.

    On some Windows environments, importing `sentence_transformers` can transitively
    import audio backends (e.g. torchcodec) that require FFmpeg DLLs. Since this
    thesis pipeline is transcript-first, we fall back gracefully when SBERT cannot
    be loaded, and treat semantic coherence as 0.0 in that case.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SBERT_MODEL_NAME)


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


def semantic_coherence_sbert(text: str) -> float:
    """Mean cosine similarity between consecutive sentence embeddings."""
    if not text.strip():
        return 0.0
    nlp = _nlp()
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 3]
    if len(sents) < 2:
        return float("nan")  # undefined; caller may coerce to 0.0
    try:
        model = _sbert()
    except Exception as exc:
        logger.warning(
            "SBERT coherence unavailable (%s: %s). Setting semantic_coherence=0.0. "
            "Fix by installing a compatible sentence-transformers / torch stack, "
            "or by ensuring FFmpeg DLLs are available on Windows.",
            type(exc).__name__,
            exc,
        )
        return 0.0
    emb = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    sims = []
    for i in range(len(emb) - 1):
        a, b = emb[i], emb[i + 1]
        if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
            continue
        sims.append(1.0 - cosine(a, b))
    return float(np.mean(sims)) if sims else float("nan")


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


def extract_row_features(cleaned_transcript: str) -> dict[str, float]:
    tokens = [w for w in cleaned_transcript.split() if w]
    ttr = type_token_ratio(tokens)
    fc, fr = filler_features(tokens)
    mcl = mean_clause_length_spacy(cleaned_transcript)
    coh = semantic_coherence_sbert(cleaned_transcript)
    if coh != coh:  # NaN
        coh = 0.0
    pr = pos_ratios(cleaned_transcript)
    cd = content_density_ratio(cleaned_transcript)
    return {
        "type_token_ratio": ttr,
        "filler_count": float(fc),
        "filler_ratio": fr,
        "mean_clause_length": mcl,
        "semantic_coherence": coh,
        "content_density": cd,
        **pr,
    }


def extract_features_table(corpus: pd.DataFrame) -> pd.DataFrame:
    """Corpus must include participant_id, cleaned_transcript, word_count."""
    feats = []
    for _, row in corpus.iterrows():
        d = extract_row_features(str(row["cleaned_transcript"]))
        feats.append({"participant_id": row["participant_id"], **d})
    feat_df = pd.DataFrame(feats)
    out = corpus[["participant_id", "cleaned_transcript", "word_count"]].merge(
        feat_df, on="participant_id", how="left"
    )
    # Ensure column order
    for c in FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out

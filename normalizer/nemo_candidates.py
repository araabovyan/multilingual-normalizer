"""
NeMo WFST-based candidate generation backend.

Uses NeMo's non-deterministic WFST normalizer to generate all plausible
spoken forms for a NumberSpan token.  The normalizer is lazily loaded and
cached per language so grammar compilation only happens once.

Supported languages (NeMo TN):
    en, es, fr, de, ar, ru, sv, vi, pt, zh, hu, it, hy, mr, hi, ja

For any language outside this set, ``generate_candidates_nemo`` returns an
empty list and the caller is expected to fall back to the num2words backend.

Installation requirement
------------------------
    conda install -c conda-forge pynini=2.1.5
    pip install nemo_text_processing
or:
    pip install "multilingual-normalizer[nemo]"
"""

from __future__ import annotations

import logging
from typing import Optional

from .language_utils import get_base_lang

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported NeMo TN languages
# ---------------------------------------------------------------------------

_NEMO_SUPPORTED_LANGS: frozenset[str] = frozenset({
    "en", "es", "fr", "de", "ar", "ru",
    "sv", "vi", "pt", "zh", "hu", "it",
    "hy", "mr", "hi", "ja",
})

# ---------------------------------------------------------------------------
# Lazy normalizer cache  {lang_2letter: NormalizerWithAudio | None}
# None means we already tried and it failed (import error / grammar error).
# ---------------------------------------------------------------------------

_normalizer_cache: dict[str, object] = {}


def _get_nemo_normalizer(
    base_lang: str,
    cache_dir: Optional[str] = None,
) -> object:
    """
    Return a cached ``NormalizerWithAudio`` for *base_lang*, or None if
    nemo_text_processing is not installed or grammar loading fails.
    """
    if base_lang in _normalizer_cache:
        return _normalizer_cache[base_lang]

    try:
        from nemo_text_processing.text_normalization.normalize_with_audio import (
            NormalizerWithAudio,
        )
    except ImportError:
        logger.warning(
            "nemo_text_processing is not installed. "
            "Install it with: pip install 'multilingual-normalizer[nemo]'\n"
            "Note: pynini must be installed first via conda: "
            "conda install -c conda-forge pynini=2.1.5"
        )
        _normalizer_cache[base_lang] = None
        return None

    try:
        logger.info("Loading NeMo WFST normalizer for language: %s", base_lang)
        normalizer = NormalizerWithAudio(
            input_case="cased",
            lang=base_lang,
            cache_dir=cache_dir,
        )
        _normalizer_cache[base_lang] = normalizer
        logger.info("NeMo normalizer loaded for language: %s", base_lang)
        return normalizer
    except Exception as exc:
        logger.warning(
            "Failed to load NeMo normalizer for lang='%s': %s", base_lang, exc
        )
        _normalizer_cache[base_lang] = None
        return None


# ---------------------------------------------------------------------------
# Deduplication (mirrors candidates.py)
# ---------------------------------------------------------------------------

def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_candidates_nemo(
    span: object,
    lang: str,
    n_tagged: int = 30,
    cache_dir: Optional[str] = None,
) -> list[str]:
    """
    Generate spoken-form candidates for *span* using NeMo's WFST normalizer.

    Args:
        span:      A ``NumberSpan`` from ``detector.detect()``.
        lang:      Any language code (ISO 639-1/3, BCP-47).
        n_tagged:  Maximum number of tagged options to generate. Higher values
                   produce more candidates but increase latency.
        cache_dir: Directory for caching compiled ``.far`` grammar files.
                   Re-using a cache dramatically speeds up the second and
                   subsequent loads for the same language.

    Returns:
        Deduplicated list of candidate strings; empty list if NeMo does not
        support the language or if generation fails for any reason.
        The caller should fall back to the num2words backend when [] is returned.
    """
    base = get_base_lang(lang)

    if base not in _NEMO_SUPPORTED_LANGS:
        logger.debug(
            "NeMo backend: language '%s' (base='%s') not supported — "
            "falling back to num2words.",
            lang, base,
        )
        return []

    normalizer = _get_nemo_normalizer(base, cache_dir)
    if normalizer is None:
        return []

    try:
        raw = normalizer.normalize_non_deterministic(
            text=span.text,
            n_tagged=n_tagged,
        )
        # normalize_non_deterministic returns a set (or list) of strings
        if isinstance(raw, (set, list, tuple)):
            return _dedup(list(raw))
        # Unexpected return type — log and give up
        logger.warning(
            "NeMo normalize_non_deterministic returned unexpected type %s for "
            "text=%r; falling back.",
            type(raw).__name__, span.text,
        )
        return []
    except Exception as exc:
        logger.debug(
            "NeMo candidate generation failed for text=%r lang=%s: %s",
            span.text, lang, exc,
        )
        return []


def clear_cache() -> None:
    """
    Evict all cached NeMo normalizer instances (useful in tests or when
    switching ``cache_dir`` at runtime).
    """
    _normalizer_cache.clear()

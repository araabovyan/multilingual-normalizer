"""
Main pipeline: NumberNormalizer

Detects numbers in text → generates candidate spoken forms →
optionally scores candidates against audio with CTC → rewrites text.

Usage
-----
    from normalizer import NumberNormalizer
    import soundfile as sf

    nn = NumberNormalizer()

    # Audio-guided (recommended)
    audio, sr = sf.read("speech.wav", dtype="float32")
    result = nn.normalize(audio, "pay $31 by 3/15", lang="en", sample_rate=sr)
    print(result.normalized_text)

    # Text-only fallback (no model loading)
    print(nn.normalize_text_only("born in 1984", lang="en"))
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .detector import detect, NumberSpan, NumberType
from .candidates import generate_candidates
from .scorer import CTCScorer
from .language_utils import get_mms_lang, get_base_lang


# ---------------------------------------------------------------------------
# English compound-number hyphenation
# ---------------------------------------------------------------------------

# Standard English rule: compound numbers twenty-one through ninety-nine are
# hyphenated when written.  Two separate patterns:
#   1. GLOBAL cardinal rule — applied to the full output text
#   2. PER-SPAN ordinal rule — applied only to the replacement of ORDINAL spans
#      (not globally, so ordinals embedded in dates/times are left as-is)
_EN_COMPOUND_RE = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"[ \t]+"
    r"(one|two|three|four|five|six|seven|eight|nine)\b",
    re.IGNORECASE,
)

_EN_ORDINAL_COMPOUND_RE = re.compile(
    r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"[ \t]+"
    r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)\b",
    re.IGNORECASE,
)


def _apply_english_hyphens(text: str) -> str:
    """Apply standard English hyphenation to cardinal compound numbers in *text*."""
    return _EN_COMPOUND_RE.sub(r"\1-\2", text)


def _apply_ordinal_hyphens(text: str) -> str:
    """Apply standard English hyphenation to ordinal compound numbers in *text*."""
    return _EN_ORDINAL_COMPOUND_RE.sub(r"\1-\2", text)


# Ordinal suffix merger: NeMo sometimes detaches the ordinal suffix as a separate
# word (e.g. "one oh five th" instead of "one oh fifth").
# Applied to ORDINAL span replacements after choosing the best candidate.
_ORDINAL_SUFFIX_FIXES: list[tuple[re.Pattern, str]] = [
    # Irregular ordinals — must come before the generic glue rule
    (re.compile(r"\bone\s+st\b",      re.IGNORECASE), "first"),
    (re.compile(r"\btwo\s+nd\b",      re.IGNORECASE), "second"),
    (re.compile(r"\bthree\s+rd\b",    re.IGNORECASE), "third"),
    (re.compile(r"\bfive\s+th\b",     re.IGNORECASE), "fifth"),
    (re.compile(r"\beight\s+th\b",    re.IGNORECASE), "eighth"),
    (re.compile(r"\bnine\s+th\b",     re.IGNORECASE), "ninth"),
    (re.compile(r"\btwelve\s+th\b",   re.IGNORECASE), "twelfth"),
    (re.compile(r"\btwenty\s+th\b",   re.IGNORECASE), "twentieth"),
    (re.compile(r"\bthirty\s+th\b",   re.IGNORECASE), "thirtieth"),
    (re.compile(r"\bforty\s+th\b",    re.IGNORECASE), "fortieth"),
    (re.compile(r"\bfifty\s+th\b",    re.IGNORECASE), "fiftieth"),
    (re.compile(r"\bsixty\s+th\b",    re.IGNORECASE), "sixtieth"),
    (re.compile(r"\bseventy\s+th\b",  re.IGNORECASE), "seventieth"),
    (re.compile(r"\beighty\s+th\b",   re.IGNORECASE), "eightieth"),
    (re.compile(r"\bninety\s+th\b",   re.IGNORECASE), "ninetieth"),
    # Regular: glue suffix to preceding word (e.g. "six th" → "sixth")
    (re.compile(r"\b(\w+)\s+(th|st|nd|rd)\b", re.IGNORECASE), r"\1\2"),
]


def _fix_ordinal_suffix(text: str) -> str:
    """Merge detached ordinal suffixes (e.g. 'five th' → 'fifth')."""
    for pattern, repl in _ORDINAL_SUFFIX_FIXES:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# Unit abbreviation expansion
# ---------------------------------------------------------------------------

# Ordered so longer/more-specific patterns are tried first (km/h before km).
_UNIT_EXPANSIONS_EN: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bkm/h\b"),  "kilometers per hour"),
    (re.compile(r"\bkm\b"),    "kilometers"),
    (re.compile(r"\bmph\b"),   "miles per hour"),
    (re.compile(r"\bkph\b"),   "kilometers per hour"),
    (re.compile(r"\bm/s\b"),   "meters per second"),
    (re.compile(r"\bkg\b"),    "kilograms"),
    (re.compile(r"\blbs\b"),   "pounds"),
    (re.compile(r"\blb\b"),    "pounds"),
    (re.compile(r"\bcm\b"),    "centimeters"),
    (re.compile(r"\bmm\b"),    "millimeters"),
    (re.compile(r"\bml\b"),    "milliliters"),
    (re.compile(r"\bGHz\b"),   "gigahertz"),
    (re.compile(r"\bMHz\b"),   "megahertz"),
    (re.compile(r"\bkHz\b"),   "kilohertz"),
    (re.compile(r"\bHz\b"),    "hertz"),
    (re.compile(r"\bkWh\b"),   "kilowatt hours"),
    (re.compile(r"\bkW\b"),    "kilowatts"),
    (re.compile(r"\bMW\b"),    "megawatts"),
    (re.compile(r"\bGW\b"),    "gigawatts"),
    (re.compile(r"\bGB\b"),    "gigabytes"),
    (re.compile(r"\bMB\b"),    "megabytes"),
    (re.compile(r"\bTB\b"),    "terabytes"),
    (re.compile(r"\bkB\b"),    "kilobytes"),
]


def _expand_units(text: str) -> str:
    """Expand common measurement unit abbreviations to their spoken forms."""
    for pattern, expansion in _UNIT_EXPANSIONS_EN:
        text = pattern.sub(expansion, text)
    return text


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_full_sentence(
    text: str,
    spans: list[NumberSpan],
    target_span: NumberSpan,
    candidate: str,
    default_forms: dict[str, str],
) -> str:
    """
    Reconstruct *text* with every number span substituted by its spoken form.

    The *target_span* is replaced by *candidate*; all other spans are replaced
    by their entry in *default_forms* (their num2words default form).

    Substitutions are applied right-to-left so character offsets stay valid.
    """
    result = text
    for sp in sorted(spans, key=lambda s: s.start, reverse=True):
        repl = candidate if sp is target_span else default_forms[sp.text]
        result = result[: sp.start] + repl + result[sp.end :]
    return result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NormalizationResult:
    """Holds the full output of a normalize() call."""
    original_text:  str
    normalized_text: str
    spans:           list[NumberSpan]
    chosen_forms:    dict[str, str]              # original token → chosen spoken form
    all_scores:      dict[str, dict[str, float]] # original token → {candidate → CTC score}
    lang:            str
    audio_used:      bool = False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NumberNormalizer:
    """
    Multilingual audio-grounded number normalizer.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for CTC scoring.
        Default: ``"facebook/mms-1b-all"`` (1,107 languages).
    device : str, optional
        ``"cuda"``, ``"cpu"``, or None for auto-detect.
    lazy_load : bool
        If True (default), the model is loaded on the first call that
        needs it.  Pass False to eagerly pre-warm for a specific language
        (call ``preload(lang)`` after construction).
    candidate_backend : str
        Candidate generation backend: ``"nemo"`` (default) or ``"num2words"``.
        ``"nemo"`` uses NeMo's non-deterministic WFST normalizer to enumerate
        spoken-form candidates.  When ``"nemo"`` is selected but the language
        is not supported by NeMo (or nemo_text_processing is not installed),
        the backend automatically falls back to ``"num2words"``.
        NeMo TN supports: en, es, fr, de, ar, ru, sv, vi, pt, zh, hu, it,
        hy, mr, hi, ja.
    nemo_n_tagged : int
        Maximum number of tagged options to request from NeMo per span
        (only used when candidate_backend="nemo"). Default: 30.
    nemo_cache_dir : str, optional
        Directory for NeMo's compiled ``.far`` grammar cache.  Re-using a
        cache directory avoids re-compiling grammars on every run
        (only used when candidate_backend="nemo").
    """

    def __init__(
        self,
        model_name: str = "facebook/mms-1b-all",
        device: Optional[str] = None,
        lazy_load: bool = True,
        candidate_backend: str = "nemo",
        nemo_n_tagged: int = 30,
        nemo_cache_dir: Optional[str] = None,
    ) -> None:
        self._scorer = CTCScorer(model_name=model_name, device=device)
        self._candidate_backend = candidate_backend
        self._nemo_n_tagged = nemo_n_tagged
        self._nemo_cache_dir = nemo_cache_dir
        if not lazy_load:
            # Eagerly warm the model for a placeholder language so the first
            # real call has no cold-start latency.  Callers should also call
            # preload(lang) for their specific language to swap the adapter.
            import threading
            threading.Thread(target=self._scorer._ensure_loaded, args=("eng",), daemon=True).start()

    # ------------------------------------------------------------------
    # Pre-warming
    # ------------------------------------------------------------------

    def preload(self, lang: str) -> None:
        """Pre-load model and language adapter (avoids latency on first call)."""
        lang_mms = get_mms_lang(lang)
        if lang_mms:
            dummy = np.zeros(16000, dtype=np.float32)   # 1 s silence
            self._scorer.score_candidates(dummy, ["test"], lang_mms)
            logger.info("Model pre-loaded for language: %s (%s)", lang, lang_mms)
        else:
            logger.warning("Unknown MMS code for lang='%s' — skipping preload.", lang)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def normalize(
        self,
        audio: Optional[np.ndarray],
        text: str,
        lang: str,
        sample_rate: int = 16000,
    ) -> NormalizationResult:
        """
        Normalize all numbers in *text* to their spoken form.

        When *audio* is provided (and the language is supported by MMS),
        candidate spoken forms are scored against the audio and the best
        match is selected.  When *audio* is None, the default num2words
        verbalization is used.

        Parameters
        ----------
        audio :       1-D float32 array (mono), or None for text-only mode.
        text :        Input text containing written numbers.
        lang :        Language code (ISO 639-1, ISO 639-3, or BCP-47).
        sample_rate : Sample rate of *audio* (will be resampled to 16 kHz).

        Returns
        -------
        NormalizationResult
        """
        spans = detect(text)
        if not spans:
            return NormalizationResult(
                original_text=text, normalized_text=text,
                spans=[], chosen_forms={}, all_scores={},
                lang=lang, audio_used=False,
            )

        lang_mms  = get_mms_lang(lang)
        use_audio = (audio is not None) and (lang_mms is not None)

        # Pre-compute the default (first candidate) spoken form for every span.
        # Used as placeholders for non-targeted spans when building full sentences.
        all_candidates: dict[str, list[str]] = {
            s.text: generate_candidates(
                s, lang,
                backend=self._candidate_backend,
                nemo_n_tagged=self._nemo_n_tagged,
                nemo_cache_dir=self._nemo_cache_dir,
            )
            for s in spans
        }
        default_forms: dict[str, str] = {
            s.text: all_candidates[s.text][0] for s in spans
        }

        chosen_forms: dict[str, str]              = {}
        all_scores:   dict[str, dict[str, float]] = {}

        for span in spans:
            candidates = all_candidates[span.text]
            if not candidates:
                continue

            if use_audio and len(candidates) > 1:
                # Build one full-sentence variant per candidate.
                # The span under test is replaced by the candidate; all other
                # spans are replaced by their default spoken form so the scorer
                # never sees raw digits or currency symbols.
                full_texts = [
                    _build_full_sentence(text, spans, span, c, default_forms)
                    for c in candidates
                ]
                raw_scores = self._scorer.score_candidates(
                    audio, full_texts, lang_mms, sample_rate
                )
                # Remap {full_sentence: score} → {fragment_candidate: score}
                scores = {c: raw_scores[full_texts[i]] for i, c in enumerate(candidates)}
                best   = max(scores, key=scores.__getitem__)
                all_scores[span.text] = scores
            else:
                best   = candidates[0]
                all_scores[span.text] = {c: 0.0 for c in candidates}

            chosen_forms[span.text] = best

        # Rebuild text in reverse order to preserve character offsets
        normalized = text
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            if span.text in chosen_forms:
                repl = chosen_forms[span.text]
                if get_base_lang(lang) == "en":
                    # Normalize standalone "o" as zero placeholder → "oh"
                    repl = re.sub(r"\bo\b(?!')", "oh", repl)
                    # For ORDINAL spans only: apply ordinal compound hyphenation
                    # (e.g. "twenty second" → "twenty-second").  This is NOT applied
                    # globally so ordinal words embedded in dates remain un-hyphenated.
                    if span.number_type == NumberType.ORDINAL:
                        repl = _apply_ordinal_hyphens(repl)
                        repl = _fix_ordinal_suffix(repl)
                normalized = normalized[: span.start] + repl + normalized[span.end :]

        # Apply standard English hyphenation to compound numbers
        # ("twenty two" → "twenty-two", "thirty second" → "thirty-second", etc.)
        if get_base_lang(lang) == "en":
            normalized = _apply_english_hyphens(normalized)
            normalized = _expand_units(normalized)
        return NormalizationResult(
            original_text=text,
            normalized_text=normalized,
            spans=spans,
            chosen_forms=chosen_forms,
            all_scores=all_scores,
            lang=lang,
            audio_used=use_audio,
        )

    def normalize_text_only(self, text: str, lang: str) -> str:
        """
        Normalize using the default spoken form (no audio required).

        Equivalent to ``normalize(None, text, lang).normalized_text``.
        Fast — does not load any model.
        """
        return self.normalize(audio=None, text=text, lang=lang).normalized_text

    def batch_normalize(
        self,
        pairs: list[tuple[Optional[np.ndarray], str]],
        lang: str,
        sample_rate: int = 16000,
    ) -> list[NormalizationResult]:
        """
        Normalize a batch of (audio, text) pairs for the same language.

        The MMS model is loaded once; adapters are not reloaded between items.
        Currently sequential; future versions may parallelize the encoder pass.
        """
        return [
            self.normalize(audio, text, lang, sample_rate)
            for audio, text in pairs
        ]

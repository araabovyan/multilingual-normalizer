"""
multilingual-normalizer
=======================
Audio-guided multilingual speech number normalizer.

Given a (audio, text) pair, converts written numbers in the text to the
spoken form as heard in the audio. Uses CTC scoring with Meta's MMS model
(1,107+ languages) to discriminate between multiple valid verbalizations.

Quick start
-----------
    from normalizer import NumberNormalizer
    import soundfile as sf
    import numpy as np

    normalizer = NumberNormalizer()

    # With audio guidance
    audio, sr = sf.read("speech.wav", dtype="float32")
    result = normalizer.normalize(audio, "pay $31 by 3/15", lang="en", sample_rate=sr)
    print(result.normalized_text)  # "pay thirty-one dollars by march fifteenth"

    # Text-only (uses default num2words form)
    text = normalizer.normalize_text_only("he was born in 1984", lang="en")
    print(text)  # "he was born in nineteen eighty-four"
"""

from .pipeline import NumberNormalizer, NormalizationResult
from .detector import detect, NumberSpan, NumberType
from .candidates import generate_candidates
from .language_utils import get_mms_lang, get_num2words_lang

__all__ = [
    "NumberNormalizer",
    "NormalizationResult",
    "detect",
    "NumberSpan",
    "NumberType",
    "generate_candidates",
    "get_mms_lang",
    "get_num2words_lang",
]

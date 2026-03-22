"""
Language code normalization utilities.

Maps between user-supplied codes (BCP-47, ISO 639-1, ISO 639-3) and the
codes expected by num2words and Meta's MMS model respectively.
"""

from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# num2words language tags
# Maps lowercase normalised input → num2words lang string
# ---------------------------------------------------------------------------
_NUM2WORDS_MAP: dict[str, str] = {
    # English
    "en": "en", "eng": "en", "en-us": "en", "en-gb": "en_GB",
    "en-in": "en_IN", "en-ng": "en_NG",
    # Spanish
    "es": "es", "spa": "es", "es-es": "es", "es-co": "es_CO",
    "es-cr": "es_CR", "es-gt": "es_GT", "es-ve": "es_VE",
    # Russian
    "ru": "ru", "rus": "ru",
}

# ---------------------------------------------------------------------------
# MMS ISO 639-3 language codes
# Maps lowercase normalised input → MMS adapter language ID
# ---------------------------------------------------------------------------
_MMS_MAP: dict[str, str] = {
    # English
    "en": "eng", "eng": "eng", "en-us": "eng", "en-gb": "eng",
    "en-in": "eng", "en-au": "eng",
    # Spanish
    "es": "spa", "spa": "spa", "es-es": "spa", "es-419": "spa",
    # Russian
    "ru": "rus", "rus": "rus",
}

# ---------------------------------------------------------------------------
# Digit words per language  (index 0–9)
# ---------------------------------------------------------------------------
_DIGIT_WORDS: dict[str, list[str]] = {
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    "es": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"],
    "ru": ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"],
}

# Zero/oh variants per language (useful for year and phone reading)
_ZERO_VARIANTS: dict[str, list[str]] = {
    "en": ["zero", "oh"],
    "es": ["cero"],
    "ru": ["ноль", "нуль"],
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _normalise_key(lang: str) -> str:
    return lang.lower().strip()


def get_num2words_lang(lang: str) -> Optional[str]:
    """Return the num2words language tag, or None if unsupported."""
    return _NUM2WORDS_MAP.get(_normalise_key(lang))


def get_mms_lang(lang: str) -> Optional[str]:
    """Return the MMS ISO 639-3 code, or None if unmapped.

    If the input is already a 3-letter alphabetic code not in the map,
    it is returned as-is (pass-through for direct ISO 639-3 inputs).
    """
    key = _normalise_key(lang)
    result = _MMS_MAP.get(key)
    if result is None and len(key) == 3 and key.isalpha():
        return key  # already ISO 639-3 — try as-is
    return result


def get_base_lang(lang: str) -> str:
    """Return the base language tag (strip region/script subtags)."""
    return _normalise_key(lang).split("-")[0].split("_")[0]


def get_digit_words(lang: str) -> list[str]:
    """Return spoken digit words for 0–9."""
    return _DIGIT_WORDS.get(get_base_lang(lang), _DIGIT_WORDS["en"])


def get_zero_word(lang: str) -> str:
    """Return the primary 'zero' word for the language."""
    variants = _ZERO_VARIANTS.get(get_base_lang(lang), ["zero"])
    return variants[0]


def get_zero_variants(lang: str) -> list[str]:
    """Return all 'zero'/'oh' variants (useful for year / phone numbers)."""
    return _ZERO_VARIANTS.get(get_base_lang(lang), ["zero"])

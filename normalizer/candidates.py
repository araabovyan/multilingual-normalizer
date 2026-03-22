"""
Generate candidate spoken-form normalizations for a NumberSpan.

For each number type (cardinal, ordinal, decimal, currency, percentage,
year, date, phone) this module produces a ranked list of plausible
verbalizations in the target language.  The first item is the most
"standard" form; later items are alternates that the audio may match.

All audio-independent: no model loading here.
"""

from __future__ import annotations

from typing import Optional

try:
    from num2words import num2words as _n2w
    _N2W_OK = True
except ImportError:
    _N2W_OK = False

from .detector import NumberSpan, NumberType, CURRENCY_SYMBOLS
from .language_utils import (
    get_num2words_lang,
    get_base_lang,
    get_digit_words,
    get_zero_word,
    get_zero_variants,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_candidates(
    span: NumberSpan,
    lang: str,
    backend: str = "nemo",
    nemo_n_tagged: int = 30,
    nemo_cache_dir: Optional[str] = None,
) -> list[str]:
    """
    Return a ranked list of candidate spoken forms for *span* in *lang*.

    Args:
        span:           A NumberSpan from detector.detect()
        lang:           Any language code (ISO 639-1/3, BCP-47)
        backend:        ``"nemo"`` (default) or ``"num2words"``.
                        When ``"nemo"`` is selected, NeMo's non-deterministic
                        WFST normalizer is used to generate candidates.  If NeMo
                        is not installed or does not support *lang*, the call
                        automatically falls back to the num2words backend.
                        NeMo and num2words candidates are merged so neither
                        source's coverage gaps cause missed spoken forms.
        nemo_n_tagged:  Maximum tagged options to request from NeMo (only used
                        when backend="nemo").
        nemo_cache_dir: Directory for NeMo's compiled ``.far`` grammar cache
                        (only used when backend="nemo").

    Returns:
        Deduplicated list of candidate strings, most likely first.
        Always contains at least one entry.
    """
    # ── NeMo backend ────────────────────────────────────────────────────────
    nemo_cands: list[str] = []
    if backend == "nemo":
        from .nemo_candidates import generate_candidates_nemo
        raw_nemo = generate_candidates_nemo(span, lang, nemo_n_tagged, nemo_cache_dir)
        # Sanitize NeMo candidates:
        #   • drop those with literal '.' (e.g. "dollar four.ninety nine") — formatting artifacts
        #   • drop those with ',' (e.g. "one eight hundred, five five five") — punctuation artifacts
        #   • drop those with '/' (e.g. "one/two") — fraction artifacts
        #   • strip hyphens: NeMo sometimes preserves source hyphens (e.g. "one-eight hundred"
        #     from "1-800-..."); our pipeline re-adds standard compound hyphens in post-processing.
        cleaned: list[str] = []
        for c in raw_nemo:
            if "." in c or "," in c or "/" in c:
                continue
            c_clean = " ".join(c.replace("-", " ").split())
            if c_clean:
                cleaned.append(c_clean)
        nemo_cands = _dedup(cleaned)

    # ── num2words backend (always generated; merged with NeMo when available) ─
    n2w = get_num2words_lang(lang)

    if span.number_type == NumberType.CARDINAL:
        cands = _cardinal(span.integer, n2w, lang)
    elif span.number_type == NumberType.ORDINAL:
        cands = _ordinal(span.integer, n2w, lang)
    elif span.number_type == NumberType.DECIMAL:
        cands = _decimal(span.integer, span.decimal_str, n2w, lang)
    elif span.number_type == NumberType.CURRENCY:
        cands = _currency(span.integer, span.decimal_str, span.currency_symbol, n2w, lang)
    elif span.number_type == NumberType.PERCENTAGE:
        cands = _percentage(span.integer, span.decimal_str, n2w, lang)
    elif span.number_type == NumberType.DATE:
        cands = _date(span.date_parts, n2w, lang)
    elif span.number_type == NumberType.PHONE:
        cands = _phone(span.raw_digits or "", n2w, lang)
    elif span.number_type == NumberType.YEAR:
        cands = _year(span.integer, n2w, lang)
    elif span.number_type == NumberType.TIME:
        cands = _time(span.time_parts, n2w, lang)
    elif span.number_type == NumberType.FRACTION:
        cands = _fraction(span.fraction_parts, n2w, lang)
    elif span.number_type == NumberType.VERSION:
        cands = _version(span.raw_digits or span.text, n2w, lang)
    else:
        cands = _cardinal(span.integer, n2w, lang)

    deduped = _dedup(cands)
    # For candidates that contain hyphens (num2words style: "twenty-two", "ninety-nine"),
    # also include the non-hyphenated form — ground truths vary in whether they use hyphens.
    extra = [" ".join(c.replace("-", " ").split()) for c in deduped if "-" in c]
    n2w_cands = _dedup(deduped + extra)

    # ── Merge NeMo + num2words ───────────────────────────────────────────────
    # NeMo candidates appear first (usually higher-quality standard forms);
    # num2words supplements with forms NeMo may not generate (colloquial times,
    # archaic year readings, currency shorthands, etc.).
    if nemo_cands:
        merged = _dedup(nemo_cands + n2w_cands)
    else:
        merged = n2w_cands

    # Ensure at least the raw text is present as last-resort fallback
    if not merged:
        merged = [span.text]
    return merged


# ---------------------------------------------------------------------------
# num2words wrapper
# ---------------------------------------------------------------------------

def _n2w_safe(n: int | float, lang: Optional[str], to: str = "cardinal", **kwargs) -> Optional[str]:
    if not _N2W_OK or lang is None:
        return None
    try:
        result = _n2w(n, lang=lang, to=to, **kwargs)
        if result:
            # Strip thousands-separator commas only ("one thousand, two hundred" → "one thousand two hundred").
            # Hyphens are preserved so "twenty-two" stays as-is; we separately add the
            # non-hyphenated variant in generate_candidates() so both forms are candidates.
            result = result.replace(",", "")
            result = " ".join(result.split())
        return result or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Deduplication
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
# Cardinal
# ---------------------------------------------------------------------------

def _cardinal(n: int, n2w: Optional[str], lang: str) -> list[str]:
    cands: list[str] = []

    std = _n2w_safe(n, n2w)
    if std:
        cands.append(std)
        # Variant without "and" (British English inserts "and": "one hundred and one")
        alt = std.replace(" and ", " ").replace("-and-", " ")
        if alt != std:
            cands.append(alt)

    # Digit-by-digit (useful for codes, serial numbers); one form per zero variant
    cands.extend(_digits_spoken_variants(str(n), lang))

    # For 3-digit numbers, add grouped "first digit + last two" reading
    # e.g. 314 → "three fourteen", 772 → "seven seventy two"
    if 100 <= n <= 999 and n % 100 > 9:
        first_spoken = _n2w_safe(n // 100, n2w) or _digit_word(str(n // 100), lang)
        last_spoken  = _n2w_safe(n % 100, n2w)
        if first_spoken and last_spoken:
            cands.append(f"{first_spoken} {last_spoken}")

    # For numbers in year range add year-style reading as an alternate
    if 1000 <= n <= 2099:
        cands.extend(_year(n, n2w, lang))

    return cands


# ---------------------------------------------------------------------------
# Ordinal
# ---------------------------------------------------------------------------

def _ordinal(n: int, n2w: Optional[str], lang: str) -> list[str]:
    cands: list[str] = []
    if n2w:
        ord_form = _n2w_safe(n, n2w, to="ordinal")
        if ord_form:
            cands.append(ord_form)
    # Cardinal as fallback (some spoken ordinals sound cardinal)
    std = _n2w_safe(n, n2w)
    if std:
        cands.append(std)
    return cands


# ---------------------------------------------------------------------------
# Decimal
# ---------------------------------------------------------------------------

def _decimal(integer: int, dec_str: Optional[str], n2w: Optional[str], lang: str) -> list[str]:
    if dec_str is None:
        return _cardinal(integer, n2w, lang)

    cands: list[str] = []
    point = _point_word(lang)
    int_spoken = _n2w_safe(integer, n2w) or str(integer)

    # "three point one four"  (digit-by-digit decimal)
    dec_digits_spoken = " ".join(_digit_word(d, lang) for d in dec_str)
    cands.append(f"{int_spoken} {point} {dec_digits_spoken}")

    # "three point fourteen"  (decimal part as a number)
    # Count leading zeros so we can re-inject them before the spoken number.
    n_leading_zeros = len(dec_str) - len(dec_str.lstrip("0"))
    dec_val = int(dec_str.lstrip("0") or "0")
    dec_spoken = _n2w_safe(dec_val, n2w)
    if dec_spoken:
        if n_leading_zeros > 0:
            zeros_spoken = " ".join(_digit_word("0", lang) for _ in range(n_leading_zeros))
            cands.append(f"{int_spoken} {point} {zeros_spoken} {dec_spoken}")
        else:
            cands.append(f"{int_spoken} {point} {dec_spoken}")

    # Grouped 2-digit-pair reading: "3.1416" → "three point fourteen sixteen"
    # Used for long decimals (e.g. precise measurements, pi digits).
    if len(dec_str) >= 4 and len(dec_str) % 2 == 0:
        pairs = [dec_str[i:i + 2] for i in range(0, len(dec_str), 2)]
        pair_spoken = " ".join(_n2w_safe(int(p), n2w) or p for p in pairs)
        if pair_spoken:
            cands.append(f"{int_spoken} {point} {pair_spoken}")

    return cands


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

# (singular, plural, cent-singular, cent-plural)
_CURR_NAMES: dict[str, dict[str, tuple[str, str, str, str]]] = {
    "en": {
        "$":  ("dollar",  "dollars",  "cent",    "cents"),
        "€":  ("euro",    "euros",    "cent",    "cents"),
        "£":  ("pound",   "pounds",   "penny",   "pence"),
        "¥":  ("yen",     "yen",      "sen",     "sen"),
        "₹":  ("rupee",   "rupees",   "paisa",   "paise"),
        "₩":  ("won",     "won",      "jeon",    "jeon"),
        "₽":  ("ruble",   "rubles",   "kopek",   "kopeks"),
        "R$": ("real",    "reais",    "centavo", "centavos"),
        "₿":  ("bitcoin", "bitcoins", "satoshi", "satoshis"),
    },
    "es": {
        "$":  ("dólar",  "dólares",  "centavo", "centavos"),
        "€":  ("euro",   "euros",    "céntimo", "céntimos"),
        "£":  ("libra",  "libras",   "penique", "peniques"),
    },
    "ru": {
        "$":  ("доллар", "долларов", "цент",   "центов"),
        "€":  ("евро",   "евро",     "цент",   "центов"),
        "£":  ("фунт",   "фунтов",   "пенс",   "пенсов"),
        "₽":  ("рубль",  "рублей",   "копейка","копеек"),
    },
}


def _currency(
    integer: int,
    dec_str: Optional[str],
    symbol: Optional[str],
    n2w: Optional[str],
    lang: str,
) -> list[str]:
    base = get_base_lang(lang)
    lang_names = _CURR_NAMES.get(base, _CURR_NAMES.get("en", {}))
    names = lang_names.get(symbol or "$")

    int_spoken = _n2w_safe(integer, n2w) or str(integer)
    cands: list[str] = []

    if names:
        sing, plur, csing, cplur = names
        curr_word = sing if integer == 1 else plur

        if not dec_str or all(d == "0" for d in dec_str):
            cands.append(f"{int_spoken} {curr_word}")
            # Year-style alternative for amounts like £1500 → "fifteen hundred pounds"
            if 1000 <= integer <= 2099:
                for yr_form in _year(integer, n2w, lang):
                    cands.append(f"{yr_form} {curr_word}")
        else:
            cents = int(dec_str.ljust(2, "0")[:2])
            cents_spoken = _n2w_safe(cents, n2w) or str(cents)
            cent_word = csing if cents == 1 else cplur

            cands.append(f"{int_spoken} {curr_word} and {cents_spoken} {cent_word}")
            cands.append(f"{int_spoken} {curr_word} {cents_spoken} {cent_word}")
            # Without cent word: "twelve pounds ninety nine" (common colloquial form)
            cands.append(f"{int_spoken} {curr_word} {cents_spoken}")

            # Colloquial: "four ninety nine" / "one fifty"
            dec_spoken = _n2w_safe(cents, n2w) or str(cents)
            if dec_spoken:
                cands.append(f"{int_spoken} {dec_spoken}")
                # "a dollar fifty" — when integer is 1 use article "a" instead
                if integer == 1:
                    cands.append(f"a {sing} {dec_spoken}")
    else:
        # Unknown currency — just say the number
        cands.append(int_spoken)

    # Try num2words built-in currency mode
    if n2w and dec_str:
        cents_int = int(dec_str.ljust(2, "0")[:2])
        full_val = integer + cents_int / 100
        curr_code = CURRENCY_SYMBOLS.get(symbol or "")
        if curr_code:
            try:
                n2w_curr = _n2w_safe(full_val, n2w, to="currency", currency=curr_code)
                if n2w_curr:
                    cands.append(n2w_curr)
            except Exception:
                pass

    return cands


# ---------------------------------------------------------------------------
# Percentage
# ---------------------------------------------------------------------------

_PCT_WORD: dict[str, str] = {
    "en": "percent",
    "es": "por ciento",
    "ru": "процентов",
}


def _percentage(
    integer: int,
    dec_str: Optional[str],
    n2w: Optional[str],
    lang: str,
) -> list[str]:
    base = get_base_lang(lang)
    pct = _PCT_WORD.get(base, "percent")
    int_spoken = _n2w_safe(integer, n2w) or str(integer)
    cands: list[str] = []

    if dec_str:
        point = _point_word(lang)
        dec_digits_spoken = " ".join(_digit_word(d, lang) for d in dec_str)
        cands.append(f"{int_spoken} {point} {dec_digits_spoken} {pct}")
    else:
        cands.append(f"{int_spoken} {pct}")

    return cands


# ---------------------------------------------------------------------------
# Year
# ---------------------------------------------------------------------------

_HUNDRED_WORD: dict[str, str] = {
    "en": "hundred",
    "es": "cien",
    "ru": "сто",
}


def _year(n: int, n2w: Optional[str], lang: str) -> list[str]:
    """Year-style spoken form: "nineteen eighty-four", "twenty twenty-four"."""
    if not (1000 <= n <= 2099):
        return _cardinal(n, n2w, lang)

    base = get_base_lang(lang)
    cands: list[str] = []
    high = n // 100   # e.g. 19 for 1984
    low  = n % 100    # e.g. 84 for 1984

    high_spoken = _n2w_safe(high, n2w) or str(high)
    hundred_word = _HUNDRED_WORD.get(base, "hundred")
    zero_word = get_zero_word(lang)
    zero_vars = get_zero_variants(lang)

    # Two-chunk year reading
    if high_spoken:
        if low == 0:
            # 1900 → "nineteen hundred"
            # Exclude 2000: "twenty hundred" is non-standard; "two thousand" comes from std below.
            if high != 20:
                cands.append(f"{high_spoken} {hundred_word}")
        elif 1 <= low <= 9:
            # 1905 → "nineteen oh five" / "nineteen zero five"
            # Also: "nineteen hundred and five" (archaic but common in speech)
            low_spoken = _n2w_safe(low, n2w) or str(low)
            if low_spoken:
                for zv in zero_vars:
                    cands.append(f"{high_spoken} {zv} {low_spoken}")
                cands.append(f"{high_spoken} {hundred_word} and {low_spoken}")
        else:
            # 1984 → "nineteen eighty-four"
            low_spoken = _n2w_safe(low, n2w) or str(low)
            if low_spoken:
                cands.append(f"{high_spoken} {low_spoken}")

    # Standard cardinal (always include as alternate)
    std = _n2w_safe(n, n2w)
    if std:
        cands.append(std)

    # 2000–2099: "two thousand [and] twenty-four" variant
    if 2000 <= n <= 2099 and low > 0:
        two_thou = _n2w_safe(2000, n2w) or "two thousand"
        low_spoken = _n2w_safe(low, n2w) or str(low)
        if two_thou and low_spoken:
            cands.append(f"{two_thou} and {low_spoken}")
            cands.append(f"{two_thou} {low_spoken}")

    # Digit-by-digit (e.g. "two oh oh five" / "two zero zero five")
    cands.extend(_digits_spoken_variants(str(n), lang))

    return cands


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------

_MONTHS: dict[str, list[str]] = {
    "en": ["", "january","february","march","april","may","june",
               "july","august","september","october","november","december"],
    "es": ["", "enero","febrero","marzo","abril","mayo","junio",
               "julio","agosto","septiembre","octubre","noviembre","diciembre"],
    "ru": ["", "января","февраля","марта","апреля","мая","июня",
               "июля","августа","сентября","октября","ноября","декабря"],
}


def _date(date_parts: Optional[dict], n2w: Optional[str], lang: str) -> list[str]:
    if not date_parts:
        return []

    day   = date_parts.get("day",   1)
    month = date_parts.get("month", 1)
    year  = date_parts.get("year",  2000)
    base  = get_base_lang(lang)

    months = _MONTHS.get(base, _MONTHS["en"])
    month_name = months[month] if 1 <= month <= 12 else str(month)

    day_ord  = _n2w_safe(day, n2w, to="ordinal") or str(day)
    day_card = _n2w_safe(day, n2w) or str(day)
    year_forms = _year(year, n2w, lang) if year > 99 else [str(year)]

    cands: list[str] = []

    if base == "en":
        # Generate candidates with multiple year readings so the audio scorer
        # can pick "two thousand twenty-five" or "twenty twenty-five" as needed.
        yr0 = year_forms[0] if year_forms else str(year)
        # Primary "Month Day Year" format — use pair year only (year_forms[0])
        # so the model can't be confused by an alternate year reading.
        cands.append(f"{month_name} {day_ord} {yr0}")
        cands.append(f"{month_name} {day_ord}")
        cands.append(f"{month_name} {day_card}")
        # "the Nth of Month Year" format — expand with multiple year readings so
        # both "twenty twenty-five" and "two thousand twenty-five" are covered.
        for yr in year_forms[:4]:
            cands.append(f"the {day_ord} of {month_name} {yr}")
        cands.append(f"the {day_ord} of {month_name}")
        # Day-first format (common in some GT annotations)
        # Use non-hyphenated ordinal form first so it wins when scores are tied.
        day_ord_no_hyp = day_ord.replace("-", " ")
        for yr in year_forms[:4]:
            cands.append(f"{day_ord_no_hyp} {month_name} {yr}")
            cands.append(f"{day_card} {month_name} {yr}")
            cands.append(f"{day_ord} {month_name} {yr}")
    elif base == "es":
        year_spoken = year_forms[0] if year_forms else str(year)
        cands.append(f"{day_card} {month_name}")
        cands.append(f"{day_card} {month_name} {year_spoken}")
        if day_ord and day_ord != day_card:
            cands.append(f"{day_ord} {month_name}")
    elif base == "ru":
        year_spoken = year_forms[0] if year_forms else str(year)
        cands.append(f"{day_ord} {month_name}")
        cands.append(f"{day_card} {month_name} {year_spoken}")
    else:
        year_spoken = year_forms[0] if year_forms else str(year)
        cands.append(f"{day_card} {month_name}")
        cands.append(f"{month_name} {day_ord}")

    # Numeric reading: "three fifteen" (month day)
    m_spoken = _n2w_safe(month, n2w) or str(month)
    d_spoken = _n2w_safe(day,   n2w) or str(day)
    if m_spoken and d_spoken:
        cands.append(f"{m_spoken} {d_spoken}")

    return cands


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------

def _phone(raw_digits: str, n2w: Optional[str], lang: str) -> list[str]:
    if not raw_digits:
        return []

    # Primary: every digit individually
    cands = [" ".join(_digit_word(d, lang) for d in raw_digits)]

    # US 10-digit: area + prefix + two-pair suffix
    if len(raw_digits) == 10 and n2w:
        area   = " ".join(_digit_word(d, lang) for d in raw_digits[:3])
        prefix = " ".join(_digit_word(d, lang) for d in raw_digits[3:6])
        suf1 = _n2w_safe(int(raw_digits[6:8]),  n2w) or raw_digits[6:8]
        suf2 = _n2w_safe(int(raw_digits[8:10]), n2w) or raw_digits[8:10]
        cands.append(f"{area} {prefix} {suf1} {suf2}")

    # 11-digit with country code "1" (e.g. 1-800-555-0199):
    # "one eight hundred five five five zero one nine nine"
    if len(raw_digits) == 11 and raw_digits[0] == "1" and n2w:
        country  = _digit_word(raw_digits[0], lang)
        area_cardinal = _n2w_safe(int(raw_digits[1:4]), n2w) or raw_digits[1:4]
        prefix   = " ".join(_digit_word(d, lang) for d in raw_digits[4:7])
        suffix   = " ".join(_digit_word(d, lang) for d in raw_digits[7:])
        cands.append(f"{country} {area_cardinal} {prefix} {suffix}")

    return cands


# ---------------------------------------------------------------------------
# Fraction
# ---------------------------------------------------------------------------

# Spoken denominator forms (singular, plural)
_FRACTION_DENOMINATORS: dict[int, tuple[str, str]] = {
    2:   ("half",       "halves"),
    3:   ("third",      "thirds"),
    4:   ("quarter",    "quarters"),
    5:   ("fifth",      "fifths"),
    6:   ("sixth",      "sixths"),
    7:   ("seventh",    "sevenths"),
    8:   ("eighth",     "eighths"),
    9:   ("ninth",      "ninths"),
    10:  ("tenth",      "tenths"),
    12:  ("twelfth",    "twelfths"),
    16:  ("sixteenth",  "sixteenths"),
    100: ("hundredth",  "hundredths"),
}


def _fraction(fraction_parts: Optional[dict], n2w: Optional[str], lang: str) -> list[str]:
    """Spoken forms for simple fractions like 1/2, 3/4, 2/5."""
    if not fraction_parts:
        return []

    num = fraction_parts["numerator"]
    den = fraction_parts["denominator"]
    cands: list[str] = []
    num_spoken = _n2w_safe(num, n2w) or str(num)
    den_spoken_card = _n2w_safe(den, n2w) or str(den)

    # Named denominator form: "one half", "three quarters", "two fifths"
    if den in _FRACTION_DENOMINATORS:
        sing, plur = _FRACTION_DENOMINATORS[den]
        den_word = sing if num == 1 else plur
        cands.append(f"{num_spoken} {den_word}")

        # Special article forms
        if num == 1 and den == 2:
            cands.append("a half")
        elif num == 1 and den == 4:
            cands.append("a quarter")

        # "N fourths" as alternate to "N quarters"
        if den == 4 and den_word == "quarters":
            quarters_alt = "fourth" if num == 1 else "fourths"
            cands.append(f"{num_spoken} {quarters_alt}")
    else:
        # Ordinal form for other denominators via num2words
        den_ord = _n2w_safe(den, n2w, to="ordinal")
        if den_ord:
            den_word = den_ord if num == 1 else den_ord + "s"
            cands.append(f"{num_spoken} {den_word}")

    # "N over D" — always included as a fallback
    cands.append(f"{num_spoken} over {den_spoken_card}")

    return cands


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def _version(version_str: str, n2w: Optional[str], lang: str) -> list[str]:
    """Spoken form for software version strings like '2.3.1' → 'two point three point one'."""
    point = _point_word(lang)
    parts = version_str.split(".")
    spoken_parts: list[str] = []
    for part in parts:
        try:
            n = int(part)
            spoken_parts.append(_n2w_safe(n, n2w) or _digits_spoken(part, lang))
        except ValueError:
            spoken_parts.append(_digits_spoken(part, lang) or part)
    cands = [f" {point} ".join(spoken_parts)]
    return cands


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def _time(time_parts: Optional[dict], n2w: Optional[str], lang: str) -> list[str]:
    """Spoken forms for clock times like 08:05, 15:30, 10:15."""
    if not time_parts:
        return []

    hour   = time_parts["hour"]
    minute = time_parts["minute"]
    base   = get_base_lang(lang)
    cands: list[str] = []

    hour_spoken   = _n2w_safe(hour,   n2w) or _digit_word(str(hour),   lang)
    minute_spoken = _n2w_safe(minute, n2w) or _digit_word(str(minute), lang)

    # 12-hour clock values (used in several forms below)
    hour_12   = hour if hour <= 12 else hour - 12
    hour_12   = 12 if hour_12 == 0 else hour_12
    h12_spoken = _n2w_safe(hour_12, n2w) or str(hour_12)

    # ── Direct 24h reading ──────────────────────────────────────────────────
    if minute == 0:
        # "ten o'clock" / "fifteen hundred"
        cands.append(f"{hour_spoken} o'clock")
        cands.append(hour_spoken)
    elif minute < 10:
        # "eight oh five"
        zero_vars = get_zero_variants(lang)
        low_spoken = _n2w_safe(minute, n2w) or _digit_word(str(minute), lang)
        for zv in zero_vars:
            cands.append(f"{hour_spoken} {zv} {low_spoken}")
    else:
        # "fifteen thirty", "ten fifteen"
        cands.append(f"{hour_spoken} {minute_spoken}")

    # ── 12-hour with am/pm ──────────────────────────────────────────────────
    # Always generate 12-hour forms for every hour so early-morning times
    # like 03:15 produce "three fifteen am" and "three fifteen a m" candidates.
    am_pm = "am" if hour < 12 else "pm"
    am_pm_spaced = "a m" if hour < 12 else "p m"
    # Midnight (00:xx) → "twelve ..." in 12-hour; 12:xx stays "twelve ..."
    h12_ampm = 12 if hour_12 == 0 else hour_12
    h12_ampm_spoken = _n2w_safe(h12_ampm, n2w) or str(h12_ampm)
    # Spaced form first so "a m"/"p m" wins when scores are tied
    for suffix in (am_pm_spaced, am_pm):
        if minute == 0:
            cands.append(f"{h12_ampm_spoken} {suffix}")
        elif minute < 10:
            low_spoken = _n2w_safe(minute, n2w) or _digit_word(str(minute), lang)
            zero_vars = get_zero_variants(lang)
            for zv in zero_vars:
                cands.append(f"{h12_ampm_spoken} {zv} {low_spoken} {suffix}")
        else:
            cands.append(f"{h12_ampm_spoken} {minute_spoken} {suffix}")

    # ── Special verbal forms (English) ──────────────────────────────────────
    if base == "en":
        if minute == 15:
            cands.append(f"a quarter past {h12_spoken}")
            cands.append(f"quarter past {h12_spoken}")
        elif minute == 30:
            cands.append(f"half past {h12_spoken}")
            cands.append(f"{h12_spoken} thirty")
        elif minute == 45:
            next_h = hour_12 % 12 + 1
            next_spoken = _n2w_safe(next_h, n2w) or str(next_h)
            cands.append(f"a quarter to {next_spoken}")
        elif 1 <= minute <= 30:
            cands.append(f"{minute_spoken} past {h12_spoken}")
        elif 31 <= minute <= 59:
            mins_to = 60 - minute
            mins_to_spoken = _n2w_safe(mins_to, n2w) or str(mins_to)
            next_h = hour_12 % 12 + 1
            next_spoken = _n2w_safe(next_h, n2w) or str(next_h)
            cands.append(f"{mins_to_spoken} to {next_spoken}")

    # ── Digit-by-digit fallback ─────────────────────────────────────────────
    h_str = f"{hour:02d}"
    m_str = f"{minute:02d}"
    cands.extend(_digits_spoken_variants(h_str + m_str, lang))

    return cands


# ---------------------------------------------------------------------------
# Helper: per-character digit word
# ---------------------------------------------------------------------------

def _digit_word(ch: str, lang: str) -> str:
    words = get_digit_words(lang)
    try:
        return words[int(ch)]
    except (ValueError, IndexError):
        return ch


def _digits_spoken(digits: str, lang: str) -> str:
    return " ".join(_digit_word(d, lang) for d in digits if d.isdigit())


def _digits_spoken_variants(digits: str, lang: str) -> list[str]:
    """Return one digit-by-digit string per zero variant (e.g. 'zero' and 'oh' for English)."""
    zero_vars = get_zero_variants(lang)
    results = []
    for zv in zero_vars:
        spoken = " ".join(zv if d == "0" else _digit_word(d, lang) for d in digits if d.isdigit())
        results.append(spoken)
    return results


# ---------------------------------------------------------------------------
# Helper: decimal point word
# ---------------------------------------------------------------------------

_POINT_WORD: dict[str, str] = {
    "en": "point",
    "es": "coma",
    "ru": "целых",
}


def _point_word(lang: str) -> str:
    return _POINT_WORD.get(get_base_lang(lang), "point")

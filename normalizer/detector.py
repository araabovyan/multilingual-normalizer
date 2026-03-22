"""
Detect numbers in text and classify them by type.

Returns a list of NumberSpan objects ordered by position in the string.
Overlapping matches are resolved by keeping the longest (most specific) match.

Supported number types
----------------------
- CARDINAL   plain integers: 42, 1,000,000
- ORDINAL    1st, 2nd, 42nd
- DECIMAL    3.14, 1,234.56
- CURRENCY   $31, €1,000.50, R$100
- PERCENTAGE 31%, 3.14%
- DATE       3/15/2024, 15-03-2024, 2024-03-15
- PHONE      555-123-4567, (555) 123-4567, +1-800-555-1234
- YEAR       1984, 2024  (4-digit numbers in range 1000–2099)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class NumberType(str, Enum):
    CARDINAL   = "cardinal"
    ORDINAL    = "ordinal"
    DECIMAL    = "decimal"
    CURRENCY   = "currency"
    PERCENTAGE = "percentage"
    DATE       = "date"
    PHONE      = "phone"
    YEAR       = "year"
    TIME       = "time"
    FRACTION   = "fraction"
    VERSION    = "version"


@dataclass
class NumberSpan:
    """A detected number in text with position and parsed metadata."""
    start: int              # character offset (inclusive)
    end: int                # character offset (exclusive)
    text: str               # original matched text, e.g. "$31"
    integer: int            # primary integer value
    decimal: Optional[float]  # fractional part (0.14 for "3.14"), or None
    decimal_str: Optional[str]  # fractional digits as string ("14" for "3.14")
    number_type: NumberType
    currency_symbol: Optional[str] = None  # "$", "€", etc.
    date_parts: Optional[dict] = None      # {"day":15, "month":3, "year":2024}
    raw_digits: Optional[str] = None       # digits only (phone numbers)
    time_parts: Optional[dict] = None      # {"hour": 8, "minute": 5}
    fraction_parts: Optional[dict] = None  # {"numerator": 1, "denominator": 2}

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)


# ---------------------------------------------------------------------------
# Currency symbols (multi-char first for regex alternation ordering)
# ---------------------------------------------------------------------------
CURRENCY_SYMBOLS: dict[str, str] = {
    "R$": "BRL",
    "$":  "USD",
    "€":  "EUR",
    "£":  "GBP",
    "¥":  "JPY",
    "₹":  "INR",
    "₩":  "KRW",
    "₽":  "RUB",
    "₿":  "BTC",
    "฿":  "THB",
    "₫":  "VND",
    "₦":  "NGN",
    "₴":  "UAH",
    "₺":  "TRY",
    "₸":  "KZT",
    "﷼":  "SAR",
}

_CURRENCY_ALT = "|".join(re.escape(s) for s in CURRENCY_SYMBOLS)  # longest-first order

# ---------------------------------------------------------------------------
# Compiled patterns — evaluated in priority order (most specific first)
# ---------------------------------------------------------------------------
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # ── Phone ──────────────────────────────────────────────────────────────
    # Optionally include a 1- or +1- country code prefix (common US toll-free format)
    # +1-800-555-1234 | 1-800-555-0199 | (555) 123-4567 | 555-123-4567 | 555.123.4567
    ("phone", re.compile(
        r"(?<!\d)"
        r"(?:\+?\d{1,3}[-.\s])?"
        r"(?:\(\d{1,4}\)[-.\s])?"
        r"\d{3}[-.\s]\d{3,4}[-.\s]\d{4}"
        r"(?!\d)"
    )),
    # ── Date ───────────────────────────────────────────────────────────────
    # YYYY-MM-DD | MM/DD/YYYY | DD-MM-YYYY
    ("date", re.compile(
        r"(?<!\d)"
        r"(?:\d{4}[/\-]\d{1,2}[/\-]\d{1,2}"
        r"|\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})"
        r"(?!\d)"
    )),
    # ── Percentage ─────────────────────────────────────────────────────────
    ("percentage", re.compile(
        r"(?<!\w)\d+(?:[.,]\d+)?\s*%"
    )),
    # ── Currency ───────────────────────────────────────────────────────────
    ("currency", re.compile(
        r"(?:" + _CURRENCY_ALT + r")\s?\d[\d,]*(?:\.\d+)?"
    )),
    # ── Ordinal ────────────────────────────────────────────────────────────
    ("ordinal", re.compile(
        r"(?<!\w)\d+(?:st|nd|rd|th)(?!\w)", re.IGNORECASE
    )),
    # ── Decimal ────────────────────────────────────────────────────────────
    # Comma-grouped integer part, dot decimal: 1,234.56 | 3.14
    ("decimal", re.compile(
        r"(?<!\w)\d{1,3}(?:,\d{3})*\.\d+(?!\w)"
        r"|(?<!\w)\d+\.\d+(?!\w)"
    )),
    # ── Version ────────────────────────────────────────────────────────────
    # Software version strings: 2.3.1, 1.0.0.1 (two or more dot-groups).
    # Must come AFTER decimal so 3.14 stays decimal, but BEFORE cardinal
    # so the subsequent digits are not re-detected.
    ("version", re.compile(
        r"(?<!\w)\d+(?:\.\d+){2,}(?!\w)"
    )),
    # ── Fraction ───────────────────────────────────────────────────────────
    # Simple N/D fractions: 1/2, 3/4, 2/5.  Dates are detected first so
    # "3/15/2024" is never misclassified as a fraction.
    ("fraction", re.compile(
        r"(?<!\w)\d+/\d+(?!\w)"
    )),
    # ── Time ───────────────────────────────────────────────────────────────
    # HH:MM (24-hour or 12-hour): 08:05, 15:30, 10:15
    # Must appear before year/cardinal so "08:05" is not split into two cardinals.
    ("time", re.compile(
        r"(?<!\w)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)"
    )),
    # ── Year ───────────────────────────────────────────────────────────────
    # 4-digit numbers in plausible year range (kept before plain cardinal
    # so "1984" gets YEAR type, not CARDINAL)
    ("year", re.compile(
        r"(?<!\w)(?:1[0-9]{3}|20[0-9]{2})(?!\w)"
    )),
    # ── Cardinal ───────────────────────────────────────────────────────────
    # Comma-formatted large numbers first, then plain integers
    ("cardinal", re.compile(
        r"(?<!\w)\d{1,3}(?:,\d{3})+(?!\w)"   # 1,000 / 1,000,000
        r"|(?<!\w)\d+(?!\w)"                   # 42 / 007
    )),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(text: str) -> list[NumberSpan]:
    """
    Detect all number spans in *text*, ordered by position.

    Overlapping matches are resolved greedily: if two patterns overlap,
    the earlier-starting (and, on ties, longer) match is kept.
    """
    raw: list[tuple[int, int, NumberSpan]] = []

    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            span = _build_span(kind, m)
            if span is not None:
                raw.append((m.start(), m.end(), span))

    # Sort by start position; on ties prefer longer match (more specific)
    raw.sort(key=lambda t: (t[0], -(t[1] - t[0])))

    result: list[NumberSpan] = []
    occupied_end = -1
    for start, end, span in raw:
        if start >= occupied_end:
            result.append(span)
            occupied_end = end

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_int_decimal(s: str) -> tuple[int, Optional[float], Optional[str]]:
    """Parse a digit string (commas as thousands sep, dot as decimal) into
    (integer, decimal_float, decimal_str).  Returns decimal_str = None when
    there is no decimal component.
    """
    clean = s.replace(",", "")
    if "." in clean:
        int_part, dec_part = clean.split(".", 1)
        return int(int_part), float("0." + dec_part), dec_part
    return int(clean), None, None


def _build_span(kind: str, m: re.Match) -> Optional[NumberSpan]:
    raw = m.group()
    start, end = m.start(), m.end()

    # ── Phone ──────────────────────────────────────────────────────────────
    if kind == "phone":
        digits = re.sub(r"\D", "", raw)
        return NumberSpan(
            start=start, end=end, text=raw,
            integer=int(digits) if digits else 0,
            decimal=None, decimal_str=None,
            number_type=NumberType.PHONE,
            raw_digits=digits,
        )

    # ── Date ───────────────────────────────────────────────────────────────
    if kind == "date":
        parts = re.split(r"[/\-]", raw)
        if len(parts) != 3:
            return None
        p = [int(x) for x in parts]
        if len(parts[0]) == 4:
            # ISO format: YYYY-MM-DD
            date_parts = {"year": p[0], "month": p[1], "day": p[2]}
        elif p[0] > 12:
            # Day-first (European): DD-MM-YYYY or DD/MM/YYYY
            year = p[2] if p[2] > 99 else (2000 + p[2])
            date_parts = {"day": p[0], "month": p[1], "year": year}
        else:
            # Month-first (US): MM/DD/YYYY or MM-DD-YYYY
            year = p[2] if p[2] > 99 else (2000 + p[2])
            date_parts = {"month": p[0], "day": p[1], "year": year}
        return NumberSpan(
            start=start, end=end, text=raw,
            integer=date_parts["year"],
            decimal=None, decimal_str=None,
            number_type=NumberType.DATE,
            date_parts=date_parts,
        )

    # ── Percentage ─────────────────────────────────────────────────────────
    if kind == "percentage":
        num_str = raw.rstrip("% \t").replace(",", ".")
        integer, dec, dec_str = _parse_int_decimal(num_str)
        return NumberSpan(
            start=start, end=end, text=raw,
            integer=integer, decimal=dec, decimal_str=dec_str,
            number_type=NumberType.PERCENTAGE,
        )

    # ── Currency ───────────────────────────────────────────────────────────
    if kind == "currency":
        sym = None
        for s in CURRENCY_SYMBOLS:
            if raw.startswith(s):
                sym = s
                break
        if sym is None:
            return None
        num_str = raw[len(sym):].strip()
        integer, dec, dec_str = _parse_int_decimal(num_str)
        return NumberSpan(
            start=start, end=end, text=raw,
            integer=integer, decimal=dec, decimal_str=dec_str,
            number_type=NumberType.CURRENCY,
            currency_symbol=sym,
        )

    # ── Ordinal ────────────────────────────────────────────────────────────
    if kind == "ordinal":
        digits_match = re.match(r"\d+", raw)
        if not digits_match:
            return None
        return NumberSpan(
            start=start, end=end, text=raw,
            integer=int(digits_match.group()),
            decimal=None, decimal_str=None,
            number_type=NumberType.ORDINAL,
        )

    # ── Decimal ────────────────────────────────────────────────────────────
    if kind == "decimal":
        integer, dec, dec_str = _parse_int_decimal(raw)
        return NumberSpan(
            start=start, end=end, text=raw,
            integer=integer, decimal=dec, decimal_str=dec_str,
            number_type=NumberType.DECIMAL,
        )

    # ── Year ───────────────────────────────────────────────────────────────
    if kind == "year":
        return NumberSpan(
            start=start, end=end, text=raw,
            integer=int(raw), decimal=None, decimal_str=None,
            number_type=NumberType.YEAR,
        )

    # ── Time ───────────────────────────────────────────────────────────────
    if kind == "time":
        h_str, m_str = raw.split(":")
        hour = int(h_str)
        minute = int(m_str)
        return NumberSpan(
            start=start, end=end, text=raw,
            integer=hour, decimal=None, decimal_str=None,
            number_type=NumberType.TIME,
            time_parts={"hour": hour, "minute": minute},
        )

    # ── Cardinal ───────────────────────────────────────────────────────────
    if kind == "cardinal":
        clean = raw.replace(",", "")
        return NumberSpan(
            start=start, end=end, text=raw,
            integer=int(clean), decimal=None, decimal_str=None,
            number_type=NumberType.CARDINAL,
        )

    # ── Fraction ───────────────────────────────────────────────────────────
    if kind == "fraction":
        num_str, den_str = raw.split("/", 1)
        numerator = int(num_str)
        denominator = int(den_str)
        return NumberSpan(
            start=start, end=end, text=raw,
            integer=numerator, decimal=None, decimal_str=None,
            number_type=NumberType.FRACTION,
            fraction_parts={"numerator": numerator, "denominator": denominator},
        )

    # ── Version ────────────────────────────────────────────────────────────
    if kind == "version":
        # Store the dot-separated parts as the raw_digits field (as a joined string)
        # and parse the first component as the primary integer.
        parts = raw.split(".")
        try:
            return NumberSpan(
                start=start, end=end, text=raw,
                integer=int(parts[0]), decimal=None, decimal_str=None,
                number_type=NumberType.VERSION,
                raw_digits=raw,  # store full version string for candidate generation
            )
        except ValueError:
            return None

    return None

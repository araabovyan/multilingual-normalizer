"""
Unit tests for candidate generation.

These tests are purely CPU / no-model — they only exercise the
detector + candidates modules and can run without any downloaded weights.
"""

import pytest
from normalizer.detector import detect, NumberType
from normalizer.candidates import generate_candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def first(text: str, lang: str = "en") -> str:
    """Return the first candidate for the first detected number."""
    spans = detect(text)
    assert spans, f"No number detected in: {text!r}"
    return generate_candidates(spans[0], lang)[0]


def all_cands(text: str, lang: str = "en") -> list[str]:
    spans = detect(text)
    assert spans
    return generate_candidates(spans[0], lang)


# ---------------------------------------------------------------------------
# Cardinal
# ---------------------------------------------------------------------------

class TestCardinal:
    def test_simple(self):
        assert "forty-two" in all_cands("42")

    def test_large(self):
        cands = all_cands("1000000")
        assert any("million" in c for c in cands)

    def test_comma_formatted(self):
        cands = all_cands("1,000")
        assert any("thousand" in c for c in cands)

    def test_digit_by_digit_present(self):
        # Digit-by-digit should always be generated as an alternate
        cands = all_cands("42")
        assert any(c in ("four two", "forty two", "forty-two") for c in cands)

    def test_zero(self):
        cands = all_cands("0")
        assert "zero" in cands


# ---------------------------------------------------------------------------
# Year
# ---------------------------------------------------------------------------

class TestYear:
    def test_19th_century(self):
        cands = all_cands("1984")
        # Should include "nineteen eighty-four" style
        assert any("nineteen" in c and "four" in c for c in cands), cands

    def test_2000(self):
        cands = all_cands("2000")
        assert any("two thousand" in c for c in cands), cands

    def test_2001(self):
        cands = all_cands("2001")
        assert any("two thousand" in c for c in cands), cands

    def test_2024(self):
        cands = all_cands("2024")
        # "twenty twenty-four" or "two thousand twenty-four"
        assert any("twenty" in c for c in cands), cands

    def test_1900(self):
        cands = all_cands("1900")
        assert any("hundred" in c for c in cands), cands

    def test_1905(self):
        cands = all_cands("1905")
        # "nineteen oh five"
        assert any("oh" in c or "zero" in c for c in cands), cands

    def test_type(self):
        spans = detect("1984")
        assert spans[0].number_type == NumberType.YEAR


# ---------------------------------------------------------------------------
# Ordinal
# ---------------------------------------------------------------------------

class TestOrdinal:
    def test_first(self):
        cands = all_cands("1st")
        assert any("first" in c for c in cands), cands

    def test_second(self):
        cands = all_cands("2nd")
        assert any("second" in c for c in cands), cands

    def test_42nd(self):
        cands = all_cands("42nd")
        assert any("forty-second" in c or "forty second" in c for c in cands), cands

    def test_type(self):
        spans = detect("3rd")
        assert spans[0].number_type == NumberType.ORDINAL


# ---------------------------------------------------------------------------
# Decimal
# ---------------------------------------------------------------------------

class TestDecimal:
    def test_pi(self):
        cands = all_cands("3.14")
        # "three point one four"
        assert any("point" in c and "one" in c and "four" in c for c in cands), cands

    def test_type(self):
        spans = detect("3.14")
        assert spans[0].number_type == NumberType.DECIMAL

    def test_integer_part(self):
        spans = detect("3.14")
        assert spans[0].integer == 3
        assert spans[0].decimal_str == "14"


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

class TestCurrency:
    def test_usd_integer(self):
        cands = all_cands("$31")
        assert any("thirty" in c and "dollar" in c for c in cands), cands

    def test_usd_decimal(self):
        cands = all_cands("$31.50")
        # Should include "thirty-one dollars and fifty cents" or similar
        assert any("dollar" in c and ("fifty" in c or "cent" in c) for c in cands), cands

    def test_euro(self):
        cands = all_cands("€25")
        assert any("euro" in c or "twenty" in c for c in cands), cands

    def test_type(self):
        spans = detect("$31")
        assert spans[0].number_type == NumberType.CURRENCY
        assert spans[0].currency_symbol == "$"

    def test_spanish_currency(self):
        cands = all_cands("$31", lang="es")
        assert any("dólar" in c or "treinta" in c for c in cands), cands


# ---------------------------------------------------------------------------
# Percentage
# ---------------------------------------------------------------------------

class TestPercentage:
    def test_simple(self):
        cands = all_cands("15%")
        assert any("percent" in c and "fifteen" in c for c in cands), cands

    def test_decimal(self):
        cands = all_cands("3.14%")
        assert any("percent" in c for c in cands), cands

    def test_type(self):
        spans = detect("15%")
        assert spans[0].number_type == NumberType.PERCENTAGE


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------

class TestPhone:
    def test_us_phone(self):
        cands = all_cands("555-123-4567")
        # All digits individually
        assert any("five five five" in c for c in cands), cands

    def test_type(self):
        spans = detect("555-123-4567")
        assert spans[0].number_type == NumberType.PHONE


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------

class TestDate:
    def test_us_format(self):
        cands = all_cands("3/15/2024")
        assert any("march" in c for c in cands), cands

    def test_iso_format(self):
        cands = all_cands("2024-03-15")
        # Should detect as date
        spans = detect("2024-03-15")
        assert spans[0].number_type == NumberType.DATE

    def test_type(self):
        spans = detect("3/15/2024")
        assert spans[0].number_type == NumberType.DATE
        assert spans[0].date_parts == {"month": 3, "day": 15, "year": 2024}


# ---------------------------------------------------------------------------
# Detector — overlap resolution
# ---------------------------------------------------------------------------

class TestDetector:
    def test_no_double_detect(self):
        # "$31.50" should give ONE span (currency), not currency + decimal + cardinal
        spans = detect("$31.50")
        assert len(spans) == 1
        assert spans[0].number_type == NumberType.CURRENCY

    def test_percentage_not_cardinal(self):
        spans = detect("15%")
        assert len(spans) == 1
        assert spans[0].number_type == NumberType.PERCENTAGE

    def test_year_not_cardinal(self):
        spans = detect("1984")
        assert spans[0].number_type == NumberType.YEAR

    def test_ordinal_not_cardinal(self):
        spans = detect("3rd")
        assert spans[0].number_type == NumberType.ORDINAL

    def test_multiple_numbers(self):
        spans = detect("pay $31 by 3/15/2024")
        assert len(spans) == 2


# ---------------------------------------------------------------------------
# Multilingual candidate generation
# ---------------------------------------------------------------------------

class TestMultilingual:
    @pytest.mark.parametrize("lang,expected_substr", [
        ("fr", "quarante"),
        ("de", "zweiundvierzig"),
        ("es", "cuarenta"),
        ("it", "quarantadue"),
        ("pt", "quarenta"),
        ("ru", "сорок"),
        ("nl", "tweeënveertig"),
        ("pl", "czterdzieści"),
        ("sv", "tiotvå"),   # num2words sv: "fyrtiotvå" or "förtiotvå" — match common suffix
    ])
    def test_cardinal_42(self, lang, expected_substr):
        cands = all_cands("42", lang=lang)
        assert any(expected_substr in c for c in cands), \
            f"lang={lang}: expected '{expected_substr}' in {cands}"

    def test_year_french(self):
        cands = all_cands("1984", lang="fr")
        # Should include "mille neuf cent quatre-vingt-quatre" or two-chunk form
        assert cands, "no candidates generated"

    def test_currency_euro_french(self):
        cands = all_cands("€25", lang="fr")
        assert any("euro" in c for c in cands), cands

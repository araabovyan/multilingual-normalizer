"""
Pipeline integration tests (text-only mode — no model required).
"""

import numpy as np
import pytest
from normalizer import NumberNormalizer


@pytest.fixture(scope="module")
def nn():
    return NumberNormalizer()


# ---------------------------------------------------------------------------
# Text-only normalization
# ---------------------------------------------------------------------------

class TestTextOnly:
    def test_simple_cardinal(self, nn):
        out = nn.normalize_text_only("there are 42 items", lang="en")
        assert "forty" in out.lower()
        assert "42" not in out

    def test_currency_usd(self, nn):
        out = nn.normalize_text_only("pay $31 now", lang="en")
        assert "thirty" in out.lower()
        assert "dollar" in out.lower()
        assert "$31" not in out

    def test_year(self, nn):
        out = nn.normalize_text_only("born in 1984", lang="en")
        assert "nineteen" in out.lower() or "eighty" in out.lower()
        assert "1984" not in out

    def test_decimal(self, nn):
        out = nn.normalize_text_only("pi is 3.14", lang="en")
        assert "point" in out.lower() or "three" in out.lower()
        assert "3.14" not in out

    def test_percentage(self, nn):
        out = nn.normalize_text_only("growth is 15%", lang="en")
        assert "fifteen" in out.lower()
        assert "percent" in out.lower()

    def test_no_numbers(self, nn):
        text = "hello world"
        out = nn.normalize_text_only(text, lang="en")
        assert out == text

    def test_multiple_numbers(self, nn):
        out = nn.normalize_text_only("pay $31 with a 10% discount", lang="en")
        assert "$31" not in out
        assert "10%" not in out

    def test_preserves_context(self, nn):
        result = nn.normalize(None, "pay $31 today", lang="en")
        assert "pay" in result.normalized_text
        assert "today" in result.normalized_text

    def test_preserves_context_words(self, nn):
        result = nn.normalize(None, "I have 3 cats and 2 dogs", lang="en")
        assert "cats" in result.normalized_text
        assert "dogs" in result.normalized_text
        assert "3" not in result.normalized_text
        assert "2" not in result.normalized_text


# ---------------------------------------------------------------------------
# Multilingual text-only
# ---------------------------------------------------------------------------

class TestTextOnlyMultilingual:
    @pytest.mark.parametrize("lang,text,must_contain,must_not_contain", [
        ("fr", "il y a 42 pommes", "quarante", "42"),
        ("de", "es sind 42 Äpfel", "zweiundvierzig", "42"),
        ("es", "hay 42 manzanas",  "cuarenta",       "42"),
        ("ru", "у меня 42 яблока", "сорок",           "42"),
    ])
    def test_cardinal(self, nn, lang, text, must_contain, must_not_contain):
        out = nn.normalize_text_only(text, lang=lang)
        assert must_contain in out.lower(), f"{lang}: expected '{must_contain}' in {out!r}"
        assert must_not_contain not in out, f"{lang}: expected no {must_not_contain!r} in {out!r}"


# ---------------------------------------------------------------------------
# NormalizationResult structure
# ---------------------------------------------------------------------------

class TestResult:
    def test_fields(self, nn):
        result = nn.normalize(None, "pay $31", lang="en")
        assert result.original_text == "pay $31"
        assert "$31" not in result.normalized_text
        assert result.audio_used is False
        assert "$31" in result.chosen_forms
        assert "$31" in result.all_scores

    def test_spans(self, nn):
        result = nn.normalize(None, "born in 1984, pay $31", lang="en")
        assert len(result.spans) == 2

    def test_batch(self, nn):
        pairs = [
            (None, "I have 3 cats"),
            (None, "born in 1984"),
        ]
        results = nn.batch_normalize(pairs, lang="en")
        assert len(results) == 2
        assert "3" not in results[0].normalized_text
        assert "1984" not in results[1].normalized_text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_number_at_start(self, nn):
        out = nn.normalize_text_only("42 is the answer", lang="en")
        assert "forty" in out.lower()
        assert out.endswith("is the answer")

    def test_number_at_end(self, nn):
        out = nn.normalize_text_only("the answer is 42", lang="en")
        assert "forty" in out.lower()

    def test_large_number(self, nn):
        out = nn.normalize_text_only("population: 1000000", lang="en")
        assert "1000000" not in out
        assert "million" in out.lower()

    def test_phone(self, nn):
        out = nn.normalize_text_only("call 555-123-4567", lang="en")
        assert "555-123-4567" not in out
        assert "five" in out.lower()

    def test_ordinal(self, nn):
        out = nn.normalize_text_only("finished 1st place", lang="en")
        assert "first" in out.lower()
        assert "1st" not in out

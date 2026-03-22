"""
Print all candidates and their CTC scores for every entry in a .jsonl file.

Usage:
    python scripts/score_candidates.py [sample.jsonl] [--lang en]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from normalizer.candidates import generate_candidates
from normalizer.detector import detect
from normalizer.language_utils import get_mms_lang
from normalizer.pipeline import _build_full_sentence
from normalizer.scorer import CTCScorer


def load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


def print_scores(
    span_text: str,
    span_type: str,
    scores: dict[str, float],
    gold_fragment: str | None = None,
) -> None:
    best = max(scores, key=scores.get)
    col = 48

    print(f"  Span: {span_text!r}  ({span_type})")
    print(f"  {'Candidate':<{col}}  {'Score':>8}  Notes")
    print(f"  {'-' * col}  {'--------'}  -----")

    for cand, score in sorted(scores.items(), key=lambda x: -x[1]):
        notes = []
        if cand == best:
            notes.append("best")
        if gold_fragment and cand.lower().strip() == gold_fragment.lower().strip():
            notes.append("gold")
        score_str = f"{score:>8.3f}" if score > float("-inf") else "    -inf"
        print(f"  {cand:<{col}}  {score_str}  {', '.join(notes)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates for each sample.")
    parser.add_argument("jsonl", nargs="?", default="sample.jsonl")
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio scoring")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        sys.exit(f"File not found: {jsonl_path}")

    entries = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    lang = args.lang
    lang_mms = get_mms_lang(lang)

    scorer = CTCScorer() if (not args.no_audio and lang_mms) else None

    for idx, entry in enumerate(entries, 1):
        text = entry["input"]
        gold = entry.get("gold", "")
        audio_path = entry.get("audio", "")

        print("=" * 70)
        print(f"Sample {idx}")
        print(f"  Input : {text}")
        print(f"  Gold  : {gold}")
        print(f"  Audio : {audio_path}")
        print()

        audio, sr = None, 16000
        if scorer and audio_path:
            try:
                audio, sr = load_audio(audio_path)
            except Exception as e:
                print(f"  [warn] Could not load audio: {e}")

        spans = detect(text)
        if not spans:
            print("  No number spans detected.\n")
            continue

        # Pre-compute candidates and default (first) form for every span.
        all_candidates: dict[str, list[str]] = {
            s.text: generate_candidates(s, lang) for s in spans
        }
        default_forms: dict[str, str] = {
            s.text: all_candidates[s.text][0] for s in spans
        }

        for span in spans:
            candidates = all_candidates[span.text]

            if scorer and audio is not None and len(candidates) > 1:
                # Build full-sentence variants: target span → candidate,
                # all other spans → their default spoken form.
                full_texts = [
                    _build_full_sentence(text, spans, span, c, default_forms)
                    for c in candidates
                ]
                raw_scores = scorer.score_candidates(audio, full_texts, lang_mms, sr)
                # Remap {full_sentence: score} → {fragment: score}
                scores = {c: raw_scores[full_texts[i]] for i, c in enumerate(candidates)}
            else:
                scores = {c: 0.0 for c in candidates}

            # Try to identify which gold token corresponds to this span
            gold_fragment = None
            if gold:
                # Rough heuristic: grab the gold tokens at the same character position
                pre_text = text[: span.start]
                word_idx = len(pre_text.split())
                gold_words = gold.split()
                n_words = len(span.text.split())
                if word_idx < len(gold_words):
                    gold_fragment = " ".join(gold_words[word_idx : word_idx + max(n_words, 3)])

            print_scores(span.text, span.number_type.value, scores, gold_fragment)

        print()


if __name__ == "__main__":
    main()

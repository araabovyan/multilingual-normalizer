"""
Command-line interface for the number normalizer.

Usage
-----
    # Text-only
    normalize-numbers --text "pay $31 by 3/15" --lang en

    # Audio-guided
    normalize-numbers --audio speech.wav --text "pay $31 by 3/15" --lang en

    # Verbose (show candidates and CTC scores)
    normalize-numbers --audio speech.wav --text "pay $31 by 3/15" --lang en --verbose
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="normalize-numbers",
        description="Audio-guided multilingual number normalizer",
    )
    parser.add_argument("--text",    required=True,  help="Input text to normalize")
    parser.add_argument("--lang",    required=True,  help="Language code (en, fr, de, ...)")
    parser.add_argument("--audio",   default=None,   help="Path to audio file (WAV/FLAC/MP3)")
    parser.add_argument("--model",   default="facebook/mms-1b-all",
                        help="HuggingFace model ID for CTC scoring")
    parser.add_argument("--verbose", action="store_true",
                        help="Print candidates and CTC scores")
    args = parser.parse_args()

    # Lazy imports so --help is instant
    import numpy as np
    from normalizer import NumberNormalizer

    nn = NumberNormalizer(model_name=args.model)

    audio_np = None
    sr       = 16000

    if args.audio:
        try:
            import soundfile as sf
            audio_np, sr = sf.read(args.audio, dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
        except Exception as exc:
            print(f"[error] Could not load audio: {exc}", file=sys.stderr)
            sys.exit(1)

    result = nn.normalize(audio_np, args.text, lang=args.lang, sample_rate=sr)

    print(result.normalized_text)

    if args.verbose:
        print()
        for original, scores in result.all_scores.items():
            chosen = result.chosen_forms.get(original, "")
            print(f"  {original!r} → {chosen!r}")
            if result.audio_used:
                sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for cand, score in sorted_items:
                    marker = "✓" if cand == chosen else " "
                    score_str = f"{score:+.3f}" if score > float("-inf") else "  -inf"
                    print(f"    {marker} {score_str}  {cand}")
            print()

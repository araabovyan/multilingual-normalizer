#!/usr/bin/env python3
"""
Evaluate the NumberNormalizer pipeline on data/samples_with_audio.jsonl.

Each sample is run through the audio-guided pipeline and the output is compared
to the ground_truth field.  Two metrics are reported:

  Exact-match accuracy (primary)  — did the pipeline pick the correct spoken form?
  Word Error Rate      (secondary) — how many words differ on misses?

A JSONL detail file is written alongside the evaluation (--out flag, default:
eval_details.jsonl next to the data file).  Each line contains the full
candidate list and CTC score for every detected number span in that sample.

Usage
-----
    python scripts/eval_on_samples.py
    python scripts/eval_on_samples.py --backend nemo
    python scripts/eval_on_samples.py --backend nemo --nemo-cache-dir ./nemo_cache
    python scripts/eval_on_samples.py [--data PATH] [--audio-dir DIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from normalizer.pipeline import NumberNormalizer  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "samples_with_audio.jsonl"
_DEFAULT_NEMO_CACHE = str(ROOT / "nemo_cache")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_audio(path: Path, target_sr: int = 16_000) -> np.ndarray:
    """Return float32 mono array resampled to target_sr."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            import torchaudio
            import torch
            t = torch.from_numpy(audio).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, target_sr)
            audio = t.squeeze(0).numpy()
    return audio


def wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using edit distance."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    n = len(ref)
    if n == 0:
        return 0.0
    # DP table: d[i][j] = edit distance for ref[:i], hyp[:j]
    d = list(range(len(hyp) + 1))
    for i in range(1, n + 1):
        prev = d[:]
        d[0] = i
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + cost)
    return d[len(hyp)] / n


def normalise(text: str) -> str:
    """Case-fold and collapse whitespace for comparison."""
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normalizer on labeled samples.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Path to the .jsonl evaluation file (default: data/samples_with_audio.jsonl)",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(ROOT),
        help="Root directory that audio_filepath values are relative to (default: project root)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Path for the per-sample JSONL detail file.  "
            "Defaults to eval_details.jsonl next to the --data file."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["num2words", "nemo"],
        default="nemo",
        help="Candidate generation backend (default: nemo)",
    )
    parser.add_argument(
        "--nemo-cache-dir",
        default=_DEFAULT_NEMO_CACHE,
        metavar="DIR",
        help=(
            "Directory for NeMo compiled grammar cache "
            f"(default: {_DEFAULT_NEMO_CACHE}). "
            "Re-using the cache avoids ~8 min grammar compilation on first run."
        ),
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    audio_root = Path(args.audio_dir)
    out_path = Path(args.out) if args.out else data_path.parent / "eval_details.jsonl"

    samples: list[dict] = []
    with open(data_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {data_path.name}")
    print(f"Backend        : {args.backend}")
    if args.backend == "nemo":
        print(f"NeMo cache dir : {args.nemo_cache_dir}")
    print("Loading model (this may take a moment)...\n")

    normalizer = NumberNormalizer(
        lazy_load=True,
        candidate_backend=args.backend,
        nemo_cache_dir=args.nemo_cache_dir,
    )

    results: list[dict] = []
    per_class: dict[str, list[dict]] = defaultdict(list)

    out_fh = open(out_path, "w")
    print(f"Writing per-sample details to {out_path}\n")

    for i, sample in enumerate(samples):
        sentence: str = sample["sentence"]
        ground_truth: str = sample["ground_truth"]
        cls: str = sample["class"]
        audio_path = audio_root / sample["audio_filepath"]

        prediction = sentence  # fallback: no-op
        error_msg: str | None = None
        all_scores: dict[str, dict[str, float]] = {}
        chosen_forms: dict[str, str] = {}

        try:
            audio = load_audio(audio_path)
            result = normalizer.normalize(audio=audio, text=sentence, lang="en")
            prediction = result.normalized_text
            all_scores = result.all_scores
            chosen_forms = result.chosen_forms
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)

        exact = normalise(prediction) == normalise(ground_truth)
        word_err = wer(ground_truth, prediction)

        record = {
            "idx": i,
            "class": cls,
            "sentence": sentence,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "exact_match": exact,
            "wer": word_err,
            "error": error_msg,
        }
        results.append(record)
        per_class[cls].append(record)

        # Build per-span candidate detail, sorted best→worst by CTC score.
        spans_detail = []
        for token, scores in all_scores.items():
            chosen = chosen_forms.get(token, "")
            sorted_candidates = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            spans_detail.append(
                {
                    "token": token,
                    "chosen": chosen,
                    "candidates": [
                        {
                            "text": cand,
                            "score": round(score, 6),
                            "is_chosen": cand == chosen,
                        }
                        for cand, score in sorted_candidates
                    ],
                }
            )

        detail_record = {**record, "spans": spans_detail}
        out_fh.write(json.dumps(detail_record, ensure_ascii=False) + "\n")
        out_fh.flush()

        tick = "PASS" if exact else "FAIL"
        print(f"[{i:02d}] {tick}  ({cls})")
        print(f"      Input : {sentence}")
        print(f"      Pred  : {prediction}")
        print(f"      Truth : {ground_truth}")
        if not exact:
            print(f"      WER   : {word_err:.3f}")
        if error_msg:
            print(f"      ERROR : {error_msg}")
        print()

    out_fh.close()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n = len(results)
    n_correct = sum(r["exact_match"] for r in results)
    overall_acc = n_correct / n if n else 0.0
    overall_wer = sum(r["wer"] for r in results) / n if n else 0.0

    sep = "=" * 60
    print(sep)
    print("RESULTS SUMMARY")
    print(sep)
    print(f"  Samples         : {n}")
    print(f"  Exact-match acc : {n_correct}/{n} = {overall_acc:.1%}")
    print(f"  Avg WER         : {overall_wer:.3f}")
    print()
    print("  Per-class breakdown:")
    col_w = max(len(c) for c in per_class) if per_class else 8
    for cls in sorted(per_class):
        cls_results = per_class[cls]
        m = len(cls_results)
        c = sum(r["exact_match"] for r in cls_results)
        a = c / m
        w = sum(r["wer"] for r in cls_results) / m
        print(f"    {cls:<{col_w}}  {c}/{m}  acc={a:.1%}  avg_wer={w:.3f}")
    print()


if __name__ == "__main__":
    main()

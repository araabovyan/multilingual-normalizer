"""
Gradio demo — Multilingual Number Normalizer
============================================

Run:
    uv run python demo/app.py
    uv run python demo/app.py --backend nemo
    uv run python demo/app.py --backend nemo --nemo-cache-dir ./nemo_cache

Options:
    --backend          Candidate backend: 'nemo' (default) or 'num2words'
    --nemo-cache-dir   Directory for NeMo compiled grammar cache.
                       Default: ./nemo_cache (pre-compiled cache in the repo root)
                       Re-using the cache avoids ~8 min grammar compilation.

Opens at http://localhost:7860
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Make sure the package is importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gradio as gr

from normalizer import NumberNormalizer
from normalizer.detector import detect
from normalizer.candidates import generate_candidates

# ---------------------------------------------------------------------------
# Globals set by parse_args() before the UI is built
# ---------------------------------------------------------------------------
_normalizer: NumberNormalizer
_backend: str
_nemo_cache_dir: str

# ---------------------------------------------------------------------------
# Language menu
# ---------------------------------------------------------------------------
LANGUAGES: dict[str, str] = {
    "English": "en",
    "Spanish": "es",
    "Russian": "ru",
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run(audio_input, text: str, language: str):
    lang = LANGUAGES.get(language, "en")

    # ── Detection preview ─────────────────────────────────────────────
    spans = detect(text)
    if not spans:
        detection_md = "_No numbers detected in the input text._"
    else:
        lines = []
        for s in spans:
            cands = generate_candidates(
                s, lang,
                backend=_backend,
                nemo_cache_dir=_nemo_cache_dir,
            )
            top5  = ", ".join(f'`{c}`' for c in cands[:5])
            lines.append(f"**`{s.text}`** ({s.number_type.value}) → {top5}")
        detection_md = "\n\n".join(lines)

    # ── Normalization ─────────────────────────────────────────────────
    audio_np: np.ndarray | None = None
    sr = 16000

    if audio_input is not None:
        sr, raw = audio_input
        audio_np = raw.astype(np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        # Gradio may deliver int16 PCM
        if audio_np.max() > 1.5:
            audio_np /= 32768.0

    result = _normalizer.normalize(audio_np, text, lang=lang, sample_rate=sr)
    

    # ── Score display ─────────────────────────────────────────────────
    scores_md = ""
    for original, candidate_scores in result.all_scores.items():
        chosen = result.chosen_forms.get(original, "")
        scores_md += f"### `{original}` → **`{chosen}`**\n"
        if result.audio_used and any(v != 0.0 for v in candidate_scores.values()):
            sorted_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            lines = []
            for i, (cand, score) in enumerate(sorted_items):
                marker = "✓" if i == 0 else " "
                if score > float("-inf"):
                    lines.append(f"`{marker}` `{score:+.3f}` → {cand}")
                else:
                    lines.append(f"`  ` `   -inf ` → {cand}")
            scores_md += "\n".join(lines) + "\n\n"
        else:
            scores_md += "_Text-only mode — first candidate used._\n\n"

    if not scores_md:
        scores_md = "_No numbers to normalize._"

    return result.normalized_text, detection_md, scores_md


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_EXAMPLES = [
    [None, 'the surcharge will be $31',                        "English"],
    [None, 'call us at 555-123-4567',                          "English"],
    [None, 'he was born in 1984',                              "English"],
    [None, 'pi is approximately 3.14',                         "English"],
    [None, 'the discount is 15%',                              "English"],
    [None, 'the meeting is on 3/15/2024',                      "English"],
    [None, 'el precio es €25,50',                              "Spanish"],
    [None, 'родился в 1984 году',                              "Russian"],
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Multilingual Number Normalizer") as demo:

        gr.Markdown("""
# Multilingual Speech Number Normalizer

Convert **written numbers** in text to their **spoken form**, optionally guided by audio.

| Mode | How |
|------|-----|
| **Audio-guided** | Upload audio → CTC scoring with [MMS](https://huggingface.co/facebook/mms-1b-all) selects the right verbalization |
| **Text-only** | Leave audio empty → default `num2words` form is used (no model needed) |
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    label="Audio (optional — enables audio-guided selection)",
                    type="numpy",
                    buttons=["download"],
                )
                text_in = gr.Textbox(
                    label="Input Text",
                    placeholder='e.g.  "the surcharge will be $31"',
                    lines=3,
                )
                lang_in = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="Language",
                )
                run_btn = gr.Button("Normalize", variant="primary")

            with gr.Column(scale=1):
                out_text = gr.Textbox(
                    label="Normalized Text",
                    lines=3,
                    interactive=False,
                )
                out_detect = gr.Markdown(label="Detected numbers & candidates")
                out_scores = gr.Markdown(label="CTC Scores")

        gr.Examples(
            examples=_EXAMPLES,
            inputs=[audio_in, text_in, lang_in],
            label="Examples (text-only mode)",
        )

        run_btn.click(
            fn=run,
            inputs=[audio_in, text_in, lang_in],
            outputs=[out_text, out_detect, out_scores],
        )

    return demo


if __name__ == "__main__":
    _DEFAULT_NEMO_CACHE = str(
        Path(__file__).resolve().parent.parent / "nemo_cache"
    )

    parser = argparse.ArgumentParser(
        description="Multilingual Number Normalizer — Gradio demo"
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
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)"
    )
    args = parser.parse_args()

    _backend = args.backend
    _nemo_cache_dir = args.nemo_cache_dir
    _normalizer = NumberNormalizer(
        candidate_backend=_backend,
        nemo_cache_dir=_nemo_cache_dir,
    )

    print(f"Backend        : {_backend}")
    if _backend == "nemo":
        print(f"NeMo cache dir : {_nemo_cache_dir}")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False, theme=gr.themes.Soft())

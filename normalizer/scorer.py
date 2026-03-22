"""
CTC-based candidate scorer using Meta's MMS model.

For each candidate spoken form, computes the CTC log-likelihood
P(candidate | audio) using the MMS encoder.  The candidate with the
highest score is the most likely spoken form in the audio.

Design
------
- Model: facebook/mms-1b-all (1,107 languages via per-language adapters)
- Fallback: facebook/wav2vec2-large-xlsr-53 (53 languages, smaller)
- Model weights are loaded lazily and cached between calls
- Language adapters are hot-swapped via model.load_adapter()

CTC scoring
-----------
    logits  = encoder(audio)                  # [1, T, V]
    log_p   = log_softmax(logits, dim=-1)     # [1, T, V]
    score   = -ctc_loss(log_p, tokens)        # higher = better

We use reduction='mean' so scores are normalised by target length and
thus comparable across candidates of different lengths.

Resample
--------
Audio is always resampled to 16 kHz before passing to the model.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from transformers import Wav2Vec2ForCTC, AutoProcessor

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "facebook/mms-1b-all"


class CTCScorer:
    """
    Scores text candidates against audio using CTC log-likelihood.

    Runs on CUDA if available; falls back to CPU otherwise.
    Uses FP16 on CUDA for ~2× throughput and half memory.

    Thread-safety: NOT thread-safe (one scorer per process/thread recommended).

    Example::

        scorer = CTCScorer()
        scores = scorer.score_candidates(audio_16k, ["thirty-one", "thirty one"], "eng")
        best   = max(scores, key=scores.get)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info(
                "CUDA available: %s  (VRAM %.1f GB)",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )
        else:
            self.device = "cpu"
            logger.warning("CUDA not available — running on CPU (will be slow).")

        # Use FP16 on CUDA for faster inference; FP32 on CPU (FP16 on CPU is slow)
        self._dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        self._model = None
        self._processor = None
        self._current_lang: Optional[str] = None
        self._is_mms = "mms" in model_name.lower()

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self, lang_mms: str) -> bool:
        """Load model weights and switch language adapter. Returns True on success."""
        try:
            # Load model once — directly onto the target device in the right dtype
            if self._model is None:
                logger.info(
                    "Loading CTC model: %s  device=%s  dtype=%s",
                    self.model_name, self.device, self._dtype,
                )
                self._model = Wav2Vec2ForCTC.from_pretrained(
                    self.model_name,
                    torch_dtype=self._dtype,
                    ignore_mismatched_sizes=True,
                ).to(self.device)
                self._model.eval()
                logger.info("CTC model loaded.")

            # Load processor once — it is language-agnostic; only the tokenizer
            # target language needs updating on subsequent switches.
            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(self.model_name)
                logger.info("Processor loaded.")

            # Switch language adapter if needed
            if self._current_lang != lang_mms:
                logger.info("Switching language adapter to: %s", lang_mms)
                if self._is_mms:
                    self._processor.tokenizer.set_target_lang(lang_mms)
                    # load_adapter() places new weights on CPU — move back to device
                    self._model.load_adapter(lang_mms)
                    self._model.to(self.device)
                self._current_lang = lang_mms
                logger.info("Language adapter active: %s  device=%s", lang_mms, self.device)

            return True

        except Exception as exc:
            logger.warning("Failed to load model/adapter for '%s': %s", lang_mms, exc)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_candidates(
        self,
        audio: np.ndarray,
        candidates: list[str],
        lang_mms: str,
        sample_rate: int = 16000,
    ) -> dict[str, float]:
        """
        Score each candidate against *audio* using CTC log-likelihood.

        Args:
            audio:       1-D float32 array (any sample rate; resampled internally)
            candidates:  list of candidate spoken-form strings
            lang_mms:    MMS ISO 639-3 language code (e.g. "eng", "fra")
            sample_rate: sample rate of *audio* (default 16 kHz)

        Returns:
            dict mapping candidate → score (higher = better match).
            Candidates that cannot be scored receive -inf.
        """
        if not candidates:
            return {}

        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)

        if not self._ensure_loaded(lang_mms):
            return {c: 0.0 for c in candidates}  # equal fallback

        # ── Run encoder ───────────────────────────────────────────────
        try:
            inputs = self._processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=False,
            )
            input_values = inputs.input_values.to(self.device)

            # autocast: FP16 ops on CUDA, no-op on CPU
            with torch.autocast(device_type=self.device.split(":")[0], dtype=self._dtype):
                logits = self._model(input_values).logits   # [1, T, V]

            # CTC loss requires FP32
            log_probs = F.log_softmax(logits.float(), dim=-1)   # [1, T, V]
            T = log_probs.shape[1]
        except Exception as exc:
            logger.warning("Encoder error: %s", exc)
            return {c: 0.0 for c in candidates}

        # ── Score each candidate ──────────────────────────────────────
        blank_id = self._processor.tokenizer.pad_token_id
        scores: dict[str, float] = {}

        for candidate in candidates:
            try:
                # Normalize before tokenizing so phonetically identical variants
                # score equally: hyphens ("ninety-nine" == "ninety nine"),
                # zero placeholder "o" ("eight o five" == "eight oh five"),
                # and am/pm spacing ("a m" == "am", "p m" == "pm").
                text_for_tokens = " ".join(candidate.replace("-", " ").split())
                text_for_tokens = re.sub(r"\bo\b(?!')", "oh", text_for_tokens)
                text_for_tokens = re.sub(r"\ba\s+m\b", "am", text_for_tokens)
                text_for_tokens = re.sub(r"\bp\s+m\b", "pm", text_for_tokens)
                token_ids = self._processor.tokenizer(text_for_tokens).input_ids
                S = len(token_ids)

                if S == 0 or T < S:
                    # No tokens or audio too short for this candidate
                    scores[candidate] = float("-inf")
                    continue

                targets = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                in_len  = torch.tensor([T], dtype=torch.long, device=self.device)
                tgt_len = torch.tensor([S], dtype=torch.long, device=self.device)

                # CTC loss = -log P(y|x) per token; negate → log-likelihood score
                loss = F.ctc_loss(
                    log_probs.permute(1, 0, 2),  # [T, 1, V]
                    targets,
                    in_len,
                    tgt_len,
                    blank=blank_id,
                    reduction="mean",
                    zero_infinity=True,
                )
                scores[candidate] = -loss.item()

            except Exception as exc:
                logger.debug("Scoring error for '%s': %s", candidate, exc)
                scores[candidate] = float("-inf")

        return scores

    def best_candidate(
        self,
        audio: np.ndarray,
        candidates: list[str],
        lang_mms: str,
        sample_rate: int = 16000,
    ) -> tuple[str, dict[str, float]]:
        """
        Return ``(best_candidate, all_scores)``.

        Falls back to ``candidates[0]`` when scoring fails entirely.
        """
        if not candidates:
            raise ValueError("candidates list is empty")

        scores = self.score_candidates(audio, candidates, lang_mms, sample_rate)
        valid = {c: s for c, s in scores.items() if s > float("-inf")}

        if not valid:
            logger.warning("All candidates scored -inf; using first as fallback.")
            return candidates[0], scores

        best = max(valid, key=valid.__getitem__)
        return best, scores


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample *audio* from *orig_sr* to *target_sr* Hz."""
    try:
        import librosa
        return librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        pass
    try:
        import torchaudio
        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        resampled = torchaudio.functional.resample(waveform, orig_sr, target_sr)
        return resampled.squeeze(0).numpy()
    except Exception:
        pass
    logger.warning("Could not resample audio — using raw (may degrade quality).")
    return audio

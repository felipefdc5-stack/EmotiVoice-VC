# emotion_perceiver.py
"""
Utilities to extract emotion/style embeddings from short reference clips.

The implementation reuses the CAMPPlus encoder shipped with Seed-VC so we keep
the representation that the diffusion backbone expects (192-D).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from torchaudio import functional as F
from torchaudio.compliance import kaldi

try:
    from vendor.seedvc.modules.campplus.DTDNN import CAMPPlus
except ImportError:  # pragma: no cover - defensive fallback quando os módulos não estão acessíveis
    CAMPPlus = None  # type: ignore


@dataclass
class EmotionPerceiverConfig:
    """Configuration options for :class:`EmotionPerceiver`."""

    checkpoint_path: Optional[str] = None
    sample_rate: int = 16_000
    feat_dim: int = 80
    embedding_size: int = 192


class EmotionPerceiver:
    """Extracts 192-D emotion/style embeddings from short reference clips."""

    def __init__(
        self,
        model_cfg: Optional[EmotionPerceiverConfig] = None,
        device: str | torch.device = "cpu",
        shared_encoder: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        Args:
            model_cfg: configuration describing how to load CAMPPlus.
            device: target device.
            shared_encoder: optional pre-loaded CAMPPlus instance (for example,
                the one already initialised by Seed-VC). When provided, the
                perceiver will reuse it instead of creating a new model.
        """
        self.device = torch.device(device)
        cfg = model_cfg or EmotionPerceiverConfig()

        if shared_encoder is not None:
            self.encoder = shared_encoder
        else:
            if CAMPPlus is None:
                raise RuntimeError(
                    "CAMPPlus import failed. Ensure Seed-VC modules are available "
                    "or provide `shared_encoder`."
                )
            if cfg.checkpoint_path is None:
                raise ValueError(
                    "EmotionPerceiver requires `checkpoint_path` when a shared "
                    "encoder is not supplied."
                )
            state = torch.load(cfg.checkpoint_path, map_location="cpu")
            # Accept checkpoints saved either as raw state dict or wrapped.
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            self.encoder = CAMPPlus(
                feat_dim=cfg.feat_dim, embedding_size=cfg.embedding_size
            )
            self.encoder.load_state_dict(state)

        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        self.sample_rate = cfg.sample_rate
        self.feat_dim = cfg.feat_dim

    @torch.no_grad()
    def encode(self, emotion_wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Args:
            emotion_wav: mono waveform tensor (channels x samples or samples).
            sr: sampling rate of ``emotion_wav``.

        Returns:
            A tensor of shape ``(1, embedding_size)`` with the style embedding.
        """
        if emotion_wav.dim() == 1:
            emotion_wav = emotion_wav.unsqueeze(0)
        elif emotion_wav.dim() > 2:
            raise ValueError("Expected waveform tensor with <=2 dimensions.")

        # Use the first channel if stereo.
        if emotion_wav.size(0) > 1:
            emotion_wav = emotion_wav[:1, :]

        if sr != self.sample_rate:
            emotion_wav = F.resample(emotion_wav, sr, self.sample_rate)

        # CAMPPlus expects FBANK features (80 dims), same preprocessing used by Seed-VC.
        feat = kaldi.fbank(
            emotion_wav,
            num_mel_bins=self.feat_dim,
            dither=0.0,
            sample_frequency=float(self.sample_rate),
        )
        feat = feat - feat.mean(dim=0, keepdim=True)

        embedding = self.encoder(feat.unsqueeze(0).to(self.device))
        return embedding.detach()

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torchaudio

from modules.emotion.emotion_bridge import EmotionBridge, EmotionBridgeConfig
from modules.emotion.emotion_perceiver import EmotionPerceiverConfig
from modules.backends.seed_vc_backend_custom import SeedVCBackendCustom

LOGGER = logging.getLogger(__name__)


def _auto_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class IndexRVCFineTune:
    """
    Variante do pipeline IndexRVC que permite informar checkpoints/configs customizados
    (por exemplo, modelos fine-tunados em `modules-treining`).
    """

    def __init__(
        self,
        *,
        checkpoints_dir: Optional[Path] = None,
        device: Optional[str] = None,
        emo_matrix_path: Optional[Path] = None,
        spk_matrix_path: Optional[Path] = None,
        emo_num: Optional[Sequence[int]] = None,
        qwen_model_dir: Optional[Path] = None,
    ) -> None:
        self.device = _auto_device(device)
        LOGGER.info("Initialising IndexRVCFineTune on %s", self.device)

        ckpt_dir = Path(checkpoints_dir) if checkpoints_dir else Path(__file__).resolve().parent / "checkpoints"
        self.emo_cfg = EmotionBridgeConfig(
            qwen_model_dir=qwen_model_dir or ckpt_dir / "qwen0.6bemo4-merge",
            emo_matrix_path=emo_matrix_path or ckpt_dir / "feat2.pt",
            spk_matrix_path=spk_matrix_path or ckpt_dir / "feat1.pt",
            emo_num=emo_num or [3, 17, 2, 8, 4, 5, 10, 24],
            perceiver=EmotionPerceiverConfig(checkpoint_path=None),
        )

        self.checkpoints_dir = ckpt_dir
        self._current_bundle: tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]] = (
            None,
            None,
            None,
            None,
        )
        self._init_backend()

    def _init_backend(
        self,
        base_checkpoint: Optional[Path] = None,
        base_config: Optional[Path] = None,
        f0_checkpoint: Optional[Path] = None,
        f0_config: Optional[Path] = None,
    ):
        LOGGER.info("Loading Seed-VC backend (custom checkpoints: %s)", base_checkpoint)
        self.backend = SeedVCBackendCustom(
            device=str(self.device),
            base_checkpoint=base_checkpoint,
            base_config=base_config,
            f0_checkpoint=f0_checkpoint,
            f0_config=f0_config,
        )
        self.emotions = EmotionBridge(
            config=self.emo_cfg,
            device=self.device,
            shared_style_encoder=self.backend.raw_style_encoder,
        )
        self._current_bundle = (base_checkpoint, base_config, f0_checkpoint, f0_config)

    def _maybe_reload_backend(
        self,
        base_checkpoint: Optional[Path],
        base_config: Optional[Path],
        f0_checkpoint: Optional[Path],
        f0_config: Optional[Path],
    ):
        bundle = (base_checkpoint, base_config, f0_checkpoint, f0_config)
        if bundle != self._current_bundle:
            self._init_backend(*bundle)

    def _load_audio(self, source) -> tuple[torch.Tensor, int]:
        if isinstance(source, (str, Path)):
            waveform, sr = torchaudio.load(str(source))
            return waveform, sr
        if isinstance(source, np.ndarray):
            tensor = torch.from_numpy(source).float()
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            return tensor, self.backend.sr
        if isinstance(source, torch.Tensor):
            raise ValueError("Provide sampling rate when passing raw tensors.")
        raise TypeError(f"Unsupported audio input type: {type(source)!r}")

    def convert(
        self,
        *,
        source_audio: str | Path,
        target_audio: str | Path,
        base_checkpoint: Optional[Path] = None,
        base_config: Optional[Path] = None,
        f0_checkpoint: Optional[Path] = None,
        f0_config: Optional[Path] = None,
        emotion_audio: Optional[str | Path | torch.Tensor] = None,
        emotion_audio_sr: Optional[int] = None,
        emotion_text: Optional[str] = None,
        emotion_vector: Optional[Sequence[float]] = None,
        text_alpha: float = 1.0,
        audio_alpha: float = 1.0,
        randomize_emotion: bool = False,
        diffusion_steps: int = 30,
        length_adjust: float = 1.0,
        f0_condition: bool = False,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0,
        stream_output: bool = False,
    ):
        self._maybe_reload_backend(base_checkpoint, base_config, f0_checkpoint, f0_config)

        emo_audio_tensor: Optional[torch.Tensor] = None
        emo_audio_rate: Optional[int] = emotion_audio_sr

        if emotion_audio is not None:
            if isinstance(emotion_audio, (str, Path)):
                emo_audio_tensor, emo_audio_rate = torchaudio.load(str(emotion_audio))
            elif isinstance(emotion_audio, torch.Tensor):
                if emotion_audio_sr is None:
                    raise ValueError("`emotion_audio_sr` required when passing tensor.")
                emo_audio_tensor = emotion_audio
            else:
                raise TypeError("emotion_audio must be path or tensor.")

        transform = self.emotions.create_style_transform(
            emotion_text=emotion_text,
            emotion_vector=emotion_vector,
            emotion_audio=emo_audio_tensor,
            emotion_audio_sr=emo_audio_rate,
            text_alpha=text_alpha,
            audio_alpha=audio_alpha,
            randomize=randomize_emotion,
        )

        with self.backend.style_override(transform):
            result = self.backend.convert(
                source_audio=source_audio,
                target_audio=target_audio,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                f0_condition=f0_condition,
                auto_f0_adjust=auto_f0_adjust,
                pitch_shift=pitch_shift,
                stream_output=stream_output,
            )

        return result

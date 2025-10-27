"""
Backend alternativo que permite usar checkpoints fine-tunados armazenados em `modules-treining`.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import GeneratorType
from typing import Callable, Generator, Optional

import numpy as np
import torch

from vendor.seedvc.seed_vc_wrapper_custom import SeedVCWrapperCustom


class EmotionAwareStyleEncoder(torch.nn.Module):
    def __init__(self, base_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self._style_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def set_transform(self, transform: Optional[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self._style_transform = transform

    @contextmanager
    def override(self, transform: Optional[Callable[[torch.Tensor], torch.Tensor]]):
        previous = self._style_transform
        self._style_transform = transform
        try:
            yield
        finally:
            self._style_transform = previous

    def forward(self, *args, **kwargs) -> torch.Tensor:  # type: ignore[override]
        style = self.base_encoder(*args, **kwargs)
        if self._style_transform is not None:
            style = self._style_transform(style)
        return style

    @torch.no_grad()
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.base_encoder(features)


class SeedVCBackendCustom:
    """
    Igual ao backend principal, mas permite informar checkpoints/configs customizados.
    """

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        base_checkpoint: Optional[str | Path] = None,
        base_config: Optional[str | Path] = None,
        f0_checkpoint: Optional[str | Path] = None,
        f0_config: Optional[str | Path] = None,
    ) -> None:
        device_arg = torch.device(device) if device else None
        self.wrapper = SeedVCWrapperCustom(
            device=device_arg,
            base_checkpoint=base_checkpoint,
            base_config=base_config,
            f0_checkpoint=f0_checkpoint,
            f0_config=f0_config,
        )

        self.raw_style_encoder = self.wrapper.campplus_model
        self.style_adapter = EmotionAwareStyleEncoder(self.raw_style_encoder)
        self.wrapper.campplus_model = self.style_adapter

    @property
    def sr(self) -> int:
        return getattr(self.wrapper, "sr", 22_050)

    @contextmanager
    def style_override(self, transform: Optional[Callable[[torch.Tensor], torch.Tensor]]):
        with self.style_adapter.override(transform):
            yield

    def convert(
        self,
        source_audio: str | Path,
        target_audio: str | Path,
        *,
        diffusion_steps: int = 30,
        length_adjust: float = 1.0,
        stream_output: bool = False,
        **kwargs,
    ) -> tuple[int, np.ndarray] | Generator:
        result = self.wrapper.convert_voice(
            str(source_audio),
            str(target_audio),
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            inference_cfg_rate=kwargs.get("inference_cfg_rate", 0.7),
            f0_condition=kwargs.get("f0_condition", False),
            auto_f0_adjust=kwargs.get("auto_f0_adjust", True),
            pitch_shift=kwargs.get("pitch_shift", 0),
            stream_output=stream_output,
        )

        if stream_output:
            return result

        if isinstance(result, np.ndarray):
            return self.sr, result

        if isinstance(result, GeneratorType):
            try:
                while True:
                    next(result)
            except StopIteration as stop:
                full_audio = stop.value
            else:  # pragma: no cover
                full_audio = None

            if isinstance(full_audio, tuple):
                _, full_audio = full_audio

            if isinstance(full_audio, np.ndarray):
                return self.sr, full_audio

        raise RuntimeError("Saida inesperada do Seed-VC; use `stream_output=False` para obter o audio completo.")

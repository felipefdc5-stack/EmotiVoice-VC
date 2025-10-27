"""
Subclasse do SeedVCWrapper original permitindo definir checkpoints/configs customizados.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import AutoFeatureExtractor, WhisperModel

from vendor.seedvc.seed_vc_wrapper import SeedVCWrapper
from .modules.audio import mel_spectrogram
from .modules.commons import build_model, load_checkpoint, recursive_munch


class SeedVCWrapperCustom(SeedVCWrapper):
    def __init__(
        self,
        device=None,
        base_checkpoint: Optional[str | Path] = None,
        base_config: Optional[str | Path] = None,
        f0_checkpoint: Optional[str | Path] = None,
        f0_config: Optional[str | Path] = None,
    ):
        self._override_base_ckpt = Path(base_checkpoint) if base_checkpoint else None
        self._override_base_cfg = Path(base_config) if base_config else None
        self._override_f0_ckpt = Path(f0_checkpoint) if f0_checkpoint else None
        self._override_f0_cfg = Path(f0_config) if f0_config else None
        super().__init__(device=device)

    def _load_base_model(self):
        if self._override_base_ckpt is None or self._override_base_cfg is None:
            return super()._load_base_model()

        config = yaml.safe_load(open(self._override_base_cfg, "r"))
        model_params = recursive_munch(config["model_params"])
        self.model = build_model(model_params, stage="DiT")
        self.hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        self.sr = config["preprocess_params"]["sr"]

        self.model, _, _, _ = load_checkpoint(
            self.model,
            None,
            self._override_base_ckpt,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        mel_fn_args = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr,
            "fmin": 0,
            "fmax": None,
            "center": False,
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        whisper_name = (
            model_params.speech_tokenizer.whisper_name
            if hasattr(model_params.speech_tokenizer, "whisper_name")
            else "openai/whisper-small"
        )
        self.whisper_model = WhisperModel.from_pretrained(
            whisper_name, torch_dtype=torch.float16
        ).to(self.device)
        del self.whisper_model.decoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    def _load_f0_model(self):
        if self._override_f0_ckpt is None or self._override_f0_cfg is None:
            return super()._load_f0_model()

        config = yaml.safe_load(open(self._override_f0_cfg, "r"))
        model_params = recursive_munch(config["model_params"])
        self.model_f0 = build_model(model_params, stage="DiT")
        self.hop_length_f0 = config["preprocess_params"]["spect_params"]["hop_length"]
        self.sr_f0 = config["preprocess_params"]["sr"]

        self.model_f0, _, _, _ = load_checkpoint(
            self.model_f0,
            None,
            self._override_f0_ckpt,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in self.model_f0:
            self.model_f0[key].eval()
            self.model_f0[key].to(self.device)
        self.model_f0.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        mel_fn_args_f0 = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr_f0,
            "fmin": 0,
            "fmax": None,
            "center": False,
        }
        self.to_mel_f0 = lambda x: mel_spectrogram(x, **mel_fn_args_f0)

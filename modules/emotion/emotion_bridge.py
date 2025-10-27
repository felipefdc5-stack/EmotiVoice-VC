"""
Bridge helpers to convert IndexTTS-style emotion controls into Seed-VC style
embeddings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .emotion_perceiver import EmotionPerceiver, EmotionPerceiverConfig

LOGGER = logging.getLogger(__name__)

EMOTION_ORDER = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


def normalize_emo_vec(emo_vector: Sequence[float], apply_bias: bool = True) -> Sequence[float]:
    """Replicates IndexTTS2 emotion normalisation logic."""

    if len(emo_vector) != len(EMOTION_ORDER):
        raise ValueError(f"Expected {len(EMOTION_ORDER)} elements, got {len(emo_vector)}.")

    if apply_bias:
        bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
        emo_vector = [value * weight for value, weight in zip(emo_vector, bias)]

    total = sum(emo_vector)
    if total > 0.8:
        scale = 0.8 / total
        emo_vector = [value * scale for value in emo_vector]

    return emo_vector


def find_most_similar_cosine(query_vector: torch.Tensor, matrix: torch.Tensor) -> int:
    """Return the index with highest cosine similarity to ``query_vector``."""

    query_vector = query_vector.float().unsqueeze(0)
    matrix = matrix.float()
    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    return int(torch.argmax(similarities).item())


@dataclass
class EmotionBridgeConfig:
    qwen_model_dir: Optional[Path] = None
    emo_matrix_path: Optional[Path] = None
    spk_matrix_path: Optional[Path] = None
    emo_num: Optional[Sequence[int]] = None
    perceiver: Optional[EmotionPerceiverConfig] = None


class QwenEmotion:
    """Thin wrapper around the emotion-aware Qwen checkpoint from IndexTTS."""

    def __init__(self, model_dir: Path, device: torch.device):
        self.model_dir = Path(model_dir)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16 if device.type.startswith("cuda") else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.prompt = "You are an assistant that maps expressive speech descriptions to eight emotion scores."
        self.max_score = 1.2
        self.min_score = 0.0

    def clamp(self, value: float) -> float:
        return max(self.min_score, min(self.max_score, value))

    def inference(self, text_input: str) -> Dict[str, float]:
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": text_input},
        ]
        chat = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        model_inputs = self.tokenizer([chat], return_tensors="pt").to(self.model.device)

        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=4096,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_ids = generated[0][len(model_inputs.input_ids[0]) :].tolist()
        decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        try:
            content = json.loads(decoded)
        except json.JSONDecodeError:
            LOGGER.warning("QwenEmotion returned non-JSON payload. Falling back to regex parsing.")
            content = {}
            for token in decoded.split(","):
                if ":" not in token:
                    continue
                key, value = token.split(":", maxsplit=1)
                try:
                    content[key.strip()] = float(value.strip())
                except ValueError:
                    continue

        emo_dict = {name: self.clamp(content.get(name, 0.0)) for name in EMOTION_ORDER}
        if all(val <= 0.0 for val in emo_dict.values()):
            emo_dict["calm"] = 1.0
        return emo_dict


class EmotionBridge:
    """Converts text/audio emotion prompts into Seed-VC style overrides."""

    def __init__(
        self,
        config: EmotionBridgeConfig,
        device: torch.device,
        shared_style_encoder: torch.nn.Module,
    ) -> None:
        self.device = device

        self.emo_matrix = None
        self.spk_matrix = None
        self.emo_num = config.emo_num

        if config.emo_matrix_path and config.spk_matrix_path and config.emo_num:
            emo_matrix = torch.load(config.emo_matrix_path, map_location="cpu")
            spk_matrix = torch.load(config.spk_matrix_path, map_location="cpu")
            self.emo_matrix = torch.split(emo_matrix, config.emo_num)
            self.spk_matrix = torch.split(spk_matrix, config.emo_num)
        else:
            LOGGER.warning("Emotion matrices not provided; text-driven emotion control disabled.")

        self.qwen = None
        if config.qwen_model_dir:
            try:
                self.qwen = QwenEmotion(Path(config.qwen_model_dir), device=device)
            except OSError as exc:  # pragma: no cover - optional dependency
                LOGGER.warning("Failed to load QwenEmotion: %s", exc)

        self.perceiver = EmotionPerceiver(
            model_cfg=config.perceiver,
            device=device,
            shared_encoder=shared_style_encoder,
        )

    def create_style_transform(
        self,
        *,
        emotion_text: Optional[str] = None,
        emotion_vector: Optional[Sequence[float]] = None,
        emotion_audio: Optional[torch.Tensor] = None,
        emotion_audio_sr: Optional[int] = None,
        text_alpha: float = 1.0,
        audio_alpha: float = 1.0,
        randomize: bool = False,
    ) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        transforms = []

        if emotion_audio is not None:
            if emotion_audio_sr is None:
                raise ValueError("`emotion_audio_sr` must be provided alongside `emotion_audio`.")
            style = self.perceiver.encode(emotion_audio.to(self.device), emotion_audio_sr)
            print(">> emotion audio applied; embedding norm =", float(torch.norm(style)))
            transforms.append(self._build_audio_transform(style, audio_alpha))

        if emotion_vector is None and emotion_text:
            if self.qwen is None:
                LOGGER.warning("No QwenEmotion model available; ignoring text prompt.")
            else:
                emotion_vector = list(self.qwen.inference(emotion_text).values())

        if emotion_vector is not None and self.emo_matrix is not None and self.spk_matrix is not None:
            weights = normalize_emo_vec(emotion_vector)
            print(">> text/vector emotion normalized =", [round(float(w), 4) for w in weights])
            transforms.append(self._build_text_transform(weights, text_alpha, randomize))
        elif emotion_vector is not None:
            LOGGER.warning("Emotion matrices missing; cannot apply emotion vectors.")

        if not transforms:
            return None

        def apply(style_tensor: torch.Tensor) -> torch.Tensor:
            output = style_tensor
            for transform in transforms:
                output = transform(output)
            return output

        return apply

    @staticmethod
    def _align_dimensions(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """Ajusta candidate para ter o mesmo tamanho da última dimensão de reference."""
        if candidate.size(-1) == reference.size(-1):
            return candidate
        if candidate.size(-1) > reference.size(-1):
            return candidate[..., : reference.size(-1)]
        pad_size = reference.size(-1) - candidate.size(-1)
        return torch.nn.functional.pad(candidate, (0, pad_size))

    def _build_audio_transform(self, target_style: torch.Tensor, alpha: float) -> Callable[[torch.Tensor], torch.Tensor]:
        target_style = target_style.detach()

        def apply(style_tensor: torch.Tensor) -> torch.Tensor:
            aligned_target = self._align_dimensions(style_tensor, target_style.to(style_tensor.device))
            delta = aligned_target - style_tensor
            print(">> emotion audio delta norm =", float(torch.norm(delta)))
            return style_tensor + alpha * delta

        return apply

    def _build_text_transform(
        self, weights: Sequence[float], alpha: float, randomize: bool
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        def apply(style_tensor: torch.Tensor) -> torch.Tensor:
            base = style_tensor.squeeze(0)
            indices = []
            selected_vectors = []

            for emo_group, spk_group in zip(self.emo_matrix, self.spk_matrix):
                emo_group = emo_group.to(base.device)
                spk_group = spk_group.to(base.device)

                if randomize:
                    idx = torch.randint(low=0, high=spk_group.size(0), size=(1,), device=base.device).item()
                else:
                    idx = find_most_similar_cosine(base, spk_group)
                indices.append(idx)

                # usamos a matriz de speaker (192 dims) como base de estilo
                selected_vectors.append(spk_group[idx].unsqueeze(0))

            style_stack = torch.cat(selected_vectors, dim=0)
            delta = torch.sum(weights_tensor.to(base.device).unsqueeze(1) * style_stack, dim=0, keepdim=True)
            delta = self._align_dimensions(style_tensor, delta)
            print(">> emotion text/vector delta norm =", float(torch.norm(delta)))
            return style_tensor + alpha * delta

        return apply

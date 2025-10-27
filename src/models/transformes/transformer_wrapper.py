# transformer_wrapper.py
import torch


class SpeechToSemanticTransformer:
"""Adaptador do Autoregressive Transformer do IndexTTS para aceitar embeddings de audio.
Ele concatena/condiciona speaker+emotion embeddings e recebe content embeddings como "prompt"."""
def __init__(self, cfg, device='cpu'):
# load index-tts transformer
self.device = device


def generate_semantic(self, content_embeddings, speaker_emb, emotion_emb):
"""Gera tokens semânticos autoregressivamente (ou via sampling acelerado)."""
raise NotImplementedError
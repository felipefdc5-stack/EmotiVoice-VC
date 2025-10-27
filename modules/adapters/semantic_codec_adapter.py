# semantic_codec_adapter.py
class SemanticCodecAdapter:
"""Adapta a saída do transformer para o formato esperado pelo MaskGCT/codec."""
def __init__(self, codec_cfg):
pass


def decode(self, semantic_tokens):
# map semantic_tokens -> codec representation (discrete codes / mels)
raise NotImplementedError
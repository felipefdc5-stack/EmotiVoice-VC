# inference_rvc.py
"""Script de inferência end-to-end IndexRVC.
Fluxo:
- carregar wavs (input, speaker_ref, emotion_ref)
- content_embeddings = ContentEncoder.encode(input)
- speaker_emb = SpeakerEncoder.encode(speaker_ref)
- emotion_emb = EmotionPerceiver.encode(emotion_ref)
- semantic = Transformer.generate_semantic(content_embeddings, speaker_emb, emotion_emb)
- codec_features = SemanticCodecAdapter.decode(semantic)
- wav_out = VocoderAdapter.synthesize(codec_features)
- salvar wav_out
"""
import torch


def run_inference(input_wav, speaker_ref, emotion_ref, out_path, device='cpu'):
# instanciar módulos (carregar checkpoints)
# executar pipeline
raise NotImplementedError


if __name__ == '__main__':
# cli minimal
raise NotImplementedError
from pathlib import Path
import sys

import torch
import torchaudio

# Garante que o diretório raiz do IndexRVC esteja no sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import IndexRVC

if __name__ == "__main__":
    checkpoints = ROOT / "checkpoints"

    model = IndexRVC(
        checkpoints_dir=checkpoints,
    )

    sr, audio = model.convert(
        source_audio="examples/euFalando.wav",
        target_audio="examples/Nero_00000006.wav",
        emotion_text="Fale com um tom animado e caloroso",
    )

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    torchaudio.save("output.wav", audio_tensor, sr)

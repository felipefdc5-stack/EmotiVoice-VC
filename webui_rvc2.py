from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional

import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_custom import IndexRVCFineTune  # noqa: E402

EMOTION_LABELS = [
    "Feliz",
    "Bravo",
    "Triste",
    "Medo",
    "Desgosto",
    "Melancolia",
    "Surpreso",
    "Calmo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "EmotiVoice VC WebUI (custom checkpoints)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host HTTP do Gradio")
    parser.add_argument("--port", default=7862, type=int, help="Porta HTTP do Gradio")
    parser.add_argument("--share", action="store_true", help="Ativa link publico")
    parser.add_argument(
        "--checkpoints",
        default=str(ROOT / "checkpoints"),
        help="Pasta com os checkpoints base do EmotiVoice VC",
    )
    parser.add_argument(
        "--training-root",
        default=str(ROOT / "modules-treining"),
        help="Pasta com os modelos fine-tunados (ex.: modules-treining/NomeDoModelo)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Forca o dispositivo (ex.: 'cuda:0', 'cpu'). Deixe vazio para autodetectar.",
    )
    return parser.parse_args()


class LazyModel:
    """Inicializa o pipeline apenas na primeira chamada."""

    def __init__(self, checkpoints_dir: Path, device: Optional[str]):
        self.checkpoints_dir = checkpoints_dir
        self.device = device
        self._instance: Optional[IndexRVCFineTune] = None
        self._lock = threading.Lock()

    def instance(self) -> IndexRVCFineTune:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = IndexRVCFineTune(
                        checkpoints_dir=self.checkpoints_dir,
                        device=self.device,
                    )
        return self._instance


def toggle_emotion_controls(mode: str):
    show_audio = mode == "Usar audio de emocao"
    show_text = mode == "Usar texto"
    show_vectors = mode == "Usar vetores"

    updates = [
        gr.update(visible=show_audio),
        gr.update(visible=show_text, interactive=show_text),
        gr.update(visible=show_vectors),
        gr.update(visible=show_text or show_vectors),
        gr.update(visible=show_audio),
        gr.update(visible=show_vectors),
    ]
    return updates


def list_training_folders(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_files(folder: Path, suffixes: tuple[str, ...]) -> List[str]:
    if not folder.exists():
        return []
    files = []
    for entry in folder.iterdir():
        if entry.is_file() and entry.suffix.lower() in suffixes:
            files.append(entry.name)
    return sorted(files)


def build_interface(model: LazyModel, training_root: Path) -> gr.Blocks:
    default_message = (
        "Importante: esta instancia esta rodando em CPU. Para usar a sua GPU (ex.: RTX 4060) "
        "instale o PyTorch com CUDA e reinicie a WebUI com `--device cuda`."
    )

    with gr.Blocks(css=".compact-slider .wrap {gap: 4px;} .compact-slider label {min-width: 110px;}") as demo:
        gr.Markdown("## EmotiVoice VC - Conversao de voz com emocao (checkpoints customizados)")
        gr.Markdown(default_message)

        with gr.Row():
            with gr.Column():
                source_audio = gr.Audio(
                    label="Fonte",
                    type="filepath",
                    sources=["microphone", "upload"],
                    interactive=True,
                )
                gr.Markdown("Audio de entrada (voz que sera convertida).")
            with gr.Column():
                target_audio = gr.Audio(
                    label="Alvo",
                    type="filepath",
                    sources=["microphone", "upload"],
                    interactive=True,
                )
                gr.Markdown("Voz de referencia que define o timbre a ser clonado.")
            with gr.Column():
                output_audio = gr.Audio(label="Resultado", interactive=False)
                status_box = gr.Markdown("")
                convert_btn = gr.Button("Converter", variant="primary")

        with gr.Accordion("Selecao do modelo fine-tunado", open=True):
            folders_available = ["(usar padrao HF)"] + list_training_folders(training_root)
            folder_dropdown = gr.Dropdown(
                label="Pasta (modules-treining)",
                choices=folders_available,
                value=folders_available[0],
            )
            checkpoint_dropdown = gr.Dropdown(label="Checkpoint (.pth)", choices=["(usar padrao HF)"], value="(usar padrao HF)")
            config_dropdown = gr.Dropdown(label="Config (.yml)", choices=["(usar padrao HF)"], value="(usar padrao HF)")
            f0_checkpoint_dropdown = gr.Dropdown(
                label="Checkpoint F0 (.pth) - opcional", choices=["(usar padrao HF)"], value="(usar padrao HF)"
            )
            f0_config_dropdown = gr.Dropdown(
                label="Config F0 (.yml) - opcional", choices=["(usar padrao HF)"], value="(usar padrao HF)"
            )

        with gr.Accordion("Controle emocional", open=True):
            emotion_mode = gr.Dropdown(
                label="Modo de emocao",
                choices=["Sem emocao extra", "Usar audio de emocao", "Usar vetores", "Usar texto"],
                value="Sem emocao extra",
            )
            emotion_audio = gr.Audio(
                label="Audio com a emocao desejada",
                type="filepath",
                sources=["microphone", "upload"],
                visible=False,
            )
            emotion_text = gr.Textbox(
                label="Descricao textual da emocao",
                placeholder="Ex.: Speak with furious anger and shouting tone",
                visible=False,
            )
            with gr.Column(visible=False) as vector_group:
                gr.Markdown("Ajuste manual das intensidades emocionais (0.0 a 1.0):")
                emotion_sliders: List[gr.Slider] = []
                for label in EMOTION_LABELS:
                    slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.0,
                        label=label,
                        elem_classes=["compact-slider"],
                    )
                    emotion_sliders.append(slider)

            text_alpha = gr.Slider(
                label="Intensidade emocao (texto/vetores)",
                minimum=0.0,
                maximum=1.5,
                step=0.05,
                value=1.0,
                visible=False,
            )
            audio_alpha = gr.Slider(
                label="Intensidade emocao (audio)",
                minimum=0.0,
                maximum=1.5,
                step=0.05,
                value=1.0,
                visible=False,
            )
            randomize_checkbox = gr.Checkbox(
                label="Escolher vetores aleatorios compativeis",
                value=False,
                visible=False,
            )

        with gr.Accordion("Parametros avancados", open=False):
            diffusion_steps = gr.Slider(
                label="Diffusion steps",
                minimum=5,
                maximum=40,
                step=1,
                value=20,
            )
            length_adjust = gr.Slider(
                label="Length adjust",
                minimum=0.6,
                maximum=1.6,
                step=0.02,
                value=1.0,
            )
            pitch_shift = gr.Slider(
                label="Deslocamento de pitch (semitons)",
                minimum=-12,
                maximum=12,
                step=1,
                value=0,
            )

        def refresh_files(selected_folder: str):
            if selected_folder == "(usar padrao HF)":
                default_choice = ["(usar padrao HF)"]
                return (
                    gr.update(choices=default_choice, value=default_choice[0]),
                    gr.update(choices=default_choice, value=default_choice[0]),
                    gr.update(choices=default_choice, value=default_choice[0]),
                    gr.update(choices=default_choice, value=default_choice[0]),
                )
            folder_path = training_root / selected_folder
            ckpts = ["(usar padrao HF)"] + list_files(folder_path, (".pth", ".pt"))
            cfgs = ["(usar padrao HF)"] + list_files(folder_path, (".yml", ".yaml"))
            return (
                gr.update(choices=ckpts, value=ckpts[0]),
                gr.update(choices=cfgs, value=cfgs[0]),
                gr.update(choices=ckpts, value=ckpts[0]),
                gr.update(choices=cfgs, value=cfgs[0]),
            )

        def do_convert(
            src_path: Optional[str],
            tgt_path: Optional[str],
            selected_folder: str,
            selected_ckpt: str,
            selected_cfg: str,
            selected_f0_ckpt: str,
            selected_f0_cfg: str,
            mode: str,
            emo_audio_path: Optional[str],
            emo_text_value: str,
            *extra_args,
        ):
            vector_values = list(extra_args[: len(EMOTION_LABELS)])
            ta_value = extra_args[len(EMOTION_LABELS)]
            aa_value = extra_args[len(EMOTION_LABELS) + 1]
            randomize_flag = extra_args[len(EMOTION_LABELS) + 2]
            diff_steps = int(extra_args[len(EMOTION_LABELS) + 3])
            length_adj = float(extra_args[len(EMOTION_LABELS) + 4])
            pitch = int(extra_args[len(EMOTION_LABELS) + 5])

            if not src_path or not os.path.exists(src_path):
                return None, "Selecione um audio na aba Fonte."
            if not tgt_path or not os.path.exists(tgt_path):
                return None, "Selecione um audio na aba Alvo."

            base_ckpt_path = None
            base_cfg_path = None
            f0_ckpt_path = None
            f0_cfg_path = None
            if selected_folder != "(usar padrao HF)":
                folder = training_root / selected_folder
                if selected_ckpt != "(usar padrao HF)":
                    base_ckpt_path = folder / selected_ckpt
                if selected_cfg != "(usar padrao HF)":
                    base_cfg_path = folder / selected_cfg
                if selected_f0_ckpt != "(usar padrao HF)":
                    f0_ckpt_path = folder / selected_f0_ckpt
                if selected_f0_cfg != "(usar padrao HF)":
                    f0_cfg_path = folder / selected_f0_cfg

            kwargs = dict(
                source_audio=src_path,
                target_audio=tgt_path,
                base_checkpoint=base_ckpt_path,
                base_config=base_cfg_path,
                f0_checkpoint=f0_ckpt_path,
                f0_config=f0_cfg_path,
                diffusion_steps=diff_steps,
                length_adjust=length_adj,
                pitch_shift=pitch,
            )

            if mode == "Usar audio de emocao" and emo_audio_path and os.path.exists(emo_audio_path):
                kwargs["emotion_audio"] = emo_audio_path
                kwargs["audio_alpha"] = float(aa_value)
            elif mode == "Usar vetores":
                kwargs["emotion_vector"] = [float(v) for v in vector_values]
                kwargs["text_alpha"] = float(ta_value)
                kwargs["randomize_emotion"] = bool(randomize_flag)
            elif mode == "Usar texto" and emo_text_value.strip():
                kwargs["emotion_text"] = emo_text_value.strip()
                kwargs["text_alpha"] = float(ta_value)

            try:
                sr, audio = model.instance().convert(**kwargs)
            except Exception as exc:  # pragma: no cover
                return None, f"Erro durante a conversao: {exc}"

            audio = np.asarray(audio, dtype=np.float32)
            return (sr, audio), "Conversao concluida."

        convert_btn.click(
            fn=do_convert,
            inputs=[
                source_audio,
                target_audio,
                folder_dropdown,
                checkpoint_dropdown,
                config_dropdown,
                f0_checkpoint_dropdown,
                f0_config_dropdown,
                emotion_mode,
                emotion_audio,
                emotion_text,
                *emotion_sliders,
                text_alpha,
                audio_alpha,
                randomize_checkbox,
                diffusion_steps,
                length_adjust,
                pitch_shift,
            ],
            outputs=[output_audio, status_box],
        )

        emotion_mode.change(
            fn=toggle_emotion_controls,
            inputs=emotion_mode,
            outputs=[
                emotion_audio,
                emotion_text,
                vector_group,
                text_alpha,
                audio_alpha,
                randomize_checkbox,
            ],
        )

        folder_dropdown.change(
            fn=refresh_files,
            inputs=folder_dropdown,
            outputs=[
                checkpoint_dropdown,
                config_dropdown,
                f0_checkpoint_dropdown,
                f0_config_dropdown,
            ],
        )

        return demo


def main():
    args = parse_args()
    checkpoints_dir = Path(args.checkpoints)
    training_root = Path(args.training_root)

    if not checkpoints_dir.exists():
        raise SystemExit(f"Pasta de checkpoints nao encontrada: {checkpoints_dir}")

    lazy_model = LazyModel(checkpoints_dir=checkpoints_dir, device=args.device)
    demo = build_interface(lazy_model, training_root)

    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()

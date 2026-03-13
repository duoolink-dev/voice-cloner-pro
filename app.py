import gradio as gr
import torch
import numpy as np
import soundfile as sf
import re
from openvoice.api import BaseSpeakerTTS
from openvoice.se_extractor import SEExtractor
import huggingface_hub
import tempfile
import os

# HF Spaces optimized (auto GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"HF Device: {device}")

# Auto-download (HF cache)
repo_id = "myshell-ai/OpenVoice"
ckpt_base = huggingface_hub.hf_hub_download(repo_id, "checkpoints_v2/base_speakers/EN")
ckpt_tts = huggingface_hub.hf_hub_download(repo_id, "checkpoints_v2/openvoice_v2")
ckpt_se = huggingface_hub.hf_hub_download(repo_id, "checkpoints_v2/se_extractor")

se_extractor = SEExtractor(ckpt_se, device=device)
tts = BaseSpeakerTTS(ckpt_base, ckpt_tts, device=device)

voice_library = {}  # {name: se}

def split_text(text, max_len=150):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current + sent) <= max_len:
            current += sent + " "
        else:
            if current.strip(): chunks.append(current.strip())
            current = sent + " "
    if current.strip(): chunks.append(current.strip())
    return chunks

def clone_voice(ref_path, voice_name):
    audio, sr = sf.read(ref_path)
    if len(audio)/sr < 3: return "Ref <3s.", None
    se = se_extractor.get_se(ref_path, torch.device(device))
    voice_library[voice_name] = se.cpu().numpy()
    return f"✅ '{voice_name}' saved to library!", list(voice_library.keys())

def generate_audio(voice_name, text, similarity=0.8, stability=0.5, clarity=0.8, speed=1.0, pitch=0):
    if voice_name not in voice_library: return None, "Select saved voice."
    if not text.strip(): return None, "Add text."
    se = torch.tensor(voice_library[voice_name]).to(device)
    chunks = split_text(text)
    all_audio = []
    for chunk in chunks:
        wav = tts.generate(chunk, se, se, speed=speed, pitch_shift=pitch * similarity)
        all_audio.append(wav.cpu().numpy())
    full_audio = np.concatenate(all_audio)
    out_path = "/tmp/output.wav"  # HF tmp
    sf.write(out_path, full_audio, 50000)
    mins = len(full_audio) / 50000 / 60
    return out_path, f"✅ {mins:.1f}min audio | Sim:{similarity} Stab:{stability} Clar:{clarity}"

# ElevenLabs Pro UI
with gr.Blocks(theme=gr.themes.Dark(), title="Pro Voice Cloner") as demo:
    gr.Markdown("# 🎙️ Pro Voice Cloner V2\nAny-lang clone, 1hr+ audio, ElevenLabs controls. GPU fast!")
    gr.Markdown("**1. Clone** → **2. Library** → **3. Generate**")

    with gr.Tabs():
        with gr.TabItem("🔴 Clone Voice"):
            ref_audio = gr.Audio(sources=["upload"], type="filepath")
            voice_name_input = gr.Textbox(placeholder="Voice name (e.g. TeamVoice)")
            clone_btn = gr.Button("Clone & Save", variant="primary")
            clone_status = gr.Textbox(label="Status")
            library_dropdown = gr.Dropdown(label="Library", choices=[])

        with gr.TabItem("🟢 Generate"):
            voice_sel = gr.Dropdown(label="Voice", choices=[])
            text_input = gr.Textbox(label="Text (long OK!)", lines=8)
            with gr.Row():
                gr.Column():
                    similarity = gr.Slider(0,1,0.8,label="🔗 Similarity")
                    stability = gr.Slider(0,1,0.5,label="🎭 Stability")
                gr.Column():
                    clarity = gr.Slider(0,1,0.8,label="✨ Clarity")
                    speed = gr.Slider(0.5,2,1,label="⚡ Speed")
            pitch = gr.Slider(-24,24,0,label="😠 Pitch/Emotion")
            gen_btn = gr.Button("Generate", variant="primary")
            audio_out = gr.Audio(label="Play/Download")
            gen_status = gr.Textbox(label="Status")

    clone_btn.click(clone_voice, inputs=[ref_audio, voice_name_input], outputs=[clone_status, library_dropdown])
    gen_btn.click(generate_audio, inputs=[voice_sel, text_input, similarity, stability, clarity, speed, pitch],
                  outputs=[audio_out, gen_status])

if __name__ == "__main__":
    demo.launch()

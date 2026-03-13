import gradio as gr
import torch
import numpy as np
import soundfile as sf
import re
import os
from openvoice.api import BaseSpeakerTTS, se_extractor

print("Loading OpenVoice V2 HF...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# HF-optimized checkpoints (V2)
ckpt_base = "checkpoints_v2/base_speakers/EN"
ckpt_tts = "checkpoints_v2/openvoice_v2"
ckpt_se = "checkpoints_v2/se_extractor"

# Download if missing
os.makedirs("checkpoints_v2", exist_ok=True)
import huggingface_hub
huggingface_hub.snapshot_download("myshell-ai/OpenVoice", local_dir="checkpoints_v2")

# Load
se_extractor = se_extractor.from_checkpoint(f"checkpoints_v2/{ckpt_se}.pth")
tts = BaseSpeakerTTS(f"checkpoints_v2/{ckpt_base}.pth", f"checkpoints_v2/{ckpt_tts}.pth", device=device)

voice_library = {}

def split_text(text, max_len=150):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
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
    if len(audio) / sr < 3:
        return "❌ Ref <3s. Use longer.", []
    source_se = se_extractor.get_se(ref_path, torch.device(device))
    voice_library[voice_name] = source_se.cpu().numpy()
    return f"✅ '{voice_name}' cloned & saved!", list(voice_library.keys())

def generate_audio(voice_name, text, similarity=0.8, stability=0.5, clarity=0.8, speed=1.0, pitch=0):
    if voice_name not in voice_library:
        return None, "❌ Select saved voice first."
    if not text.strip():
        return None, "❌ Add text."
    se = torch.tensor(voice_library[voice_name]).unsqueeze(0).to(device)
    chunks = split_text(text)
    all_audio = []
    for chunk in chunks:
        wav = tts.generate(
            chunk,
            se,
            se,  # Target = source for clone
            speed=speed,
            pitch_shift=int(pitch * similarity * 12)
        )
        all_audio.append(wav.cpu().numpy())
    full_audio = np.concatenate(all_audio, axis=0)
    out_path = "/tmp/output.wav"
    sf.write(out_path, full_audio, 50000)
    mins = len(full_audio) / 50000 / 60
    return out_path, f"✅ {len(chunks)} chunks = {mins:.1f}min audio\nSim: {similarity} | Stab: {stability} | Clar: {clarity}"

# Pro ElevenLabs UI (Dark)
with gr.Blocks(theme=gr.themes.Dark(), title="Voice Cloner Pro") as demo:
    gr.Markdown("# 🎙️ **Pro Voice Cloner (OpenVoice V2)**\n*Any language, 1hr+ audio, fast GPU.*\n**Clone → Save → Generate**")
    
    with gr.Tabs():
        with gr.TabItem("🔴 1. Clone Voice"):
            ref_audio = gr.Audio(sources=["upload"], type="filepath", label="Upload sample (3s+ any lang)")
            voice_name = gr.Textbox(placeholder="e.g. 'MyVoice'", label="Voice name")
            clone_btn = gr.Button("Clone & Save to Library", variant="primary", size="lg")
            clone_status = gr.Textbox(label="Status")
            library_choices = gr.Dropdown(label="Saved Voices", choices=[], interactive=True)
        
        with gr.TabItem("🟢 2. Generate"):
            voice_sel = gr.Dropdown(label="Select Voice", choices=[])
            text_input = gr.Textbox(label="Text to speak (1hr+ OK!)", lines=8, placeholder="Paste long script here...")
            with gr.Row():
                with gr.Column():
                    similarity = gr.Slider(0, 1, 0.8, step=0.1, label="🔗 Similarity")
                    stability = gr.Slider(0, 1, 0.5, step=0.1, label="🎭 Stability")
                with gr.Column():
                    clarity = gr.Slider(0, 1, 0.8, step=0.1, label="✨ Clarity")
                    speed = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="⚡ Speed")
            pitch_slider = gr.Slider(-24, 24, 0, step=2, label="😠 Pitch/Emotion (-calm +excited)")
            gen_btn = gr.Button("🎤 Generate Cloned Audio", variant="primary", size="lg")
            audio_out = gr.Audio(label="✅ Play & Download")
            gen_status = gr.Textbox(label="Status")
    
    # Connect buttons
    clone_btn.click(clone_voice, inputs=[ref_audio, voice_name], outputs=[clone_status, library_choices])
    clone_btn.change(lambda x: x, outputs=[voice_sel])  # Sync library
    gen_btn.click(generate_audio, inputs=[voice_sel, text_input, similarity, stability, clarity, speed, pitch_slider],
                  outputs=[audio_out, gen_status])

demo.launch()

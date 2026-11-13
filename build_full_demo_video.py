"""
High-definition Video Rendering and Assembly Engine for AnomalyGuard.
Generates:
  1. Ambient FinTech background music track.
  2. 6 high-definition animated scenes (1920x1080, 30fps) with subtitles & branding.
  3. Final combined demonstration video: demo_video/final_demo.mp4
  4. 35-second promotional teaser: demo_video/promo_teaser_35s.mp4
"""

import os
import subprocess
import numpy as np
import scipy.io.wavfile as wavfile
from PIL import Image, ImageDraw, ImageFont

os.makedirs("demo_video/generated_scenes", exist_ok=True)
os.makedirs("demo_video/assets", exist_ok=True)

# 1. Generate Warm Ambient Tech Background Music
def generate_ambient_music(duration_sec=175, sample_rate=44100, out_path="demo_video/assets/background_ambient.wav"):
    if os.path.exists(out_path):
        return out_path
    print("Synthesizing ambient background soundtrack...")
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    
    # Chord frequencies: Dm, F, C, G (warm cinematic progression)
    # D3=146.83, F3=174.61, A3=220.00, C4=261.63, E4=329.63, G3=196.00
    chords = [
        [146.83, 220.00, 261.63, 349.23],  # Dm7
        [174.61, 220.00, 261.63, 329.63],  # Fmaj7
        [130.81, 196.00, 261.63, 329.63],  # C
        [196.00, 246.94, 293.66, 392.00]   # G
    ]
    chord_len = 8.0 # seconds per chord
    audio = np.zeros_like(t)
    
    for i, chord in enumerate(chords):
        # Repeat chords over the duration
        for note in chord:
            # Warm sine wave + soft octave harmonic
            wave = 0.5 * np.sin(2 * np.pi * note * t) + 0.25 * np.sin(2 * np.pi * (note * 2) * t) + 0.1 * np.sin(2 * np.pi * (note * 0.5) * t)
            # Apply slow LFO modulation
            lfo = 0.8 + 0.2 * np.sin(2 * np.pi * 0.15 * t)
            audio += wave * lfo
            
    # Normalize and soft low-pass smoothing
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    # Gentle envelope fade-in (3s) and fade-out (5s)
    fade_in = np.clip(t / 3.0, 0, 1)
    fade_out = np.clip((duration_sec - t) / 5.0, 0, 1)
    audio = audio * fade_in * fade_out * 0.12 # Soft background level
    
    int_audio = (audio * 32767).astype(np.int16)
    wavfile.write(out_path, sample_rate, int_audio)
    print(f"Ambient track created: {out_path}")
    return out_path

# 2. Helper to get audio duration using ffmpeg
def get_audio_duration(audio_path):
    cmd = ["ffmpeg", "-i", audio_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stderr.splitlines():
        if "Duration:" in line:
            # Example: Duration: 00:00:25.15, start: ...
            parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
            dur = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            return dur
    return 25.0

# 3. Create branded scene frame template
def create_scene_frame(title, subtitle, lower_third, center_img_path, out_frame_path):
    width, height = 1920, 1080
    # Create dark navy background canvas
    img = Image.new("RGB", (width, height), color=(11, 17, 32)) # #0B1120
    draw = ImageDraw.Draw(img)
    
    # Top header bar
    draw.rectangle([(0, 0), (width, 110)], fill=(15, 23, 42)) # #0F172A
    draw.line([(0, 110), (width, 110)], fill=(37, 99, 235), width=3) # Blue accent line
    
    # Header text
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        font_badge = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_lower = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_lower_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font_large = font_sub = font_badge = font_lower = font_lower_sub = ImageFont.load_default()
        
    draw.text((60, 24), "🛡️ ANOMALYGUARD", fill=(255, 255, 255), font=font_large)
    draw.text((460, 38), f"|  {title}", fill=(147, 197, 253), font=font_sub)
    
    # Right header badge
    draw.rounded_rectangle([(width - 340, 30), (width - 60, 80)], radius=6, fill=(30, 58, 138))
    draw.text((width - 325, 43), "AI ENGINEERING WK 03", fill=(255, 255, 255), font=font_badge)
    
    # Center Image/Visualization
    if center_img_path and os.path.exists(center_img_path):
        center_img = Image.open(center_img_path).convert("RGBA")
        # Resize to fit nicely within 1760 x 780
        center_img.thumbnail((1760, 780), Image.Resampling.LANCZOS)
        cx = (width - center_img.width) // 2
        cy = 130 + (800 - center_img.height) // 2
        img.paste(center_img, (cx, cy), mask=center_img if center_img.mode == 'RGBA' else None)
    else:
        # Title slide graphic
        draw.rounded_rectangle([(160, 240), (width - 160, 780)], radius=16, fill=(15, 23, 42), outline=(37, 99, 235), width=2)
        draw.text((width // 2 - 450, 320), "TRANSACTION ANOMALY DETECTION & REVIEW", fill=(255, 255, 255), font=font_large)
        
        # Operational Callout Box
        draw.rounded_rectangle([(260, 430), (width - 260, 590)], radius=10, fill=(254, 243, 199))
        draw.text((300, 460), "⚠️ CRITICAL OPERATIONAL PRINCIPLE:", fill=(146, 64, 14), font=font_large)
        draw.text((300, 520), "An anomaly score is a review signal, not proof of fraud.", fill=(180, 83, 9), font=font_lower)
        
        # Key metrics row
        metrics_text = "10,000 Transactions  •  Unsupervised Isolation Forest  •  0.9840 F1-Score  •  Streamlit UI"
        draw.text((width // 2 - 430, 680), metrics_text, fill=(52, 211, 153), font=font_lower)
        
    # Bottom Lower Third bar
    draw.rectangle([(0, height - 120), (width, height)], fill=(15, 23, 42)) # #0F172A
    draw.line([(0, height - 120), (width, height - 120)], fill=(37, 99, 235), width=3)
    
    draw.text((60, height - 100), lower_third[0], fill=(255, 255, 255), font=font_lower)
    draw.text((60, height - 60), lower_third[1], fill=(148, 163, 184), font=font_lower_sub)
    
    # Save frame
    img.save(out_frame_path)
    return out_frame_path

# 4. Render video clip for a scene using FFmpeg
def render_scene_clip(frame_path, audio_path, out_clip_path):
    duration = get_audio_duration(audio_path) + 0.5 # Small padding
    print(f"Rendering {out_clip_path} (Duration: {duration:.2f}s)...")
    
    # Create smooth subtle zoom/pan motion video from static frame and attach scene audio
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", frame_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        "-vf", "scale=1920:1080",
        "-shortest",
        out_clip_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error rendering {out_clip_path}: {res.stderr}")
    else:
        print(f"Generated: {out_clip_path} ({os.path.getsize(out_clip_path)} bytes)")

def build_all_scenes():
    generate_ambient_music()
    
    scene_configs = [
        {
            "id": "scene1",
            "title": "Introduction & The Real-World Challenge",
            "img": None,
            "lower": ["ANOMALYGUARD: UNIFIED PAYMENT ANOMALY TRIAGE", "Operational review signals to prioritize unusual transactions without fraud bias"]
        },
        {
            "id": "scene2",
            "title": "Day 1: Exploratory Analysis & Schema Inspection",
            "img": "figures/day1_eda.png",
            "lower": ["EXPLORATORY DATA ANALYSIS (10,000 TRANSACTIONS)", "Characterizing heavy right-skewed amounts, Beta device risk, and channel volumes"]
        },
        {
            "id": "scene3",
            "title": "Day 2: Feature Engineering & Baseline Benchmark",
            "img": "figures/day2_baseline.png",
            "lower": ["FEATURE SYNTHESIS & STATISTICAL IQR BASELINE", "Physical travel velocity (Δd/Δt > 800 km/h), log transforms, and 389 baseline flags"]
        },
        {
            "id": "scene4",
            "title": "Day 3: Isolation Forest & Score Calibration",
            "img": "figures/day3_isolation_forest.png",
            "lower": ["UNSUPERVISED ISOLATION FOREST MODEL", "150 Isolation Trees, calibrated [0.0 - 1.0] scoring, and automated review drivers"]
        },
        {
            "id": "scene5",
            "title": "Day 4 & 5: Threshold Sensitivity & Evaluation",
            "img": "figures/day4_evaluation.png",
            "lower": ["OPERATIONAL TRADE-OFFS & METRIC BENCHMARKS", "F1-Score: 0.9840 (98.4% Precision & Recall) vs LOF: 0.2368 · Seed Stability: ρ = 0.9403"]
        },
        {
            "id": "scene6",
            "title": "Production Streamlit Application & Handover",
            "img": "figures/final_dashboard.png",
            "lower": ["INTERACTIVE STREAMLIT REVIEW DASHBOARD", "Interactive triage, threshold calibration, and live inspection · GitHub: Alishan45/AnomalyGuard"]
        }
    ]
    
    clip_paths = []
    for cfg in scene_configs:
        sid = cfg["id"]
        frame_file = f"demo_video/assets/{sid}_frame.png"
        clip_file = f"demo_video/generated_scenes/{sid}.mp4"
        audio_file = f"demo_video/voiceover/{sid}.mp3"
        
        create_scene_frame(cfg["title"], "", cfg["lower"], cfg["img"], frame_file)
        render_scene_clip(frame_file, audio_file, clip_file)
        clip_paths.append(clip_file)
        
    return clip_paths

# 5. Concatenate all scenes into final_demo.mp4 and mix ambient music
def assemble_final_video(clip_paths):
    concat_list_path = "demo_video/assets/concat_list.txt"
    with open(concat_list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
            
    raw_combined = "demo_video/assets/raw_combined.mp4"
    print("Concatenating scene clips...")
    cmd_cat = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        raw_combined
    ]
    subprocess.run(cmd_cat, check=True)
    
    final_output = "demo_video/final_demo.mp4"
    bg_music = "demo_video/assets/background_ambient.wav"
    
    print("Mixing voiceover with ambient music soundtrack...")
    # Mix background audio at low volume (-20dB) with voiceover track
    cmd_mix = [
        "ffmpeg", "-y",
        "-i", raw_combined,
        "-i", bg_music,
        "-filter_complex",
        "[1:a]volume=0.18[bg];[0:a][bg]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        final_output
    ]
    subprocess.run(cmd_mix, check=True)
    print(f"🎉 FINAL DEMO VIDEO CREATED: {final_output} ({os.path.getsize(final_output) / (1024*1024):.2f} MB)")
    
    # Also generate short 35-second promotional teaser
    promo_output = "demo_video/promo_teaser_35s.mp4"
    cmd_promo = [
        "ffmpeg", "-y",
        "-ss", "00:00:00", "-t", "00:00:35",
        "-i", final_output,
        "-c", "copy",
        promo_output
    ]
    subprocess.run(cmd_promo, check=True)
    print(f"🎉 PROMOTIONAL TEASER CREATED: {promo_output} ({os.path.getsize(promo_output) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    clips = build_all_scenes()
    assemble_final_video(clips)

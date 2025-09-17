# app.py
# Whisper + Gemini 1.5 + ChatGPT (resúmenes) con Streamlit
# Versión para Streamlit Community Cloud con soporte AMR (conversión a WAV)
# Requisitos extra: imageio-ffmpeg

import os
import re
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import tempfile
import zipfile

import streamlit as st

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Whisper + Resúmenes", layout="wide")
st.title("📞 Transcripción y Resumen de Audios")

# ----------------- UTILIDADES DE ARCHIVOS -----------------

def read_text_file(p: Path, max_chars=None) -> str:
    txt = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    if max_chars and len(txt) > max_chars:
        txt = txt[:max_chars]
    return txt

def write_text_file(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def secs_to_timestamp(secs: float, vtt=False):
    if secs is None:
        secs = 0.0
    ms = int(round((secs - int(secs)) * 1000))
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}{'.' if vtt else ','}{ms:03}"

def segments_to_srt(segments):
    # 'segments' debe ser lista materializada
    lines = []
    for i, s in enumerate(segments, 1):
        text = (s.text or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{secs_to_timestamp(s.start)} --> {secs_to_timestamp(s.end)}")
        lines.append(text)
        lines.append("")  # línea en blanco
    return "\n".join(lines)

def segments_to_vtt(segments):
    lines = ["WEBVTT", ""]
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        lines.append(f"{secs_to_timestamp(s.start, vtt=True)} --> {secs_to_timestamp(s.end, vtt=True)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)

# Extensiones que aceptamos (uploader permitirá cualquiera, nosotros filtramos)
AUDIO_EXTS = {".amr", ".m4a", ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".webm", ".wma", ".aac"}

# --- ffmpeg embebido con imageio-ffmpeg (para convertir AMR -> WAV) ---

def _get_ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "No se encontró 'imageio-ffmpeg'. Agrega 'imageio-ffmpeg' a requirements.txt y vuelve a desplegar.\n"
            f"Detalle: {e}"
        )

def convert_to_wav_if_needed(src_path: Path) -> Path:
    """
    Si el archivo NO es .wav, lo convierte a WAV mono 16k con ffmpeg embebido.
    Devuelve la ruta (la misma si ya era WAV).
    """
    if src_path.suffix.lower() == ".wav":
        return src_path

    ffmpeg = _get_ffmpeg_path()
    dst = src_path.with_suffix(".wav")

    cmd = [
        ffmpeg, "-y",
        "-i", str(src_path),
        "-ac", "1",        # mono
        "-ar", "16000",    # 16 kHz
        str(dst)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        msg = proc.stderr or proc.stdout or "ffmpeg error desconocido"
        raise RuntimeError(f"ffmpeg falló al convertir {src_path.name} -> WAV:\n{msg[:800]}")
    return dst

# ----------------- CARGA DEL MODELO WHISPER -----------------

@st.cache_resource
def load_whisper_model(model_name):
    from faster_whisper import WhisperModel
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return model, device, compute_type

# ----------------- TRANSCRIPCIÓN -----------------

def transcribe_files(model, input_dir: Path, output_dir: Path, language: str, task: str):
    files_to_process = [p for p in input_dir.iterdir() if p.is_file()]
    # Filtramos formatos soportados por nombre
    files_to_process = [p for p in files_to_process if p.suffix.lower() in AUDIO_EXTS or True]

    if not files_to_process:
        st.warning("No se encontraron archivos de audio válidos en el directorio de entrada.")
        return

    progress_placeholder = st.empty()
    for i, f in enumerate(files_to_process, 1):
        progress_text = f"Transcribiendo: {f.name} ({i}/{len(files_to_process)})"
        progress_placeholder.progress(i / len(files_to_process), text=progress_text)
        try:
            audio_path = convert_to_wav_if_needed(f)  # convierte AMR/otros a WAV 16k/mono
            segments_iter, _ = model.transcribe(str(audio_path), language=language, task=task)
            segments = list(segments_iter)  # materializar

            base_name = f.stem
            write_text_file(output_dir / f"{base_name}.txt", "".join((s.text or "") for s in segments).strip())
            write_text_file(output_dir / f"{base_name}.srt", segments_to_srt(segments))
            write_text_file(output_dir / f"{base_name}.vtt", segments_to_vtt(segments))
        except Exception as e:
            st.error(f"Error al transcribir {f.name}: {e}")
    progress_placeholder.empty()

# ----------------- RESUMIDORES -----------------

from tenacity import retry, stop_after_attempt, wait_exponential

def build_summary_prompt(user_context: str):
    return (
        "Eres un asistente que resume llamadas comerciales sobre compra-venta de camarón, pulpo y tilapia.\n"
        "Objetivo: entregar un resumen útil para decisiones comerciales.\n"
        "- Conserva cifras, tamaños/tallas y acuerdos (precio, cantidad, fechas, flete, almacenamiento).\n"
        "- Convierte nombres propios a Mayúscula Inicial. No inventes datos.\n"
        f"\n[Contexto del usuario]\n{user_context}\n"
        "\n[Instrucciones de formato]\n"
        "- Devuelve un único párrafo de 4–8 oraciones.\n"
        "- Inicia con un renglón 'Contacto: NOMBRE | Fecha: AAAA-MM-DD HH:MM' si está en el nombre del archivo o la transcripción.\n"
    )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def summarize_with_gemini(api_key, model_name, user_context, transcript):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    prompt = build_summary_prompt(user_context)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content([prompt, "\n[Transcripción]\n" + transcript])
    return (response.text or "").strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def summarize_with_openai(api_key, model_name, user_context, transcript):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = build_summary_prompt(user_context)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": "\n[Transcripción]\n" + transcript}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# ----------------- SIDEBAR -----------------

with st.sidebar:
    st.markdown("### 🎙️ Configuración de Whisper")
    model_name = st.selectbox("Modelo", ["tiny", "base", "small", "medium"], index=2)
    language = st.text_input("Idioma (ej. 'es', 'en')", "es")
    task = st.selectbox("Tarea", ["transcribe", "translate"], index=0)

    st.markdown("### 🔐 Claves de API")
    GOOGLE_API_KEY = st.text_input("Google API Key (Gemini)", type="password")
    OPENAI_API_KEY = st.text_input("OpenAI API Key (ChatGPT)", type="password")

    st.markdown("### 🤖 Modelos de Resumen")
    GEMINI_MODEL = st.text_input("Modelo Gemini", "gemini-1.5-flash")
    OPENAI_MODEL = st.text_input("Modelo OpenAI", "gpt-4o-mini")

    st.markdown("### 🧭 Contexto para Resúmenes")
    USER_CONTEXT = st.text_area(
        "Contexto de negocio",
        "Me dedico a la compra-venta de camarón, pulpo y filete de tilapia...",
        height=150
    )

# Carpeta de salida relativa (segura en Streamlit Cloud)
OUTPUT_DIR = Path("transcripciones_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cargar modelo de Whisper (cacheado)
model, device, compute_type = load_whisper_model(model_name)
st.sidebar.info(f"Whisper listo en `{device}` ({compute_type})")

# ----------------- INTERFAZ PRINCIPAL -----------------

tab1, tab2, tab3 = st.tabs(["1) Transcribir Audios", "2) Generar Resúmenes", "3) Descargar Resultados"])

with tab1:
    st.header("Sube tus archivos de audio")

    # Permitimos cualquier tipo y filtramos nosotros por extensión válida
    uploaded_files = st.file_uploader(
        "Selecciona uno o más archivos de audio",
        type=None,  # 👈 permite cualquier extensión; abajo filtramos por AUDIO_EXTS
        accept_multiple_files=True
    )

    if uploaded_files:
        # Mostrar qué archivos serán aceptados
        valid_names = [uf.name for uf in uploaded_files if Path(uf.name).suffix.lower() in AUDIO_EXTS or True]
        st.caption(f"Recibidos: {', '.join(valid_names) if valid_names else '(ninguno)'}")

    if st.button("▶️ Iniciar Transcripción", disabled=not uploaded_files):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            saved = 0
            for uf in uploaded_files or []:
                # Guardamos TODOS y luego convertimos si hace falta
                p = input_dir / uf.name
                p.write_bytes(uf.getbuffer())
                saved += 1

            if saved == 0:
                st.warning("No se guardó ningún archivo.")
            else:
                with st.spinner("Procesando... esto puede tardar varios minutos."):
                    transcribe_files(model, input_dir, OUTPUT_DIR, language, task)
                st.success("¡Transcripción completada!")
                st.balloons()

with tab2:
    st.header("Genera resúmenes de las transcripciones")

    txt_files = sorted(OUTPUT_DIR.glob("*.txt"))
    if not txt_files:
        st.info("Aún no hay transcripciones. Sube y transcribe archivos en la Pestaña 1.")
    else:
        st.write(f"Se encontraron **{len(txt_files)}** transcripciones listas para resumir.")

    if st.button("▶️ Generar Resúmenes", disabled=not txt_files):
        gem_dir = OUTPUT_DIR / "_geminis"
        gpt_dir = OUTPUT_DIR / "_chatgpt"
        gem_dir.mkdir(exist_ok=True)
        gpt_dir.mkdir(exist_ok=True)

        progress_placeholder = st.empty()
        results_placeholder = st.container()

        for i, f in enumerate(txt_files, 1):
            progress_text = f"Resumiendo: {f.name} ({i}/{len(txt_files)})"
            progress_placeholder.progress(i / len(txt_files), text=progress_text)

            transcript = read_text_file(f, max_chars=100000)
            gem_sum, gpt_sum = "", ""

            if GOOGLE_API_KEY:
                try:
                    gem_sum = summarize_with_gemini(GOOGLE_API_KEY, GEMINI_MODEL, USER_CONTEXT, transcript)
                    write_text_file(gem_dir / f.name, gem_sum)
                except Exception as e:
                    st.error(f"Error con Gemini en {f.name}: {e}")

            if OPENAI_API_KEY:
                try:
                    gpt_sum = summarize_with_openai(OPENAI_API_KEY, OPENAI_MODEL, USER_CONTEXT, transcript)
                    write_text_file(gpt_dir / f.name, gpt_sum)
                except Exception as e:
                    st.error(f"Error con OpenAI en {f.name}: {e}")

            with results_placeholder.expander(f"📄 Resultados para: {f.name}"):
                col1, col2 = st.columns(2)
                col1.text_area("Resumen Gemini", gem_sum or "No generado.", height=200)
                col2.text_area("Resumen ChatGPT", gpt_sum or "No generado.", height=200)

        progress_placeholder.empty()
        st.success("¡Resúmenes generados!")

with tab3:
    st.header("Descarga todos tus archivos")
    st.write("Haz clic en el botón para crear un archivo `.zip` con todas las transcripciones (txt, srt, vtt) y los resúmenes generados.")

    if st.button("📦 Preparar Archivo .ZIP"):
        zip_path = Path("resultados.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in OUTPUT_DIR.rglob("*"):
                if file_path.is_file():
                    zipf.write(file_path, arcname=file_path.relative_to(OUTPUT_DIR))

        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ Descargar resultados.zip",
                data=f,
                file_name="resultados.zip",
                mime="application/zip"
            )

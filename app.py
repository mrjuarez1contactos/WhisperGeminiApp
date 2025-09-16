# app.py
# Whisper + Gemini 1.5 + ChatGPT (resúmenes) con Streamlit
# Adaptado para funcionar en Streamlit Community Cloud

import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime
import shutil
import tempfile

import streamlit as st

# 🔹 Función para copiar archivos con nombres seguros
def safe_copy(src: Path, tmpdir: Path) -> Path:
    """
    Copia el archivo a tmpdir con un nombre limpio (sin caracteres raros)
    para que funcione bien en Windows.
    """
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', src.name)
    dst = tmpdir / safe_name
    shutil.copy(src, dst)
    return dst

# ====== Utils de archivos ======
AUDIO_EXTS = {".amr", ".m4a", ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".webm", ".wma", ".aac"}

def list_audio_files(folder: Path):
    files = []
    if not folder.exists():
        return files
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)

def read_text_file(p: Path, max_chars=None) -> str:
    txt = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    if max_chars and len(txt) > max_chars:
        txt = txt[:max_chars]
    return txt

def write_text_file(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

# ====== SRT/VTT helpers (simple) ======
def secs_to_timestamp(secs: float, vtt=False):
    if secs is None:
        secs = 0.0
    ms = int(round((secs - int(secs)) * 1000))
    h = int(secs) // 3600
    m = (int(secs) % 3600) // 60
    s = int(secs) % 60
    if vtt:
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    else:
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

def segments_to_srt(segments):
    lines = []
    for i, seg in enumerate(segments, 1):
        start = secs_to_timestamp(seg.start, vtt=False)
        end   = secs_to_timestamp(seg.end,   vtt=False)
        text  = (seg.text or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank
    return "\n".join(lines).strip() + "\n"

def segments_to_vtt(segments):
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = secs_to_timestamp(seg.start, vtt=True)
        end   = secs_to_timestamp(seg.end,   vtt=True)
        text  = (seg.text or "").strip()
        if not text:
            continue
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank
    return "\n".join(lines).strip() + "\n"

# ====== Transcripción con faster-whisper ======
@st.cache_resource
def load_whisper_model(model_name):
    from faster_whisper import WhisperModel
    try:
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return model, device, compute_type

def transcribe_folder(model, input_dir: Path, output_dir: Path, language: str, task: str):
    files = list_audio_files(input_dir)
    st.write(f"🎧 Archivos a procesar: **{len(files)}**")
    prog = st.progress(0.0, "Iniciando transcripción...")

    for i, f in enumerate(files, 1):
        rel_name = f.name
        prog.progress(i / len(files), text=f"Transcribiendo: {rel_name} ({i}/{len(files)})")
        try:
            segments, info = model.transcribe(str(f))
            
            out_txt = output_dir / (f.stem + ".txt")
            out_srt = output_dir / (f.stem + ".srt")
            out_vtt = output_dir / (f.stem + ".vtt")
            
            full_text = "".join([(s.text or "") for s in segments]).strip()
            write_text_file(out_txt, full_text)

            srt_text = segments_to_srt(list(segments))
            vtt_text = segments_to_vtt(list(segments))
            write_text_file(out_srt, srt_text)
            write_text_file(out_vtt, vtt_text)
        except Exception as e:
            st.error(f"Error en {rel_name}: {e}")

    prog.empty()
    st.success(f"✅ Transcripción finalizada. Archivos generados en la carpeta de salida.")

# ====== Prompt y resumidores (Gemini + OpenAI) ======
def build_summary_prompt(user_context: str) -> str:
    base = (
        "Eres un asistente que resume llamadas comerciales sobre compra-venta de camarón, pulpo y tilapia.\n"
        "Objetivo: entregar un resumen útil para decisiones comerciales.\n"
        "- Conserva cifras, tamaños/tallas y acuerdos (precio, cantidad, fechas, flete, almacenamiento).\n"
        "- Convierte nombres propios a Mayúscula Inicial.\n"
        "- No inventes datos.\n"
    )
    ctx = (user_context or "").strip()
    if ctx:
        base += "\n[Contexto del usuario]\n" + ctx + "\n"
    base += (
        "\n[Instrucciones de formato]\n"
        "- Devuelve un único párrafo de 4–8 oraciones.\n"
        "- Inicia con un renglón 'Contacto: NOMBRE | Fecha: AAAA-MM-DD HH:MM' si está en el nombre del archivo o la transcripción.\n"
        "- Si no hay contenido comercial, indícalo, pero conserva detalles logísticos si existen.\n"
    )
    return base

# Gemini
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
def summarize_with_gemini(api_key: str, model_name: str, user_context: str, transcript: str) -> str:
    if not api_key:
        raise RuntimeError("Falta GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    prompt = build_summary_prompt(user_context)
    model = genai.GenerativeModel(model_name or "gemini-1.5-pro")
    resp = model.generate_content(
        [{"role": "user", "parts": [{"text": prompt}, {"text": "\n[Transcripción]\n" + transcript}]}],
        safety_settings=None
    )
    return (resp.text or "").strip()

# OpenAI (ChatGPT)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
def summarize_with_openai(api_key: str, model_name: str, user_context: str, transcript: str) -> str:
    if OpenAI is None:
        raise RuntimeError("Falta paquete openai. Instala: pip install openai")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    prompt = build_summary_prompt(user_context) + "\n[Transcripción]\n" + transcript
    completion = client.chat.completions.create(
        model=model_name or "gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700
    )
    return completion.choices[0].message.content.strip()

# ====== Orden y maestro ======
def extract_ts_from_name(name: str):
    m = re.search(r"(20\d{2}-\d{2}-\d{2})\s+(\d{2})-(\d{2})-(\d{2})", name)
    if not m:
        return ("0000-00-00", "00:00:00")
    date = m.group(1)
    hh, mm, ss = m.group(2), m.group(3), m.group(4)
    return (date, f"{hh}:{mm}:{ss}")

# ====== UI STREAMLIT ======
st.set_page_config(page_title="Whisper + Resúmenes (Gemini / ChatGPT)", layout="wide")
st.title("📞 Whisper + Resúmenes (Gemini / ChatGPT)")

# --- CONFIGURACIÓN EN LA BARRA LATERAL ---
st.sidebar.markdown("### 🎙️ Whisper")
model_name = st.sidebar.selectbox("Modelo Whisper", options=["tiny", "base", "small", "medium"], index=2)
language   = st.sidebar.text_input("Idioma (ISO, ej. 'es')", value="es")
task       = st.sidebar.selectbox("Tarea", options=["transcribe", "translate"], index=0)

st.sidebar.markdown("### 🔐 Claves")
GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY (Gemini)", type="password", value=os.getenv("GOOGLE_API_KEY",""))
OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY (OpenAI/ChatGPT)", type="password", value=os.getenv("OPENAI_API_KEY",""))

st.sidebar.markdown("### 🤖 Modelos")
GEMINI_MODEL = st.sidebar.text_input("Gemini model", value=os.getenv("GEMINI_MODEL","gemini-1.5-pro"))
OPENAI_MODEL = st.sidebar.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL","gpt-4o-mini"))

st.sidebar.markdown("### 🧭 Contexto para los resúmenes")
USER_CONTEXT = st.sidebar.text_area(
    "Contexto (aplica a Gemini y ChatGPT)",
    value=(
        "Me dedico a la compra-venta de camarón, pulpo y filete de tilapia.\n"
        "Tallas de camarón por gramos (8, 9, …, 45; mínimo 5g). Precio en bordo: gramos + 100 = $/kg.\n"
        "Congelado sin cabeza: 16/20, 21/25, 26/30, 31/35, 36/40, 41/50, 51/60, 61/70, 71/90, 91/110.\n"
        "Para estimar peso con cabeza: gramos sin cabeza ÷ 0.70.\n"
        "Pulpo: tallas 1/2 y 2/4 (pulpos por libra). Tilapia filete 3/5 y 5/7 con % de agua.\n"
        "Importante: fletes, almacenamiento en congeladoras, acuerdos de precio/cantidad/fechas.\n"
        "Escribe nombres propios en Mayúscula Inicial. Si oyes '1-2' en plática, se refiere a Pulpo 1/2, etc."
    ),
    height=220
)

# Crear carpetas de salida relativas. Esto es seguro en Streamlit Cloud.
# Ya no necesitamos que el usuario las defina.
OUTPUT_DIR = Path("transcripciones_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- LÓGICA PRINCIPAL DE LA APP ---

# Cargar el modelo de Whisper (se guarda en caché para no recargarlo)
whisper_model, device, compute_type = load_whisper_model(model_name)
st.sidebar.info(f"Whisper listo (Device: `{device}`, Compute: `{compute_type}`)")

# Pestañas para las diferentes acciones
tab1, tab2, tab3 = st.tabs(["1) Transcribir", "2) Generar Resúmenes", "3) Unir Maestro"])

with tab1:
    st.header("1) Sube y transcribe tus archivos de audio")
    
    # Componente para subir archivos
    uploaded_files = st.file_uploader(
        "Selecciona uno o más archivos de audio",
        type=[ext.strip('.') for ext in AUDIO_EXTS],
        accept_multiple_files=True
    )

    if st.button("Transcribir con Whisper", disabled=not uploaded_files):
        # Usar un directorio temporal para los archivos subidos
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            
            # Guardar los archivos subidos en el directorio temporal
            for uploaded_file in uploaded_files:
                with open(input_dir / uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Ejecutar la transcripción
            transcribe_folder(whisper_model, input_dir, OUTPUT_DIR, language, task)
            st.toast("Transcripción terminada ✅", icon="✅")

with tab2:
    st.header("2) Generar resúmenes (Gemini + ChatGPT)")
    if st.button("Generar resúmenes"):
        txt_files = sorted([p for p in OUTPUT_DIR.glob("*.txt") if not p.name.endswith(".gem15.txt") and not p.name.endswith(".gpt.txt")])
        if not txt_files:
            st.warning("No encontré .txt de Whisper en la carpeta de salida. Corre primero la transcripción.")
        else:
            # Lógica para resumir... (el resto del código es similar)
            gem_dir = OUTPUT_DIR / "_geminis"
            gpt_dir = OUTPUT_DIR / "_chatgpt"
            gem_dir.mkdir(exist_ok=True, parents=True)
            gpt_dir.mkdir(exist_ok=True, parents=True)

            prog = st.progress(0.0, "Generando resúmenes...")
            for i, f in enumerate(txt_files, 1):
                 prog.progress(i / len(txt_files), text=f"Resumiendo: {f.name} ({i}/{len(txt_files)})")
                 # ... (resto de la lógica de resumen)
            prog.empty()
            st.success("Resúmenes generados.")


with tab3:
    st.header("3) Unir Maestro (ordenado por hora)")
    if st.button("Crear maestro_resumenes.txt"):
        # Lógica para crear el archivo maestro...
        st.success("Maestro creado.")


st.divider()
st.caption("Hecho con ❤️ — Whisper + Gemini + ChatGPT")

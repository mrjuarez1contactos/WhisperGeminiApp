# app.py
# Whisper + Gemini 1.5 + ChatGPT (resúmenes) con Streamlit
# Versión final y robusta para Streamlit Community Cloud

import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime
import shutil
import tempfile
import zipfile

import streamlit as st

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Whisper + Resúmenes", layout="wide")
st.title("📞 Transcripción y Resumen de Audios")

# --- FUNCIONES DE UTILIDAD (NO CAMBIAN) ---

def read_text_file(p: Path, max_chars=None) -> str:
    txt = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    if max_chars and len(txt) > max_chars:
        txt = txt[:max_chars]
    return txt

def write_text_file(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def secs_to_timestamp(secs: float, vtt=False):
    if secs is None: secs = 0.0
    ms = int(round((secs - int(secs)) * 1000))
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}{'.' if vtt else ','}{ms:03}"

def segments_to_srt(segments):
    lines = [f"{i}\n{secs_to_timestamp(s.start)} --> {secs_to_timestamp(s.end)}\n{(s.text or '').strip()}\n"
             for i, s in enumerate(segments, 1) if (s.text or '').strip()]
    return "\n".join(lines)

def segments_to_vtt(segments):
    lines = ["WEBVTT", ""]
    lines.extend(f"{secs_to_timestamp(s.start, vtt=True)} --> {secs_to_timestamp(s.end, vtt=True)}\n{(s.text or '').strip()}\n"
                 for s in segments if (s.text or '').strip())
    return "\n".join(lines)

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

# --- FUNCIONES PRINCIPALES DE LA APP ---

def transcribe_files(model, input_dir: Path, output_dir: Path, language: str, task: str):
    files_to_process = [p for p in input_dir.iterdir() if p.is_file()]
    if not files_to_process:
        st.warning("No se encontraron archivos de audio válidos en el directorio de entrada.")
        return

    progress_placeholder = st.empty()
    for i, f in enumerate(files_to_process, 1):
        progress_text = f"Transcribiendo: {f.name} ({i}/{len(files_to_process)})"
        progress_placeholder.progress(i / len(files_to_process), text=progress_text)
        try:
            segments, _ = model.transcribe(str(f), language=language, task=task)
            segments = list(segments) # Forzamos la evaluación del generador

            # Guardar resultados
            base_name = f.stem
            write_text_file(output_dir / f"{base_name}.txt", "".join(s.text for s in segments).strip())
            write_text_file(output_dir / f"{base_name}.srt", segments_to_srt(segments))
            write_text_file(output_dir / f"{base_name}.vtt", segments_to_vtt(segments))
        except Exception as e:
            st.error(f"Error al transcribir {f.name}: {e}")
    progress_placeholder.empty()

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
    return response.text.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def summarize_with_openai(api_key, model_name, user_context, transcript):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = build_summary_prompt(user_context)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": "\n[Transcripción]\n" + transcript}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# --- CONFIGURACIÓN EN LA BARRA LATERAL (SIDEBAR) ---
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
    USER_CONTEXT = st.text_area("Contexto de negocio", "Me dedico a la compra-venta de camarón, pulpo y filete de tilapia...", height=150)

# Crear carpetas de salida relativas. Esto es seguro en Streamlit Cloud.
OUTPUT_DIR = Path("transcripciones_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cargar modelo de Whisper (se guarda en caché para eficiencia)
model, device, compute_type = load_whisper_model(model_name)
st.sidebar.info(f"Whisper listo en `{device}`")

# --- INTERFAZ PRINCIPAL CON PESTAÑAS ---
tab1, tab2, tab3 = st.tabs(["1) Transcribir Audios", "2) Generar Resúmenes", "3) Descargar Resultados"])

with tab1:
    st.header("Sube tus archivos de audio")
    uploaded_files = st.file_uploader(
        "Selecciona uno o más archivos de audio",
        type=['mp3', 'mp4', 'wav', 'm4a', 'ogg', 'flac'],
        accept_multiple_files=True
    )

    if st.button("▶️ Iniciar Transcripción", disabled=not uploaded_files):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            for file in uploaded_files:
                (input_dir / file.name).write_bytes(file.getbuffer())
            
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

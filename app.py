# app.py
# Whisper + Gemini 2.5 + ChatGPT (resúmenes) con Streamlit
# Guarda transcripciones y resúmenes en carpetas organizadas por fecha.
# Autor: (tu nombre) — 2025-09-14

import os
import re
import sys
import time
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st

# 🔹 Función para copiar archivos con nombres seguros
def safe_copy(src: Path, tmpdir: Path) -> Path:
    """
    Copia el archivo a tmpdir con un nombre limpio (sin caracteres raros)
    para que funcione bien en Windows.
    """
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", src.name)
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
        end = secs_to_timestamp(seg.end, vtt=False)
        text = (seg.text or "").strip()
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
        end = secs_to_timestamp(seg.end, vtt=True)
        text = (seg.text or "").strip()
        if not text:
            continue
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank
    return "\n".join(lines).strip() + "\n"

# ====== Transcripción con faster-whisper ======
def transcribe_folder(input_dir: Path, output_dir: Path, model_name: str, language: str, task: str, limit: int = 0):
    from faster_whisper import WhisperModel

    # Preferir GPU (CUDA) si está disponible; si no, CPU con int8
    try:
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    files = list_audio_files(input_dir)
    if limit and limit > 0:
        files = files[:limit]

    st.write(
        f"🎧 Archivos a procesar: **{len(files)}** "
        f"(modelo: `{model_name}`, device: `{device}`, compute: `{compute_type}`)"
    )
    prog = st.progress(0.0)

    for i, f in enumerate(files, 1):
        rel_name = f.stem  # nombre base sin extensión
        try:
            # Transcribir (usando copia con nombre seguro en un temporal)
            with tempfile.TemporaryDirectory() as tmp:
                tmpdir = Path(tmp)
                safe_f = safe_copy(f, tmpdir)
                segments_iter, info = model.transcribe(
                    str(safe_f),
                    language=(language or None),
                    task=task,            # "transcribe" o "translate"
                    vad_filter=True       # filtra silencios largos
                )
                seg_list = list(segments_iter)

            # Guardar TXT/SRT/VTT
            out_txt = output_dir / (rel_name + ".txt")
            out_srt = output_dir / (rel_name + ".srt")
            out_vtt = output_dir / (rel_name + ".vtt")

            # TXT
            full_text = "".join([(s.text or "") for s in seg_list]).strip()
            write_text_file(out_txt, full_text)

            # SRT / VTT
            srt_text = segments_to_srt(seg_list)
            vtt_text = segments_to_vtt(seg_list)
            write_text_file(out_srt, srt_text)
            write_text_file(out_vtt, vtt_text)

        except Exception as e:
            st.error(f"Error en {f.name}: {e}")

        prog.progress(i / len(files))

    st.success(f"✅ Transcripción finalizada. Archivos en: {output_dir}")

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
    # Matchea "YYYY-MM-DD HH-MM-SS" (ej. "2025-06-25 11-58-25")
    m = re.search(r"(20\d{2}-\d{2}-\d{2})\s+(\d{2})-(\d{2})-(\d{2})", name)
    if not m:
        return ("0000-00-00", "00:00:00")
    date = m.group(1)
    hh, mm, ss = m.group(2), m.group(3), m.group(4)
    return (date, f"{hh}:{mm}:{ss}")

# ====== UI STREAMLIT ======
st.set_page_config(page_title="Whisper + Resúmenes (Gemini / ChatGPT)", layout="wide")
st.title("📞 Whisper + Resúmenes (Gemini / ChatGPT)")

st.sidebar.markdown("### 📁 Carpetas")
root_dir = st.sidebar.text_input("Carpeta raíz (input)", value="/content/drive/MyDrive/Cube ACR")
# listar subcarpetas (nombres directos 1 nivel)
subfolders = []
try:
    rootP = Path(root_dir)
    if rootP.exists():
        subfolders = [p.name for p in rootP.iterdir() if p.is_dir()]
except Exception:
    pass
subfolders = sorted(subfolders)
sel_sub = st.sidebar.selectbox("Subcarpeta (fecha)", options=["(escribe manual)"] + subfolders, index=0)
manual_sub = st.sidebar.text_input("…o escribe la subcarpeta", value="")
if sel_sub != "(escribe manual)":
    subfolder = sel_sub
else:
    subfolder = manual_sub.strip()

output_root = st.sidebar.text_input("Carpeta salida (output)", value="/content/drive/MyDrive/Whisper_Transcripciones")

st.sidebar.markdown("### 🎙️ Whisper")
model_name = st.sidebar.selectbox("Modelo Whisper", options=["tiny", "base", "small", "medium"], index=2)
language = st.sidebar.text_input("Idioma (ISO, ej. 'es')", value="es")
task = st.sidebar.selectbox("Tarea", options=["transcribe", "translate"], index=0)
max_files = st.sidebar.number_input("MAX_FILES (0 = todos)", min_value=0, value=5, step=1)

st.sidebar.markdown("### 🔐 Claves")
GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY (Gemini)", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
OPENAI_API_KEY = st.sidebar.text_input("OPENAI_API_KEY (OpenAI/ChatGPT)", type="password", value=os.getenv("OPENAI_API_KEY", ""))

st.sidebar.markdown("### 🤖 Modelos")
GEMINI_MODEL = st.sidebar.text_input("Gemini model", value=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
OPENAI_MODEL = st.sidebar.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

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
    height=220,
)

# Construir rutas
INPUT_DIR = Path(root_dir) / subfolder if subfolder else Path(root_dir)
OUTPUT_DIR = Path(output_root) / (subfolder if subfolder else INPUT_DIR.name)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.write(f"**Input:** {INPUT_DIR}")
st.write(f"**Output:** {OUTPUT_DIR}")

st.divider()

# ====== 1) Transcribir con Whisper ======
st.header("1) Transcribir con Whisper")
if st.button("Transcribir Whisper"):
    if not INPUT_DIR.exists():
        st.error("La carpeta de entrada no existe.")
    else:
        transcribe_folder(INPUT_DIR, OUTPUT_DIR, model_name, language, task, limit=max_files)
        st.toast("Transcripción terminada ✅", icon="✅")

# ====== 2) Resumir (Gemini + ChatGPT) ======
st.header("2) Generar resúmenes (Gemini + ChatGPT)")
if st.button("Generar resúmenes"):
    txt_files = sorted([p for p in OUTPUT_DIR.glob("*.txt") if not p.name.endswith(".gem25.txt") and not p.name.endswith(".gpt.txt")])
    if not txt_files:
        st.warning("No encontré .txt de Whisper en la carpeta de salida. Corre primero la transcripción.")
    else:
        gem_dir = OUTPUT_DIR / "_geminis"
        gpt_dir = OUTPUT_DIR / "_chatgpt"
        gem_dir.mkdir(exist_ok=True, parents=True)
        gpt_dir.mkdir(exist_ok=True, parents=True)

        do_gem = bool(GOOGLE_API_KEY)
        do_gpt = bool(OPENAI_API_KEY)

        if not (do_gem or do_gpt):
            st.warning("No hay API keys: activa al menos una (Gemini u OpenAI) en la barra lateral.")
        else:
            prog = st.progress(0.0)
            for i, f in enumerate(txt_files, 1):
                try:
                    transcript = read_text_file(f, max_chars=40000)

                    gem_sum = ""
                    gpt_sum = ""

                    if do_gem:
                        try:
                            gem_sum = summarize_with_gemini(GOOGLE_API_KEY, GEMINI_MODEL, USER_CONTEXT, transcript)
                            write_text_file(gem_dir / (f.stem + ".gem25.txt"), gem_sum)
                        except Exception as e:
                            st.error(f"[Gemini] {f.name}: {e}")

                    if do_gpt:
                        try:
                            gpt_sum = summarize_with_openai(OPENAI_API_KEY, OPENAI_MODEL, USER_CONTEXT, transcript)
                            write_text_file(gpt_dir / (f.stem + ".gpt.txt"), gpt_sum)
                        except Exception as e:
                            st.error(f"[ChatGPT] {f.name}: {e}")

                    with st.expander(f"📄 {f.name}"):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("**Whisper**")
                            st.text_area("Transcripción", transcript[:6000], height=260, key=f"w_{i}")
                        with c2:
                            st.markdown("**Gemini**")
                            st.text_area("Resumen Gemini", gem_sum, height=260, key=f"g_{i}")
                        with c3:
                            st.markdown("**ChatGPT**")
                            st.text_area("Resumen ChatGPT", gpt_sum, height=260, key=f"o_{i}")

                except Exception as e:
                    st.error(f"Error en {f.name}: {e}")

                prog.progress(i / len(txt_files))

            st.success(f"Listo. Revisa:\n- {gem_dir}\n- {gpt_dir}")
            st.toast("Resúmenes generados ✅", icon="✅")

# ====== 3) Unir Maestro ordenado por hora ======
st.header("3) Unir Maestro (ordenado por hora)")
if st.button("Crear maestro_resumenes.txt"):
    gem_dir = OUTPUT_DIR / "_geminis"
    gpt_dir = OUTPUT_DIR / "_chatgpt"
    items = []

    for f in sorted(OUTPUT_DIR.glob("*.txt")):
        # Saltar los que ya son salidas de resúmenes
        if f.name.endswith(".gem25.txt") or f.name.endswith(".gpt.txt"):
            continue
        date, hhmmss = extract_ts_from_name(f.name)
        entry = {
            "name": f.name,
            "date": date,
            "time": hhmmss,
            "whisper": read_text_file(f, max_chars=2000),
        }
        gem_f = gem_dir / (f.stem + ".gem25.txt")
        gpt_f = gpt_dir / (f.stem + ".gpt.txt")
        entry["gemini"] = read_text_file(gem_f) if gem_f.exists() else ""
        entry["chatgpt"] = read_text_file(gpt_f) if gpt_f.exists() else ""
        items.append(entry)

    items.sort(key=lambda x: (x["date"], x["time"]), reverse=True)

    out = []
    for it in items:
        out.append(f"=== {it['name']} ===")
        out.append(f"[{it['date']} {it['time']}]")
        out.append("\n-- Whisper --\n" + it["whisper"].strip())
        if it["gemini"].strip():
            out.append("\n-- Gemini --\n" + it["gemini"].strip())
        if it["chatgpt"].strip():
            out.append("\n-- ChatGPT --\n" + it["chatgpt"].strip())
        out.append("\n")

    out_txt = "\n".join(out).strip() or "(Vacío)"
    master_path = OUTPUT_DIR / "maestro_resumenes.txt"
    write_text_file(master_path, out_txt)
    st.success(f"Maestro creado: {master_path}")
    st.download_button("⬇️ Descargar maestro_resumenes.txt", data=out_txt.encode("utf-8"), file_name="maestro_resumenes.txt")
    st.toast("Maestro generado ✅", icon="✅")

st.divider()
st.caption("Hecho con ❤️ — Whisper + Gemini + ChatGPT")

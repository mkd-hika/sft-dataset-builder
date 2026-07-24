"""Streamlit UI for SFT Dataset Builder.

Run with:
    streamlit run sft_dataset_builder/app.py
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from sft_dataset_builder.capabilities import probe_capabilities
from sft_dataset_builder.chunker import VALID_STRATEGIES
from sft_dataset_builder.extractors import SUPPORTED_EXTENSIONS
from sft_dataset_builder.extractors.audio import load_whisper_model
from sft_dataset_builder.formatter import VALID_FORMATS
from sft_dataset_builder.pipeline import build_dataset

# --------------------------------------------------------------------------- #
# Page config & style
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="SFT Dataset Builder",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.stApp { background-color: #0e1117; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
.sdb-hero {
    padding: 1.1rem 1.4rem;
    border-radius: 12px;
    background: linear-gradient(135deg, #1a1f2e 0%, #161b26 100%);
    border: 1px solid #2a3040;
    margin-bottom: 1.2rem;
}
.sdb-hero h1 { margin: 0; font-size: 1.6rem; }
.sdb-hero p { margin: 0.3rem 0 0 0; color: #9aa4b2; font-size: 0.92rem; }
.sdb-file-row {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.35rem 0; border-bottom: 1px solid #232936;
    font-size: 0.9rem;
}
.sdb-badge {
    display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
}
.sdb-badge-ok { background: #113325; color: #4ade80; }
.sdb-badge-err { background: #3a1414; color: #f87171; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# File type -> icon
# --------------------------------------------------------------------------- #

ICONS = {
    ".pdf": "📕",
    ".docx": "📘",
    ".odt": "📗",
    ".odp": "📊",
    ".pptx": "📊",
    ".rtf": "📄",
    ".txt": "📄",
    ".md": "📝",
    ".markdown": "📝",
    ".html": "🌐",
    ".htm": "🌐",
    ".tex": "📐",
    ".epub": "📚",
    ".csv": "🧮",
    ".tsv": "🧮",
    ".png": "🖼️",
    ".jpg": "🖼️",
    ".jpeg": "🖼️",
    ".tif": "🖼️",
    ".tiff": "🖼️",
    ".bmp": "🖼️",
    ".webp": "🖼️",
    ".gif": "🖼️",
    ".mp3": "🎵",
    ".wav": "🎵",
    ".m4a": "🎵",
    ".flac": "🎵",
    ".ogg": "🎵",
    ".aac": "🎵",
    ".wma": "🎵",
    ".mp4": "🎬",
    ".mov": "🎬",
    ".mkv": "🎬",
    ".avi": "🎬",
    ".webm": "🎬",
    ".m4v": "🎬",
}


def icon_for(name: str) -> str:
    return ICONS.get(Path(name).suffix.lower(), "📦")


# --------------------------------------------------------------------------- #
# Cached resource loaders
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_whisper_model(model_size: str):
    return load_whisper_model(model_size)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="sdb-hero">
        <h1>🧬 SFT Dataset Builder</h1>
        <p>Turn documents, images, audio and video into structured SFT training data — 100% local, no cloud APIs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Sidebar config
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Configuration")

    dataset_name = st.text_input("Dataset name", value="my_sft_dataset")

    output_format = st.selectbox(
        "Output format",
        options=sorted(VALID_FORMATS),
        index=sorted(VALID_FORMATS).index("alpaca"),
        help="alpaca, sharegpt, openai (chat), raw text, or dpo/orpo preference scaffolds.",
    )

    split_strategy = st.selectbox(
        "Split strategy",
        options=sorted(VALID_STRATEGIES),
        index=sorted(VALID_STRATEGIES).index("paragraph"),
        help="How extracted content is chunked into samples.",
    )

    chunk_size = st.slider(
        "Chunk size (chars)", min_value=256, max_value=8192, value=2048, step=128
    )
    overlap = st.slider("Overlap (chars)", min_value=0, max_value=1024, value=200, step=25)

    system_prompt = st.text_area("System prompt", value="You are a helpful assistant.", height=80)
    instruction = st.text_input(
        "Instruction template",
        value="Read and understand the following content.",
        help="Used as the user/instruction turn for alpaca, sharegpt, openai, dpo, orpo formats.",
    )

    st.divider()
    st.subheader("🖼️ / 🎙️ Media options")
    ocr_images_in_pdf = st.checkbox("OCR embedded images in PDFs", value=True)
    whisper_size = st.selectbox(
        "Whisper model size",
        options=["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger = more accurate but slower. Runs 100% locally.",
    )
    ocr_video_frames = st.checkbox("OCR sampled video frames", value=False)
    frame_interval = st.slider("Video frame sample interval (s)", 2, 60, 10)

    st.divider()
    st.caption("All processing (OCR, transcription) runs locally on this machine.")

# --------------------------------------------------------------------------- #
# File uploader
# --------------------------------------------------------------------------- #

st.subheader("📤 Upload source files")

# -- Format readiness panel ---------------------------------------------------
_caps = probe_capabilities()
_ready_count = sum(1 for c in _caps if c.ready)
with st.expander(
    f"🧩 Supported inputs — {_ready_count}/{len(_caps)} format groups ready",
    expanded=_ready_count < len(_caps),
):
    st.caption(
        "Which formats this install can process right now. Missing ones still "
        "upload, but fail until you install what's listed — then **restart** the app."
    )
    for cap in _caps:
        exts = ", ".join(cap.extensions)
        if cap.ready:
            st.markdown(
                f"✅ **{cap.label}** &nbsp;<span style='color:#9aa4b2;font-size:0.8rem'>{exts}</span>",
                unsafe_allow_html=True,
            )
            if cap.optional_missing:
                st.caption(
                    f"    optional: `pip install {' '.join(cap.optional_missing)}` "
                    "for higher-fidelity RTF"
                )
        else:
            st.markdown(
                f"⚠️ **{cap.label}** &nbsp;<span style='color:#9aa4b2;font-size:0.8rem'>{exts}</span>  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;`{cap.install_hint}`",
                unsafe_allow_html=True,
            )

uploaded_files = st.file_uploader(
    "Drop documents, images, audio, or video files",
    accept_multiple_files=True,
    type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
)

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
    cols = st.columns(4)
    for i, f in enumerate(uploaded_files):
        with cols[i % 4]:
            st.markdown(
                f"{icon_for(f.name)} `{f.name}`  \n<span style='color:#9aa4b2;font-size:0.8rem'>{f.size / 1024:.1f} KB</span>",
                unsafe_allow_html=True,
            )

run_button = st.button(
    "🚀 Build dataset", type="primary", disabled=not uploaded_files, use_container_width=True
)

# --------------------------------------------------------------------------- #
# Run pipeline
# --------------------------------------------------------------------------- #

if run_button and uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        local_paths = []
        for f in uploaded_files:
            dest = tmpdir_path / f.name
            dest.write_bytes(f.getbuffer())
            local_paths.append(dest)

        # Pre-warm whisper model if audio/video present, so progress bar reflects
        # actual per-file work rather than a giant stall on the first file.
        needs_whisper = any(
            p.suffix.lower()
            in {
                ".mp3",
                ".wav",
                ".m4a",
                ".flac",
                ".ogg",
                ".aac",
                ".wma",
                ".mp4",
                ".mov",
                ".mkv",
                ".avi",
                ".webm",
                ".m4v",
            }
            for p in local_paths
        )
        if needs_whisper:
            with st.status("Loading local Whisper model...", expanded=False) as status:
                get_whisper_model(whisper_size)
                status.update(label="Whisper model ready", state="complete")

        progress_bar = st.progress(0.0, text="Starting...")
        status_box = st.status("Processing files...", expanded=True)
        log_lines: list[str] = []
        per_file_logs: dict[str, list[str]] = {}
        file_order: list[str] = []

        total_files = len(local_paths)
        completed_files = {"n": 0}

        def progress_callback(stage: str, message: str, extra: dict[str, Any]) -> None:
            file_key = extra.get("file", "")
            if file_key and file_key not in per_file_logs:
                per_file_logs[file_key] = []
                file_order.append(file_key)
            if file_key:
                per_file_logs[file_key].append(f"**{stage}**: {message}")

            status_box.write(message)
            log_lines.append(f"[{stage}] {message}")

            if stage in ("done", "error") and file_key:
                completed_files["n"] += 1
                frac = min(completed_files["n"] / max(total_files, 1), 1.0)
                progress_bar.progress(
                    frac, text=f"{completed_files['n']}/{total_files} files processed"
                )

        extractor_kwargs = {
            "ocr_images": ocr_images_in_pdf,
            "model_size": whisper_size,
            "ocr_frames": ocr_video_frames,
            "frame_interval_seconds": float(frame_interval),
        }

        start_time = time.time()
        result = build_dataset(
            files=local_paths,
            output_format=output_format,
            split_strategy=split_strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            system_prompt=system_prompt,
            instruction=instruction,
            extractor_kwargs=extractor_kwargs,
            progress_callback=progress_callback,
        )
        elapsed = time.time() - start_time

        progress_bar.progress(1.0, text="Done")
        status_box.update(label=f"Finished in {elapsed:.1f}s", state="complete")

        # -- Stats cards ------------------------------------------------------
        st.subheader("📊 Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files processed", f"{result.num_successful_files}/{result.num_files}")
        c2.metric("Total characters", f"{result.total_chars:,}")
        c3.metric("Samples generated", f"{result.num_samples:,}")
        c4.metric("Est. JSONL size", f"{result.estimated_size_bytes() / 1024:.1f} KB")

        # -- Per-file report ----------------------------------------------------
        st.subheader("📁 Per-file report")
        for report in result.file_reports:
            name = Path(report.path).name
            badge = (
                '<span class="sdb-badge sdb-badge-ok">OK</span>'
                if report.success
                else '<span class="sdb-badge sdb-badge-err">FAILED</span>'
            )
            with st.expander(
                f"{icon_for(name)} {name}  —  {'✅' if report.success else '❌'}",
                expanded=not report.success,
            ):
                st.markdown(badge, unsafe_allow_html=True)
                if report.success:
                    st.write(
                        f"- Characters extracted: **{report.char_count:,}**\n"
                        f"- Units: **{report.num_units}**\n"
                        f"- Chunks: **{report.num_chunks}**\n"
                        f"- Samples: **{report.num_samples}**"
                    )
                else:
                    st.error(report.error)
                if report.warnings:
                    st.warning("\n".join(f"- {w}" for w in report.warnings))
                logs = per_file_logs.get(report.path) or per_file_logs.get(str(report.path))
                if logs:
                    st.markdown("**Log**")
                    st.markdown("\n".join(f"- {line}" for line in logs))

        # -- Preview -------------------------------------------------------------
        if result.samples:
            st.subheader("👀 Dataset preview (first 3 samples)")
            with st.expander("Show JSON preview", expanded=True):
                st.json(result.samples[:3])

            # -- Download ----------------------------------------------------
            jsonl_bytes = "\n".join(
                json.dumps(s, ensure_ascii=False) for s in result.samples
            ).encode("utf-8")
            st.download_button(
                label=f"⬇️ Download {dataset_name}.jsonl",
                data=jsonl_bytes,
                file_name=f"{dataset_name}.jsonl",
                mime="application/jsonl",
                use_container_width=True,
                type="primary",
            )
        else:
            st.info("No samples were generated. Check the per-file report above for errors.")

elif not uploaded_files:
    st.info(
        "Upload one or more files above, configure options in the sidebar, then click **Build dataset**."
    )

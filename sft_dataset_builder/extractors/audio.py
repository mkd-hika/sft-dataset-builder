"""Audio extractor: local transcription via openai-whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Optional

from .base import BaseExtractor, ExtractedUnit, ExtractionResult

_MODEL_CACHE: dict[str, Any] = {}


def load_whisper_model(model_size: str = "base"):
    """Load (and cache in-process) a whisper model by size.

    Cached at module level so repeated calls / repeated files within one
    process don't reload weights. The Streamlit app wraps this with
    @st.cache_resource for cross-rerun caching.
    """
    if model_size not in _MODEL_CACHE:
        import whisper  # openai-whisper

        _MODEL_CACHE[model_size] = whisper.load_model(model_size)
    return _MODEL_CACHE[model_size]


class AudioExtractor(BaseExtractor):
    """Handles common audio formats via local Whisper transcription."""

    supported_extensions: ClassVar[set[str]] = {
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".ogg",
        ".aac",
        ".wma",
    }

    def extract(
        self,
        path: Path,
        model_size: str = "base",
        language: Optional[str] = None,
        segment_units: bool = True,
        **kwargs: Any,
    ) -> ExtractionResult:
        try:
            import whisper  # noqa: F401  (import check)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai-whisper is required for audio transcription. "
                "Install with `pip install openai-whisper` (also requires ffmpeg)."
            ) from exc

        model = load_whisper_model(model_size)
        result = model.transcribe(str(path), language=language, verbose=False)

        warnings: list[str] = []
        units: list[ExtractedUnit] = []

        segments = result.get("segments") or []
        if segment_units and segments:
            for seg in segments:
                text = (seg.get("text") or "").strip()
                if not text:
                    continue
                units.append(
                    ExtractedUnit(
                        text=text,
                        start_time=seg.get("start"),
                        end_time=seg.get("end"),
                        metadata={"segment_id": seg.get("id")},
                    )
                )
        else:
            full_text = (result.get("text") or "").strip()
            if full_text:
                units.append(ExtractedUnit(text=full_text))

        if not units:
            warnings.append("Transcription produced no text.")

        return ExtractionResult(
            source_path=str(path),
            source_type=path.suffix.lower().lstrip("."),
            units=units,
            full_text=(result.get("text") or "").strip(),
            warnings=warnings,
        )

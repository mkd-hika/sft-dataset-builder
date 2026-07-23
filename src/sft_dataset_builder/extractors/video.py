"""Video extractor: keyframe extraction + OCR, and audio track transcription."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Optional

from ._ocr import configure_tesseract
from .audio import AudioExtractor
from .base import BaseExtractor, ExtractedUnit, ExtractionResult


class VideoExtractor(BaseExtractor):
    """Handles video files: samples frames (optionally OCR'd) and transcribes audio."""

    supported_extensions: ClassVar[set[str]] = {
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".m4v",
    }

    def extract(
        self,
        path: Path,
        frame_interval_seconds: float = 10.0,
        ocr_frames: bool = False,
        transcribe_audio: bool = True,
        whisper_model_size: str = "base",
        language: Optional[str] = None,
        max_frames: int = 200,
        **kwargs: Any,
    ) -> ExtractionResult:
        warnings: list[str] = []
        units: list[ExtractedUnit] = []

        # -- audio track transcription --------------------------------------
        if transcribe_audio:
            try:
                audio_units, audio_warnings = self._transcribe_audio_track(
                    path, whisper_model_size, language
                )
                units.extend(audio_units)
                warnings.extend(audio_warnings)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"audio transcription failed: {exc}")

        # -- frame sampling / OCR ---------------------------------------------
        try:
            frame_units, frame_warnings = self._sample_frames(
                path, frame_interval_seconds, ocr_frames, max_frames
            )
            units.extend(frame_units)
            warnings.extend(frame_warnings)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"frame extraction failed: {exc}")

        units.sort(key=lambda u: u.start_time if u.start_time is not None else 0.0)

        return ExtractionResult(
            source_path=str(path),
            source_type=path.suffix.lower().lstrip("."),
            units=units,
            warnings=warnings,
        )

    @staticmethod
    def _transcribe_audio_track(
        path: Path, model_size: str, language: Optional[str]
    ) -> tuple[list[ExtractedUnit], list[str]]:
        """Extract the audio track to a temp WAV file (via ffmpeg) and transcribe it."""
        warnings: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "audio.wav"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(wav_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not wav_path.exists():
                warnings.append(
                    "ffmpeg could not extract an audio track (video may be silent, "
                    "or ffmpeg is not installed)."
                )
                return [], warnings

            audio_extractor = AudioExtractor()
            result = audio_extractor.extract(wav_path, model_size=model_size, language=language)
            units: list[ExtractedUnit] = []
            for u in result.units:
                u.metadata["source"] = "audio_transcript"
                units.append(u)
            warnings.extend(result.warnings)
            return units, warnings

    @staticmethod
    def _sample_frames(
        path: Path,
        interval_seconds: float,
        ocr_frames: bool,
        max_frames: int,
    ) -> tuple[list[ExtractedUnit], list[str]]:
        warnings: list[str] = []
        units: list[ExtractedUnit] = []

        if not ocr_frames:
            return units, warnings  # frame sampling only matters if we OCR them

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "opencv-python is required for video frame extraction. "
                "Install with `pip install opencv-python`."
            ) from exc

        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            warnings.append("pytesseract/Pillow not installed; skipping frame OCR.")
            return units, warnings

        configure_tesseract(pytesseract)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            warnings.append("could not open video for frame extraction.")
            return units, warnings

        try:
            frame_rate = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_step = max(int(frame_rate * interval_seconds), 1)

            frame_idx = 0
            sampled = 0
            while sampled < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = frame_idx / frame_rate

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                text = pytesseract.image_to_string(pil_image).strip()
                if text:
                    units.append(
                        ExtractedUnit(
                            text=text,
                            start_time=timestamp,
                            end_time=timestamp,
                            metadata={"source": "frame_ocr", "frame_index": frame_idx},
                        )
                    )
                sampled += 1
                frame_idx += frame_step
        finally:
            cap.release()

        return units, warnings

"""
Qwen3-TTS model wrapper for local Korean-first voice cloning.
"""

from __future__ import annotations

import os
import re
import socket
import tempfile
import uuid
import json
from datetime import datetime
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import gradio as gr
from storage import configure_runtime_storage

try:
    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
except ImportError:  # pragma: no cover - surfaced in UI status
    Qwen3TTSModel = None
    VoiceClonePromptItem = None


AudioTuple = Tuple[np.ndarray, int]
STORAGE_PATHS = configure_runtime_storage()


@dataclass
class PostprocessOptions:
    speed: float = 1.0
    pitch_semitones: float = 0.0
    ending_style: str = "default"
    ending_length_ms: int = 180


@dataclass
class MultiSpeakerLine:
    line_id: str
    line_index: int
    speaker: str
    text: str


class TTSModel:
    """Thin stateful wrapper around the official Qwen3-TTS inference API."""

    MODELS = {
        "Qwen3-TTS-1.7B Base (음성 복제)": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen3-TTS-VoiceDesign (설명형 스타일)": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "Qwen3-TTS-CustomVoice 0.6B (기본 화자)": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    }

    LANGUAGE_MAP = {
        "korean": "Korean",
        "japanese": "Japanese",
        "english": "English",
        "chinese": "Chinese",
    }

    def __init__(self):
        self.model = None
        self.device = None
        self.current_model_name = None
        self.output_dir = STORAGE_PATHS["outputs"]
        self.postprocess_presets_path = os.path.join(self.output_dir, "postprocess_presets.json")

    def get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _resolve_dtype(self) -> torch.dtype:
        if self.device and "cuda" in self.device:
            return torch.bfloat16
        return torch.float32

    def _resolve_attn_impl(self) -> str:
        if self.device and "cuda" in self.device and find_spec("flash_attn") is not None:
            return "flash_attention_2"
        return "eager"

    def _normalize_audio_input(self, audio_tuple) -> Optional[AudioTuple]:
        if audio_tuple is None:
            return None

        if isinstance(audio_tuple, tuple) and len(audio_tuple) == 2:
            first, second = audio_tuple
            if isinstance(first, int):
                sample_rate = int(first)
                audio = np.asarray(second, dtype=np.float32)
            else:
                audio = np.asarray(first, dtype=np.float32)
                sample_rate = int(second)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / 32768.0
            return audio.astype(np.float32), sample_rate

        raise TypeError("Unsupported audio input format.")

    def load_model(self, model_choice: str) -> str:
        if Qwen3TTSModel is None:
            return "❌ qwen-tts 패키지가 설치되지 않았습니다.\n💡 새 환경에서: pip install -r requirements-tts.txt"

        if model_choice == self.current_model_name and self.model is not None:
            return f"✅ {model_choice} 이미 로드됨"

        model_id = self.MODELS.get(model_choice)
        if not model_id:
            return f"❌ 알 수 없는 모델: {model_choice}"

        self.device = self.get_device()
        device_name = "GPU (CUDA)" if "cuda" in self.device else "CPU"

        try:
            if self.model is not None:
                del self.model
                if "cuda" in self.device:
                    torch.cuda.empty_cache()

            self.model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=self.device,
                dtype=self._resolve_dtype(),
                attn_implementation=self._resolve_attn_impl(),
            )
            self.current_model_name = model_choice

            return (
                f"✅ {model_choice} 로드 완료!\n"
                f"📍 디바이스: {device_name}\n"
                f"🧠 모델: {model_id}\n"
                f"⚠️ ASR와 TTS는 별도 가상환경 사용 권장"
            )
        except Exception as exc:
            return f"❌ 모델 로드 실패: {type(exc).__name__}: {exc}"

    def _build_output_stem(self, text: str, fallback_prefix: str) -> str:
        words = re.findall(r"\S+", (text or "").strip())
        stem = "_".join(words[:2]) if words else fallback_prefix.rstrip("_")
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", stem)
        stem = re.sub(r"\s+", "_", stem)
        stem = re.sub(r"_+", "_", stem).strip("._ ")
        return stem[:60] or fallback_prefix.rstrip("_")

    def _save_wav(self, wav: np.ndarray, sample_rate: int, text: str, fallback_prefix: str) -> str:
        stem = self._build_output_stem(text, fallback_prefix)
        path = os.path.join(self.output_dir, f"{stem}_{uuid.uuid4().hex[:8]}.wav")
        sf.write(path, wav, sample_rate)
        return path

    def _sanitize_preset_name(self, name: str) -> str:
        cleaned = re.sub(r"\s+", " ", (name or "").strip())
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", cleaned)
        return cleaned[:40]

    def _sanitize_filename_part(self, value: str, fallback: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", (value or "").strip())
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("._ ")
        return cleaned[:80] or fallback

    def _sanitize_project_name(self, name: str) -> str:
        return self._sanitize_filename_part(name, "multi_speaker_job")

    def _first_word_token(self, text: str) -> str:
        words = re.findall(r"\S+", (text or "").strip())
        first = words[0] if words else "line"
        return self._sanitize_filename_part(first, "line")

    def _build_multi_speaker_filename(self, line_index: int, speaker: str, text: str, version_id: str) -> str:
        line_part = f"{int(line_index):03d}"
        speaker_part = self._sanitize_filename_part(speaker, "speaker")
        first_word = self._first_word_token(text)
        version_part = self._sanitize_filename_part(version_id, "v1")
        return f"{line_part}_{speaker_part}_{first_word}_{version_part}.wav"

    def _default_postprocess_presets(self) -> dict:
        return {
            "기본": asdict(PostprocessOptions()),
            "나레이션 또렷": asdict(PostprocessOptions(speed=1.05, pitch_semitones=0.5, ending_style="soften", ending_length_ms=220)),
            "차분한 마무리": asdict(PostprocessOptions(speed=0.95, pitch_semitones=-0.5, ending_style="natural", ending_length_ms=320)),
            "짧고 단단하게": asdict(PostprocessOptions(speed=1.08, pitch_semitones=0.0, ending_style="fade", ending_length_ms=140)),
        }

    def _read_postprocess_presets(self) -> dict:
        defaults = self._default_postprocess_presets()
        if not os.path.exists(self.postprocess_presets_path):
            return defaults

        try:
            with open(self.postprocess_presets_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return defaults
            defaults.update(payload)
            return defaults
        except Exception:
            return defaults

    def _write_postprocess_presets(self, presets: dict) -> None:
        with open(self.postprocess_presets_path, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)

    def get_postprocess_preset_names(self) -> List[str]:
        return list(self._read_postprocess_presets().keys())

    def save_postprocess_preset(
        self,
        preset_name: str,
        speed: float,
        pitch_semitones: float,
        ending_style: str,
        ending_length_ms: int,
    ) -> Tuple[List[str], str]:
        name = self._sanitize_preset_name(preset_name)
        if not name:
            return self.get_postprocess_preset_names(), "❌ 프리셋 이름을 입력해주세요."

        presets = self._read_postprocess_presets()
        presets[name] = asdict(
            PostprocessOptions(
                speed=float(speed),
                pitch_semitones=float(pitch_semitones),
                ending_style=str(ending_style),
                ending_length_ms=int(ending_length_ms),
            )
        )
        self._write_postprocess_presets(presets)
        return list(presets.keys()), f"✅ 후처리 프리셋 저장: {name}"

    def load_postprocess_preset(self, preset_name: str) -> Tuple[float, float, str, int, str]:
        presets = self._read_postprocess_presets()
        payload = presets.get(preset_name)
        if not payload:
            defaults = PostprocessOptions()
            return defaults.speed, defaults.pitch_semitones, defaults.ending_style, defaults.ending_length_ms, "❌ 프리셋을 찾을 수 없습니다."

        options = PostprocessOptions(
            speed=float(payload.get("speed", 1.0)),
            pitch_semitones=float(payload.get("pitch_semitones", 0.0)),
            ending_style=str(payload.get("ending_style", "default")),
            ending_length_ms=int(payload.get("ending_length_ms", 180)),
        )
        return (
            options.speed,
            options.pitch_semitones,
            options.ending_style,
            options.ending_length_ms,
            f"✅ 후처리 프리셋 불러오기: {preset_name}",
        )

    def _resolve_file_path(self, file_input) -> Optional[str]:
        if not file_input:
            return None
        return getattr(file_input, "name", None) or getattr(file_input, "path", None) or str(file_input)

    def _resolve_file_paths(self, file_inputs) -> List[str]:
        if not file_inputs:
            return []
        if isinstance(file_inputs, (list, tuple)):
            resolved = [self._resolve_file_path(item) for item in file_inputs]
            return [path for path in resolved if path]
        single = self._resolve_file_path(file_inputs)
        return [single] if single else []

    def _is_voice_prompt_candidate(self, path: str) -> bool:
        lower_name = os.path.basename(path).lower()
        if not lower_name.endswith(".pt"):
            return False
        blocked = {"hubert_base.pt", "rmvpe.pt"}
        if lower_name in blocked:
            return False
        return True

    def get_voice_prompt_choices(self) -> List[Tuple[str, str]]:
        prompt_paths: List[str] = []
        for root, _, files in os.walk(self.output_dir):
            for filename in files:
                path = os.path.join(root, filename)
                if self._is_voice_prompt_candidate(path):
                    prompt_paths.append(path)

        prompt_paths = sorted(set(prompt_paths), key=lambda item: item.lower())
        choices: List[Tuple[str, str]] = []
        for path in prompt_paths:
            label = os.path.basename(path)
            relative_dir = os.path.relpath(os.path.dirname(path), self.output_dir)
            if relative_dir != ".":
                label = f"{label}  ({relative_dir})"
            choices.append((label, path))
        return choices

    def refresh_voice_prompt_dropdown(self) -> Tuple[gr.Dropdown, str]:
        choices = self.get_voice_prompt_choices()
        return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None), f"✅ voice prompt {len(choices)}개 검색 완료"

    def _load_voice_prompt_items(self, prompt_path: str):
        payload = torch.load(prompt_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "items" not in payload:
            raise ValueError("올바른 voice prompt 파일이 아닙니다.")
        return [VoiceClonePromptItem(**item) for item in payload["items"]]

    def _resample_linear(self, wav: np.ndarray, target_length: int) -> np.ndarray:
        if target_length <= 1 or len(wav) <= 1:
            return wav.astype(np.float32)
        if target_length == len(wav):
            return wav.astype(np.float32)

        source_positions = np.linspace(0.0, 1.0, num=len(wav), endpoint=True)
        target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
        return np.interp(target_positions, source_positions, wav).astype(np.float32)

    def _apply_speed(self, wav: np.ndarray, speed: float) -> np.ndarray:
        speed = float(np.clip(speed, 0.5, 1.8))
        if abs(speed - 1.0) < 1e-3:
            return wav.astype(np.float32)

        target_length = max(1, int(round(len(wav) / speed)))
        return self._resample_linear(wav, target_length)

    def _apply_pitch_shift(self, wav: np.ndarray, semitones: float) -> np.ndarray:
        semitones = float(np.clip(semitones, -8.0, 8.0))
        if abs(semitones) < 1e-3 or len(wav) < 2:
            return wav.astype(np.float32)

        pitch_ratio = 2 ** (semitones / 12.0)
        shifted_length = max(1, int(round(len(wav) / pitch_ratio)))
        shifted = self._resample_linear(wav, shifted_length)
        return self._resample_linear(shifted, len(wav))

    def _build_hold_extension(self, tail: np.ndarray, extra_samples: int) -> np.ndarray:
        if extra_samples <= 0 or len(tail) < 4:
            return np.zeros(0, dtype=np.float32)

        sustain_size = max(4, min(len(tail), extra_samples, max(8, len(tail) // 3)))
        sustain = tail[-sustain_size:].astype(np.float32)
        tiled = np.tile(sustain, int(np.ceil(extra_samples / sustain_size)))[:extra_samples]
        fade = np.linspace(1.0, 0.0, num=extra_samples, endpoint=True, dtype=np.float32)
        return (tiled * fade).astype(np.float32)

    def _apply_ending(self, wav: np.ndarray, sample_rate: int, ending_style: str, ending_length_ms: int) -> np.ndarray:
        ending_style = (ending_style or "default").strip().lower()
        if ending_style == "default" or len(wav) < 8:
            return wav.astype(np.float32)

        tail_samples = max(1, int(sample_rate * max(0, ending_length_ms) / 1000))
        tail_samples = min(len(wav), tail_samples)
        result = wav.astype(np.float32).copy()
        tail = result[-tail_samples:]

        if ending_style == "fade":
            envelope = np.linspace(1.0, 0.0, num=tail_samples, endpoint=True, dtype=np.float32)
            result[-tail_samples:] = tail * envelope
            return result

        if ending_style == "soften":
            envelope = np.linspace(1.0, 0.35, num=tail_samples, endpoint=True, dtype=np.float32)
            result[-tail_samples:] = tail * envelope
            fade_samples = max(8, tail_samples // 5)
            fade = np.linspace(1.0, 0.0, num=fade_samples, endpoint=True, dtype=np.float32)
            result[-fade_samples:] *= fade
            return result

        if ending_style == "hold":
            extension = self._build_hold_extension(tail, tail_samples)
            if extension.size == 0:
                return result
            crossfade_samples = min(len(tail), len(extension), max(8, sample_rate // 100))
            if crossfade_samples > 0:
                fade_out = np.linspace(1.0, 0.0, num=crossfade_samples, endpoint=True, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, num=crossfade_samples, endpoint=True, dtype=np.float32)
                extension[:crossfade_samples] = (
                    result[-crossfade_samples:] * fade_out + extension[:crossfade_samples] * fade_in
                )
            return np.concatenate([result, extension]).astype(np.float32)

        if ending_style == "natural":
            soften_envelope = np.linspace(1.0, 0.65, num=tail_samples, endpoint=True, dtype=np.float32)
            result[-tail_samples:] = tail * soften_envelope
            extension = self._build_hold_extension(result[-tail_samples:], max(1, tail_samples // 2))
            if extension.size > 0:
                fade = np.linspace(0.75, 0.0, num=len(extension), endpoint=True, dtype=np.float32)
                extension = extension * fade
                return np.concatenate([result, extension]).astype(np.float32)
            fade_samples = max(8, tail_samples // 4)
            result[-fade_samples:] *= np.linspace(1.0, 0.0, num=fade_samples, endpoint=True, dtype=np.float32)
            return result

        return result

    def _apply_postprocess(self, wav: np.ndarray, sample_rate: int, options: PostprocessOptions) -> np.ndarray:
        processed = wav.astype(np.float32)
        processed = self._apply_pitch_shift(processed, options.pitch_semitones)
        processed = self._apply_speed(processed, options.speed)
        processed = self._apply_ending(processed, sample_rate, options.ending_style, options.ending_length_ms)
        peak = float(np.max(np.abs(processed))) if len(processed) else 0.0
        if peak > 1.0:
            processed = processed / peak
        return processed.astype(np.float32)

    def apply_postprocess_to_file(
        self,
        audio_file,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        ending_style: str = "default",
        ending_length_ms: int = 180,
    ) -> Tuple[Optional[str], str]:
        source_path = self._resolve_file_path(audio_file)
        if not source_path or not os.path.exists(source_path):
            return None, "❌ 먼저 생성된 원본 음성이 필요합니다."

        try:
            wav, sample_rate = sf.read(source_path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            postprocess_options = PostprocessOptions(
                speed=speed,
                pitch_semitones=pitch_semitones,
                ending_style=ending_style,
                ending_length_ms=int(ending_length_ms),
            )
            processed = self._apply_postprocess(wav, sample_rate, postprocess_options)
            output_path = self._save_wav(processed, sample_rate, "postprocess_preview", "tts_post_")
            return output_path, "✅ 후처리 미리듣기 파일 생성 완료"
        except Exception as exc:
            return None, f"❌ 후처리 실패: {type(exc).__name__}: {exc}"

    def _split_text(self, text: str, max_chars: int = 90) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        pieces = re.split(r"(?<=[.!?。！？])\s+", text)
        chunks: List[str] = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            if len(piece) > max_chars:
                sub_pieces = re.split(r"(?<=[,;:，、])\s*", piece)
            else:
                sub_pieces = [piece]

            for sub_piece in sub_pieces:
                sub_piece = sub_piece.strip()
                if not sub_piece:
                    continue

                if not current:
                    current = sub_piece
                    continue

                if len(current) + 1 + len(sub_piece) <= max_chars:
                    current = f"{current} {sub_piece}"
                else:
                    chunks.append(current)
                    current = sub_piece

        if current:
            chunks.append(current)

        return chunks or [text]

    def _concat_wavs(self, wavs: List[np.ndarray], sample_rate: int, silence_ms: int = 250) -> np.ndarray:
        if len(wavs) == 1:
            return wavs[0]

        silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.float32)
        merged: List[np.ndarray] = []
        for index, wav in enumerate(wavs):
            merged.append(wav.astype(np.float32))
            if index < len(wavs) - 1:
                merged.append(silence)
        return np.concatenate(merged)

    def _estimate_max_new_tokens(self, text: str) -> int:
        return min(2048, max(512, len(text) * 12))

    def parse_script_lines(self, script_text: str) -> List[MultiSpeakerLine]:
        if not script_text or not script_text.strip():
            raise ValueError("대본을 입력해주세요.")

        parsed_lines: List[MultiSpeakerLine] = []
        raw_lines = script_text.splitlines()

        for source_index, raw_line in enumerate(raw_lines, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if ":" not in stripped:
                raise ValueError(f"{source_index}번째 줄에 ':' 구분자가 없습니다.")

            speaker, text = stripped.split(":", 1)
            speaker = speaker.strip()
            text = text.strip()
            if not speaker:
                raise ValueError(f"{source_index}번째 줄에 화자 이름이 비어 있습니다.")
            if not text:
                raise ValueError(f"{source_index}번째 줄의 대사가 비어 있습니다.")

            line_index = len(parsed_lines) + 1
            parsed_lines.append(
                MultiSpeakerLine(
                    line_id=f"line_{line_index:03d}",
                    line_index=line_index,
                    speaker=speaker,
                    text=text,
                )
            )

        if not parsed_lines:
            raise ValueError("유효한 대본 줄이 없습니다.")

        return parsed_lines

    def parse_paragraph_blocks(self, script_text: str, max_blocks: int = 10) -> List[str]:
        if not script_text or not script_text.strip():
            return []

        normalized = script_text.replace("\r\n", "\n").replace("\r", "\n")
        blocks = re.split(r"\n\s*\n+", normalized)
        cleaned = [block.strip() for block in blocks if block and block.strip()]
        return cleaned[:max_blocks]

    def build_multi_speaker_rows(self, script_text: str) -> Tuple[List[List[Any]], str]:
        try:
            parsed_lines = self.parse_script_lines(script_text)
        except ValueError as exc:
            return [], f"❌ {exc}"

        seen = set()
        rows: List[List[Any]] = []
        for line in parsed_lines:
            if line.speaker in seen:
                continue
            seen.add(line.speaker)
            rows.append([line.speaker, "", 1.0, 0.0, "default", 180])

        return rows, f"✅ 화자 {len(rows)}명 추출 완료 / 대사 {len(parsed_lines)}줄"

    def build_speaker_selector(self, speaker_rows) -> gr.Dropdown:
        if hasattr(speaker_rows, "fillna") and hasattr(speaker_rows, "values"):
            speaker_rows = speaker_rows.fillna("").values.tolist()
        elif isinstance(speaker_rows, tuple):
            speaker_rows = list(speaker_rows)

        speakers: List[str] = []
        if isinstance(speaker_rows, list):
            for row in speaker_rows:
                if row and len(row) > 0 and str(row[0]).strip():
                    speakers.append(str(row[0]).strip())
        unique_speakers = list(dict.fromkeys(speakers))
        return gr.Dropdown(choices=unique_speakers, value=unique_speakers[0] if unique_speakers else None)

    def _next_unassigned_speaker(self, speaker_rows, current_speaker: str = "") -> Optional[str]:
        if hasattr(speaker_rows, "fillna") and hasattr(speaker_rows, "values"):
            speaker_rows = speaker_rows.fillna("").values.tolist()
        elif isinstance(speaker_rows, tuple):
            speaker_rows = list(speaker_rows)

        if not isinstance(speaker_rows, list):
            return None

        ordered = []
        for row in speaker_rows:
            if row and len(row) > 0 and str(row[0]).strip():
                speaker = str(row[0]).strip()
                prompt_path = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
                ordered.append((speaker, prompt_path))

        if not ordered:
            return None

        for speaker, prompt_path in ordered:
            if not prompt_path:
                return speaker

        if current_speaker:
            speakers = [speaker for speaker, _ in ordered]
            if current_speaker in speakers:
                current_index = speakers.index(current_speaker)
                if current_index + 1 < len(speakers):
                    return speakers[current_index + 1]
        return ordered[0][0]

    def summarize_speaker_rows(self, speaker_rows) -> str:
        if hasattr(speaker_rows, "fillna") and hasattr(speaker_rows, "values"):
            speaker_rows = speaker_rows.fillna("").values.tolist()
        elif isinstance(speaker_rows, tuple):
            speaker_rows = list(speaker_rows)

        if not isinstance(speaker_rows, list):
            return "화자 설정 표를 준비해주세요."

        total = 0
        assigned = 0
        for row in speaker_rows:
            if row and len(row) > 0 and str(row[0]).strip():
                total += 1
                prompt_path = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
                if prompt_path:
                    assigned += 1
        return f"화자 {assigned}/{total}명에 voice prompt가 연결됨"

    def get_speaker_editor_values(
        self,
        speaker_rows,
        selected_speaker: str,
    ) -> Tuple[str, float, float, str, int, str]:
        if hasattr(speaker_rows, "fillna") and hasattr(speaker_rows, "values"):
            speaker_rows = speaker_rows.fillna("").values.tolist()
        elif isinstance(speaker_rows, tuple):
            speaker_rows = list(speaker_rows)

        if not selected_speaker:
            return "", 1.0, 0.0, "default", 180, "화자를 먼저 선택해주세요."

        if not isinstance(speaker_rows, list):
            return "", 1.0, 0.0, "default", 180, "화자 설정 표를 먼저 준비해주세요."

        for row in speaker_rows:
            if row and len(row) > 0 and str(row[0]).strip() == selected_speaker:
                prompt_path = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
                speed = float(row[2]) if len(row) > 2 and row[2] not in (None, "") else 1.0
                pitch = float(row[3]) if len(row) > 3 and row[3] not in (None, "") else 0.0
                ending_style = str(row[4]).strip() if len(row) > 4 and row[4] not in (None, "") else "default"
                ending_length = int(float(row[5])) if len(row) > 5 and row[5] not in (None, "") else 180
                return prompt_path, speed, pitch, ending_style, ending_length, f"✅ '{selected_speaker}' 설정을 불러왔습니다."

        return "", 1.0, 0.0, "default", 180, f"❌ 화자 '{selected_speaker}'를 표에서 찾을 수 없습니다."

    def update_speaker_row(
        self,
        speaker_rows,
        selected_speaker: str,
        selected_prompt_path: str,
        browser_prompt_path: str,
        uploaded_prompt_file,
        speed: float,
        pitch: float,
        ending_style: str,
        ending_length_ms: int,
    ) -> Tuple[List[List[Any]], str, str, str]:
        if hasattr(speaker_rows, "fillna") and hasattr(speaker_rows, "values"):
            speaker_rows = speaker_rows.fillna("").values.tolist()
        elif isinstance(speaker_rows, tuple):
            speaker_rows = list(speaker_rows)

        if not isinstance(speaker_rows, list) or not speaker_rows:
            return [], "", "❌ 화자 설정 표가 없습니다.", ""
        if not selected_speaker:
            return speaker_rows, "", "❌ 수정할 화자를 먼저 선택해주세요.", ""

        uploaded_path = self._resolve_file_path(uploaded_prompt_file)
        final_prompt_path = uploaded_path or (browser_prompt_path or "").strip() or (selected_prompt_path or "").strip()
        if not final_prompt_path:
            return speaker_rows, "", "❌ voice prompt를 드롭하거나 목록에서 선택해주세요.", ""
        if not os.path.exists(final_prompt_path):
            return speaker_rows, final_prompt_path, f"❌ 파일이 없습니다: {final_prompt_path}", ""

        updated_rows: List[List[Any]] = []
        updated = False
        for row in speaker_rows:
            if row and len(row) > 0 and str(row[0]).strip() == selected_speaker:
                updated_rows.append(
                    [
                        selected_speaker,
                        final_prompt_path,
                        float(speed),
                        float(pitch),
                        str(ending_style),
                        int(ending_length_ms),
                    ]
                )
                updated = True
            else:
                updated_rows.append(row)

        if not updated:
            return speaker_rows, final_prompt_path, f"❌ 화자 '{selected_speaker}'를 찾지 못했습니다.", ""

        next_speaker = self._next_unassigned_speaker(updated_rows, selected_speaker) or selected_speaker
        summary = self.summarize_speaker_rows(updated_rows)
        return updated_rows, final_prompt_path, f"✅ '{selected_speaker}' 설정 적용 완료\n{summary}", next_speaker

    def bulk_assign_speaker_rows(
        self,
        speaker_rows,
        uploaded_prompt_files,
    ) -> Tuple[List[List[Any]], str, str]:
        if hasattr(speaker_rows, "fillna") and hasattr(speaker_rows, "values"):
            speaker_rows = speaker_rows.fillna("").values.tolist()
        elif isinstance(speaker_rows, tuple):
            speaker_rows = list(speaker_rows)

        if not isinstance(speaker_rows, list) or not speaker_rows:
            return [], "❌ 화자 설정 표가 없습니다.", ""

        file_paths = [path for path in self._resolve_file_paths(uploaded_prompt_files) if path and os.path.exists(path)]
        if not file_paths:
            return speaker_rows, "❌ 일괄 배정할 `.pt` 파일을 하나 이상 드롭해주세요.", ""

        unassigned_indexes = []
        for index, row in enumerate(speaker_rows):
            if row and len(row) > 0 and str(row[0]).strip():
                prompt_path = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
                if not prompt_path:
                    unassigned_indexes.append(index)

        target_indexes = unassigned_indexes if unassigned_indexes else [index for index, row in enumerate(speaker_rows) if row and len(row) > 0 and str(row[0]).strip()]
        if not target_indexes:
            return speaker_rows, "❌ 배정할 화자가 없습니다.", ""

        assigned_count = 0
        for target_index, prompt_path in zip(target_indexes, file_paths):
            row = list(speaker_rows[target_index])
            row[1] = prompt_path
            speaker_rows[target_index] = row
            assigned_count += 1

        next_speaker = self._next_unassigned_speaker(speaker_rows, "")
        summary = self.summarize_speaker_rows(speaker_rows)
        return speaker_rows, f"✅ {assigned_count}개 voice prompt 일괄 배정 완료\n{summary}", next_speaker or ""

    def _normalize_speaker_rows(self, speaker_rows) -> Dict[str, Dict[str, Any]]:
        if speaker_rows is None:
            raise ValueError("화자 설정 표가 비어 있습니다.")

        if hasattr(speaker_rows, "fillna") and hasattr(speaker_rows, "values"):
            speaker_rows = speaker_rows.fillna("").values.tolist()
        elif isinstance(speaker_rows, tuple):
            speaker_rows = list(speaker_rows)

        if not isinstance(speaker_rows, list) or not speaker_rows:
            raise ValueError("화자 설정 표가 비어 있습니다.")

        configs: Dict[str, Dict[str, Any]] = {}
        for index, row in enumerate(speaker_rows, start=1):
            if not row or all((str(cell).strip() == "" for cell in row if cell is not None)):
                continue

            speaker = str(row[0]).strip() if len(row) > 0 and row[0] is not None else ""
            prompt_path = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
            if not speaker:
                raise ValueError(f"화자 설정 {index}행의 화자 이름이 비어 있습니다.")
            if not prompt_path:
                raise ValueError(f"화자 '{speaker}'의 voice prompt 경로를 입력해주세요.")
            if not os.path.exists(prompt_path):
                raise ValueError(f"화자 '{speaker}'의 voice prompt 파일이 없습니다: {prompt_path}")

            ending_style = str(row[4]).strip() if len(row) > 4 and row[4] is not None else "default"
            if ending_style not in {"default", "soften", "fade", "hold", "natural"}:
                raise ValueError(f"화자 '{speaker}'의 끝음 처리 값이 올바르지 않습니다: {ending_style}")

            configs[speaker] = {
                "prompt_path": prompt_path,
                "options": PostprocessOptions(
                    speed=float(row[2]) if len(row) > 2 and row[2] is not None else 1.0,
                    pitch_semitones=float(row[3]) if len(row) > 3 and row[3] is not None else 0.0,
                    ending_style=ending_style,
                    ending_length_ms=int(float(row[5])) if len(row) > 5 and row[5] is not None else 180,
                ),
            }

        if not configs:
            raise ValueError("유효한 화자 설정이 없습니다.")

        return configs

    def build_paragraph_card_rows(self, script_text: str, max_blocks: int = 10) -> List[Dict[str, Any]]:
        blocks = self.parse_paragraph_blocks(script_text, max_blocks=max_blocks)
        rows: List[Dict[str, Any]] = []
        for index, block in enumerate(blocks, start=1):
            rows.append(
                {
                    "card_index": index,
                    "text": block,
                    "speaker": f"화자{index}",
                    "prompt_path": "",
                    "speed": 1.0,
                    "pitch": 0.0,
                    "ending_style": "default",
                    "ending_length_ms": 180,
                }
            )
        return rows

    def generate_multi_speaker_paragraphs(
        self,
        paragraph_cards: List[Dict[str, Any]],
        language: str,
        job_name: str = "",
        save_line_files: bool = True,
        merge_output: bool = True,
        silence_ms: int = 120,
    ) -> Tuple[Optional[str], List[str], str, List[List[Any]], str, gr.Dropdown]:
        if self.model is None:
            return None, [], "❌ 먼저 TTS 모델을 로드해주세요!", [], "", gr.Dropdown(choices=[], value=None)
        if not paragraph_cards:
            return None, [], "❌ 생성할 문단 카드가 없습니다.", [], "", gr.Dropdown(choices=[], value=None)

        parsed_lines: List[MultiSpeakerLine] = []
        speaker_rows: List[List[Any]] = []
        seen_speakers: Dict[str, List[Any]] = {}

        for index, card in enumerate(paragraph_cards, start=1):
            text = str(card.get("text", "")).strip()
            speaker = str(card.get("speaker", "")).strip()
            prompt_path = str(card.get("prompt_path", "")).strip()
            if not text:
                continue
            if not speaker:
                return None, [], f"❌ {index}번 문단 카드의 화자 이름이 비어 있습니다.", [], "", gr.Dropdown(choices=[], value=None)
            if not prompt_path:
                return None, [], f"❌ {index}번 문단 카드의 voice prompt가 비어 있습니다.", [], "", gr.Dropdown(choices=[], value=None)

            parsed_lines.append(
                MultiSpeakerLine(
                    line_id=f"line_{len(parsed_lines)+1:03d}",
                    line_index=len(parsed_lines) + 1,
                    speaker=speaker,
                    text=text,
                )
            )

            if speaker not in seen_speakers:
                seen_speakers[speaker] = [
                    speaker,
                    prompt_path,
                    float(card.get("speed", 1.0)),
                    float(card.get("pitch", 0.0)),
                    str(card.get("ending_style", "default")),
                    int(card.get("ending_length_ms", 180)),
                ]

        if not parsed_lines:
            return None, [], "❌ 비어 있지 않은 문단 카드가 없습니다.", [], "", gr.Dropdown(choices=[], value=None)

        speaker_rows = list(seen_speakers.values())
        try:
            speaker_configs = self._normalize_speaker_rows(speaker_rows)
        except ValueError as exc:
            return None, [], f"❌ {exc}", [], "", gr.Dropdown(choices=[], value=None)

        job_dir = self._create_multi_speaker_job_dir(job_name or "multi_speaker")
        prompt_cache: Dict[str, Any] = {}
        rendered_wavs: List[np.ndarray] = []
        output_files: List[str] = []
        result_rows: List[List[Any]] = []
        final_mix_path: Optional[str] = None
        sample_rate_for_merge: Optional[int] = None

        try:
            with open(os.path.join(job_dir, "script.txt"), "w", encoding="utf-8") as script_file:
                script_file.write("\n\n".join(line.text for line in parsed_lines) + "\n")

            for line in parsed_lines:
                speaker_config = speaker_configs[line.speaker]
                prompt_path = speaker_config["prompt_path"]
                options: PostprocessOptions = speaker_config["options"]

                if prompt_path not in prompt_cache:
                    prompt_cache[prompt_path] = self._load_voice_prompt_items(prompt_path)

                wav, sample_rate, chunk_count = self._generate_voice_clone_wav(
                    text=line.text,
                    language=language,
                    audio_input=None,
                    reference_text=None,
                    x_vector_only_mode=False,
                    voice_clone_prompt=prompt_cache[prompt_path],
                )
                processed_wav = self._apply_postprocess(wav, sample_rate, options)
                sample_rate_for_merge = sample_rate
                rendered_wavs.append(processed_wav)

                line_filename = self._build_multi_speaker_filename(
                    line_index=line.line_index,
                    speaker=line.speaker,
                    text=line.text,
                    version_id="v1",
                )
                line_path = os.path.join(job_dir, line_filename)
                self._save_wav_to_path(processed_wav, sample_rate, line_path)
                if save_line_files:
                    output_files.append(line_path)

                result_rows.append(
                    [
                        line.line_id,
                        line.speaker,
                        line.text,
                        "완료",
                        line_path,
                        chunk_count,
                        "v1",
                        1,
                    ]
                )

            if merge_output and rendered_wavs and sample_rate_for_merge is not None:
                merged_wav = self._merge_wavs_with_silence(rendered_wavs, sample_rate_for_merge, silence_ms)
                final_mix_path = os.path.join(job_dir, "final_mix.wav")
                self._save_wav_to_path(merged_wav, sample_rate_for_merge, final_mix_path)
                output_files.append(final_mix_path)

            manifest = {
                "job_name": job_name or os.path.basename(job_dir),
                "job_dir": job_dir,
                "language": language,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "save_line_files": bool(save_line_files),
                "merge_output": bool(merge_output),
                "silence_ms": int(silence_ms),
                "paragraph_mode": True,
                "paragraph_cards": paragraph_cards,
                "speakers": {
                    speaker: {
                        "prompt_path": config["prompt_path"],
                        "options": asdict(config["options"]),
                    }
                    for speaker, config in speaker_configs.items()
                },
                "lines": [
                    {
                        "line_id": row[0],
                        "speaker": row[1],
                        "text": row[2],
                        "status": row[3],
                        "audio_path": row[4],
                        "chunk_count": row[5],
                        "selected_version": row[6],
                        "versions": [
                            {
                                "version_id": row[6],
                                "audio_path": row[4],
                                "chunk_count": row[5],
                                "created_at": datetime.now().isoformat(timespec="seconds"),
                            }
                        ],
                    }
                    for row in result_rows
                ],
                "final_mix_path": final_mix_path,
            }
            manifest_path = self._write_manifest(job_dir, manifest)
            output_files.append(manifest_path)
            line_choices = self._build_line_choices_from_manifest(manifest)
            line_dropdown = gr.Dropdown(choices=line_choices, value=line_choices[0] if line_choices else None)

            summary = (
                f"✅ 문단 카드 음성 생성 완료\n"
                f"문단 수: {len(result_rows)}\n"
                f"화자 수: {len(speaker_configs)}\n"
                f"작업 폴더: {job_dir}"
            )
            return final_mix_path, output_files, summary, result_rows, job_dir, line_dropdown
        except Exception as exc:
            return None, output_files, f"❌ 문단 카드 생성 실패: {type(exc).__name__}: {exc}", result_rows, job_dir, gr.Dropdown(choices=[], value=None)

    def _create_multi_speaker_job_dir(self, job_name: str) -> str:
        safe_name = self._sanitize_project_name(job_name or "multi_speaker_job")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir = os.path.join(self.output_dir, "multi_speaker", f"{timestamp}_{safe_name}")
        os.makedirs(job_dir, exist_ok=True)
        return job_dir

    def _save_wav_to_path(self, wav: np.ndarray, sample_rate: int, path: str) -> str:
        sf.write(path, wav, sample_rate)
        return path

    def _manifest_path(self, job_dir: str) -> str:
        return os.path.join(job_dir, "script_manifest.json")

    def _write_manifest(self, job_dir: str, manifest: Dict[str, Any]) -> str:
        manifest_path = self._manifest_path(job_dir)
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        return manifest_path

    def _read_manifest(self, job_dir: str) -> Dict[str, Any]:
        manifest_path = self._manifest_path(job_dir)
        if not os.path.exists(manifest_path):
            raise ValueError(f"manifest 파일이 없습니다: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            payload = json.load(manifest_file)
        if not isinstance(payload, dict):
            raise ValueError("manifest 형식이 올바르지 않습니다.")
        return payload

    def _build_result_rows_from_manifest(self, manifest: Dict[str, Any]) -> List[List[Any]]:
        rows: List[List[Any]] = []
        for line in manifest.get("lines", []):
            selected_version = line.get("selected_version", "v1")
            version_count = len(line.get("versions", []))
            rows.append(
                [
                    line.get("line_id", ""),
                    line.get("speaker", ""),
                    line.get("text", ""),
                    line.get("status", ""),
                    line.get("audio_path", ""),
                    line.get("chunk_count", 0),
                    selected_version,
                    version_count,
                ]
            )
        return rows

    def _build_line_choices_from_manifest(self, manifest: Dict[str, Any]) -> List[str]:
        choices = []
        for line in manifest.get("lines", []):
            choices.append(f"{line.get('line_id', '')} | {line.get('speaker', '')}")
        return choices

    def _parse_line_choice(self, selected_line: str) -> str:
        return (selected_line or "").split("|", 1)[0].strip()

    def _find_manifest_line(self, manifest: Dict[str, Any], line_id: str) -> Dict[str, Any]:
        for line in manifest.get("lines", []):
            if line.get("line_id") == line_id:
                return line
        raise ValueError(f"해당 줄을 찾을 수 없습니다: {line_id}")

    def _next_line_version(self, line_entry: Dict[str, Any]) -> Tuple[str, int]:
        versions = line_entry.get("versions", [])
        next_index = len(versions) + 1
        return f"v{next_index}", next_index

    def _render_line_with_prompt(
        self,
        text: str,
        language: str,
        prompt_path: str,
        options: PostprocessOptions,
    ) -> Tuple[np.ndarray, int, int]:
        items = self._load_voice_prompt_items(prompt_path)
        wav, sample_rate, chunk_count = self._generate_voice_clone_wav(
            text=text,
            language=language,
            audio_input=None,
            reference_text=None,
            x_vector_only_mode=False,
            voice_clone_prompt=items,
        )
        processed_wav = self._apply_postprocess(wav, sample_rate, options)
        return processed_wav, sample_rate, chunk_count

    def _merge_wavs_with_silence(self, wavs: List[np.ndarray], sample_rate: int, silence_ms: int) -> np.ndarray:
        if not wavs:
            return np.zeros(0, dtype=np.float32)
        return self._concat_wavs(wavs, sample_rate, silence_ms=max(0, int(silence_ms)))

    def generate_multi_speaker_script(
        self,
        script_text: str,
        speaker_rows,
        language: str,
        job_name: str = "",
        save_line_files: bool = True,
        merge_output: bool = True,
        silence_ms: int = 120,
    ) -> Tuple[Optional[str], List[str], str, List[List[Any]], str, gr.Dropdown]:
        if self.model is None:
            return None, [], "❌ 먼저 TTS 모델을 로드해주세요!", [], "", gr.Dropdown(choices=[], value=None)

        try:
            parsed_lines = self.parse_script_lines(script_text)
            speaker_configs = self._normalize_speaker_rows(speaker_rows)
        except ValueError as exc:
            return None, [], f"❌ {exc}", [], "", gr.Dropdown(choices=[], value=None)

        missing_speakers = [line.speaker for line in parsed_lines if line.speaker not in speaker_configs]
        if missing_speakers:
            unique_missing = ", ".join(dict.fromkeys(missing_speakers))
            return None, [], f"❌ 다음 화자에 대한 설정이 없습니다: {unique_missing}", [], "", gr.Dropdown(choices=[], value=None)

        job_dir = self._create_multi_speaker_job_dir(job_name or "multi_speaker")
        prompt_cache: Dict[str, Any] = {}
        rendered_wavs: List[np.ndarray] = []
        output_files: List[str] = []
        result_rows: List[List[Any]] = []
        final_mix_path: Optional[str] = None
        sample_rate_for_merge: Optional[int] = None

        try:
            with open(os.path.join(job_dir, "script.txt"), "w", encoding="utf-8") as script_file:
                script_file.write(script_text.strip() + "\n")

            for line in parsed_lines:
                speaker_config = speaker_configs[line.speaker]
                prompt_path = speaker_config["prompt_path"]
                options: PostprocessOptions = speaker_config["options"]

                if prompt_path not in prompt_cache:
                    prompt_cache[prompt_path] = self._load_voice_prompt_items(prompt_path)

                wav, sample_rate, chunk_count = self._generate_voice_clone_wav(
                    text=line.text,
                    language=language,
                    audio_input=None,
                    reference_text=None,
                    x_vector_only_mode=False,
                    voice_clone_prompt=prompt_cache[prompt_path],
                )
                processed_wav = self._apply_postprocess(wav, sample_rate, options)
                sample_rate_for_merge = sample_rate
                rendered_wavs.append(processed_wav)

                line_filename = self._build_multi_speaker_filename(
                    line_index=line.line_index,
                    speaker=line.speaker,
                    text=line.text,
                    version_id="v1",
                )
                line_path = os.path.join(job_dir, line_filename)
                self._save_wav_to_path(processed_wav, sample_rate, line_path)
                if save_line_files:
                    output_files.append(line_path)

                result_rows.append(
                    [
                        line.line_id,
                        line.speaker,
                        line.text,
                        "완료",
                        line_path,
                        chunk_count,
                        "v1",
                        1,
                    ]
                )

            if merge_output and rendered_wavs and sample_rate_for_merge is not None:
                merged_wav = self._merge_wavs_with_silence(rendered_wavs, sample_rate_for_merge, silence_ms)
                final_mix_path = os.path.join(job_dir, "final_mix.wav")
                self._save_wav_to_path(merged_wav, sample_rate_for_merge, final_mix_path)
                output_files.append(final_mix_path)

            manifest = {
                "job_name": job_name or os.path.basename(job_dir),
                "job_dir": job_dir,
                "language": language,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "save_line_files": bool(save_line_files),
                "merge_output": bool(merge_output),
                "silence_ms": int(silence_ms),
                "script_text": script_text,
                "speakers": {
                    speaker: {
                        "prompt_path": config["prompt_path"],
                        "options": asdict(config["options"]),
                    }
                    for speaker, config in speaker_configs.items()
                },
                "lines": [
                    {
                        "line_id": row[0],
                        "speaker": row[1],
                        "text": row[2],
                        "status": row[3],
                        "audio_path": row[4],
                        "chunk_count": row[5],
                        "selected_version": row[6],
                        "versions": [
                            {
                                "version_id": row[6],
                                "audio_path": row[4],
                                "chunk_count": row[5],
                                "created_at": datetime.now().isoformat(timespec="seconds"),
                            }
                        ],
                    }
                    for row in result_rows
                ],
                "final_mix_path": final_mix_path,
            }
            manifest_path = self._write_manifest(job_dir, manifest)
            output_files.append(manifest_path)
            line_choices = self._build_line_choices_from_manifest(manifest)
            line_dropdown = gr.Dropdown(choices=line_choices, value=line_choices[0] if line_choices else None)

            summary = (
                f"✅ 다화자 대본 생성 완료\n"
                f"대사 줄 수: {len(result_rows)}\n"
                f"화자 수: {len(speaker_configs)}\n"
                f"작업 폴더: {job_dir}"
            )
            return final_mix_path, output_files, summary, result_rows, job_dir, line_dropdown
        except Exception as exc:
            return None, output_files, f"❌ 다화자 생성 실패: {type(exc).__name__}: {exc}", result_rows, job_dir, gr.Dropdown(choices=[], value=None)

    def preview_multi_speaker_line(
        self,
        job_dir: str,
        selected_line: str,
    ) -> Tuple[Optional[str], str, str, str]:
        if not job_dir or not str(job_dir).strip():
            return None, "", "", "❌ 먼저 다화자 대본을 생성해주세요."
        if not selected_line or not str(selected_line).strip():
            return None, "", "", "❌ 미리들을 줄을 선택해주세요."

        try:
            manifest = self._read_manifest(job_dir)
            line_id = self._parse_line_choice(selected_line)
            line_entry = self._find_manifest_line(manifest, line_id)
            audio_path = line_entry.get("audio_path")
            if not audio_path or not os.path.exists(audio_path):
                return None, line_entry.get("speaker", ""), line_entry.get("text", ""), "❌ 선택한 줄의 오디오 파일이 없습니다."
            status = (
                f"✅ {line_entry.get('line_id')} 미리듣기 준비 완료\n"
                f"화자: {line_entry.get('speaker')}\n"
                f"선택 버전: {line_entry.get('selected_version', 'v1')}"
            )
            return audio_path, line_entry.get("speaker", ""), line_entry.get("text", ""), status
        except Exception as exc:
            return None, "", "", f"❌ 줄 미리듣기 실패: {type(exc).__name__}: {exc}"

    def regenerate_multi_speaker_line(
        self,
        job_dir: str,
        selected_line: str,
        speaker_rows,
        language: str,
        merge_output: bool = True,
        silence_ms: int = 120,
    ) -> Tuple[Optional[str], Optional[str], List[str], str, List[List[Any]], gr.Dropdown]:
        if self.model is None:
            return None, None, [], "❌ 먼저 TTS 모델을 로드해주세요!", [], gr.Dropdown(choices=[], value=None)
        if not job_dir or not str(job_dir).strip():
            return None, None, [], "❌ 먼저 다화자 대본을 생성해주세요.", [], gr.Dropdown(choices=[], value=None)
        if not selected_line or not str(selected_line).strip():
            return None, None, [], "❌ 다시 생성할 줄을 선택해주세요.", [], gr.Dropdown(choices=[], value=None)

        try:
            speaker_configs = self._normalize_speaker_rows(speaker_rows)
            manifest = self._read_manifest(job_dir)
            line_id = self._parse_line_choice(selected_line)
            line_entry = self._find_manifest_line(manifest, line_id)
            speaker = line_entry.get("speaker", "")
            if speaker not in speaker_configs:
                raise ValueError(f"화자 '{speaker}'의 설정이 현재 표에 없습니다.")

            speaker_config = speaker_configs[speaker]
            version_id, version_index = self._next_line_version(line_entry)
            processed_wav, sample_rate, chunk_count = self._render_line_with_prompt(
                text=line_entry.get("text", ""),
                language=language,
                prompt_path=speaker_config["prompt_path"],
                options=speaker_config["options"],
            )
            line_number = int(re.sub(r"\D", "", line_id) or "0")
            file_name = self._build_multi_speaker_filename(
                line_index=line_number,
                speaker=speaker,
                text=line_entry.get("text", ""),
                version_id=version_id,
            )
            line_path = os.path.join(job_dir, file_name)
            self._save_wav_to_path(processed_wav, sample_rate, line_path)

            line_entry["audio_path"] = line_path
            line_entry["chunk_count"] = chunk_count
            line_entry["status"] = "완료"
            line_entry["selected_version"] = version_id
            line_entry.setdefault("versions", []).append(
                {
                    "version_id": version_id,
                    "audio_path": line_path,
                    "chunk_count": chunk_count,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            manifest["speakers"] = {
                speaker_name: {
                    "prompt_path": config["prompt_path"],
                    "options": asdict(config["options"]),
                }
                for speaker_name, config in speaker_configs.items()
            }

            output_files = [line_path]
            final_mix_path = manifest.get("final_mix_path")
            if merge_output:
                merged_wavs: List[np.ndarray] = []
                sample_rate_for_merge: Optional[int] = None
                for manifest_line in manifest.get("lines", []):
                    current_path = manifest_line.get("audio_path")
                    if not current_path or not os.path.exists(current_path):
                        raise ValueError(f"합본에 필요한 줄 파일이 없습니다: {manifest_line.get('line_id')}")
                    wav, current_sr = sf.read(current_path, dtype="float32")
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    sample_rate_for_merge = current_sr
                    merged_wavs.append(wav.astype(np.float32))
                if sample_rate_for_merge is not None:
                    merged_wav = self._merge_wavs_with_silence(merged_wavs, sample_rate_for_merge, silence_ms)
                    final_mix_path = os.path.join(job_dir, "final_mix.wav")
                    self._save_wav_to_path(merged_wav, sample_rate_for_merge, final_mix_path)
                    manifest["final_mix_path"] = final_mix_path
                    output_files.append(final_mix_path)

            manifest_path = self._write_manifest(job_dir, manifest)
            output_files.append(manifest_path)
            result_rows = self._build_result_rows_from_manifest(manifest)
            line_choices = self._build_line_choices_from_manifest(manifest)
            line_dropdown = gr.Dropdown(choices=line_choices, value=selected_line if selected_line in line_choices else (line_choices[0] if line_choices else None))
            status = (
                f"✅ {line_id} 다시 생성 완료\n"
                f"화자: {speaker}\n"
                f"새 버전: {version_id} ({version_index}번째 렌더)"
            )
            return line_path, final_mix_path, output_files, status, result_rows, line_dropdown
        except Exception as exc:
            return None, None, [], f"❌ 줄 다시 생성 실패: {type(exc).__name__}: {exc}", [], gr.Dropdown(choices=[], value=None)

    def _generate_voice_clone_wav(
        self,
        text: str,
        language: str,
        audio_input,
        reference_text: Optional[str],
        x_vector_only_mode: bool,
        voice_clone_prompt=None,
    ) -> Tuple[np.ndarray, int, int]:
        text_chunks = self._split_text(text)
        generated_wavs: List[np.ndarray] = []
        sample_rate: Optional[int] = None

        for chunk in text_chunks:
            kwargs = {
                "text": chunk,
                "language": self.LANGUAGE_MAP.get(language),
                "non_streaming_mode": True,
                "max_new_tokens": self._estimate_max_new_tokens(chunk),
            }
            if voice_clone_prompt is not None:
                kwargs["voice_clone_prompt"] = voice_clone_prompt
            else:
                kwargs["ref_audio"] = audio_input
                kwargs["ref_text"] = reference_text
                kwargs["x_vector_only_mode"] = bool(x_vector_only_mode)

            wavs, current_sample_rate = self.model.generate_voice_clone(**kwargs)
            sample_rate = current_sample_rate
            generated_wavs.append(wavs[0])

        return self._concat_wavs(generated_wavs, sample_rate), sample_rate, len(text_chunks)

    def generate_voice_clone(
        self,
        text: str,
        language: str,
        reference_audio,
        reference_text: str = "",
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        ending_style: str = "default",
        ending_length_ms: int = 180,
    ) -> Tuple[Optional[str], str]:
        if self.model is None:
            return None, "❌ 먼저 TTS 모델을 로드해주세요!"
        if not text or not text.strip():
            return None, "❌ 합성할 텍스트를 입력해주세요."

        try:
            audio_input = self._normalize_audio_input(reference_audio)
            if audio_input is None:
                return None, "❌ 참조 음성을 업로드하거나 녹음해주세요."

            if not x_vector_only_mode and not reference_text.strip():
                return None, "❌ 정밀 voice clone 모드에서는 참조 음성 텍스트가 필요합니다."

            wav, sample_rate, chunk_count = self._generate_voice_clone_wav(
                text=text.strip(),
                language=language,
                audio_input=audio_input,
                reference_text=reference_text.strip() or None,
                x_vector_only_mode=bool(x_vector_only_mode),
            )
            output_path = self._save_wav(wav, sample_rate, text, "tts_clone_")
            mode_name = "x-vector only" if x_vector_only_mode else "reference text"
            return output_path, f"✅ 원본 음성 생성 완료 ({mode_name} 모드, {chunk_count}개 구간 분할)"
        except Exception as exc:
            return None, f"❌ 음성 생성 실패: {type(exc).__name__}: {exc}"

    def save_voice_prompt(
        self,
        reference_audio,
        reference_text: str = "",
        x_vector_only_mode: bool = False,
    ) -> Tuple[Optional[str], str]:
        if self.model is None:
            return None, "❌ 먼저 TTS 모델을 로드해주세요!"

        try:
            audio_input = self._normalize_audio_input(reference_audio)
            if audio_input is None:
                return None, "❌ 참조 음성이 필요합니다."
            if not x_vector_only_mode and not reference_text.strip():
                return None, "❌ 프롬프트 저장 시 참조 텍스트를 입력해주세요."

            items = self.model.create_voice_clone_prompt(
                ref_audio=audio_input,
                ref_text=reference_text.strip() or None,
                x_vector_only_mode=bool(x_vector_only_mode),
            )
            fd, path = tempfile.mkstemp(prefix="voice_prompt_", suffix=".pt", dir=self.output_dir)
            os.close(fd)
            torch.save({"items": [asdict(item) for item in items]}, path)
            return path, "✅ 재사용 가능한 음성 프롬프트 저장 완료"
        except Exception as exc:
            return None, f"❌ 프롬프트 저장 실패: {type(exc).__name__}: {exc}"

    def generate_from_prompt_file(
        self,
        prompt_file,
        text: str,
        language: str,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        ending_style: str = "default",
        ending_length_ms: int = 180,
    ) -> Tuple[Optional[str], str]:
        if self.model is None:
            return None, "❌ 먼저 TTS 모델을 로드해주세요!"
        if not prompt_file:
            return None, "❌ 저장된 음성 프롬프트 파일을 선택해주세요."
        if not text or not text.strip():
            return None, "❌ 합성할 텍스트를 입력해주세요."

        try:
            prompt_path = getattr(prompt_file, "name", None) or getattr(prompt_file, "path", None) or str(prompt_file)
            items = self._load_voice_prompt_items(prompt_path)
            wav, sample_rate, chunk_count = self._generate_voice_clone_wav(
                text=text.strip(),
                language=language,
                audio_input=None,
                reference_text=None,
                x_vector_only_mode=False,
                voice_clone_prompt=items,
            )
            output_path = self._save_wav(wav, sample_rate, text, "tts_prompt_")
            return output_path, f"✅ 저장된 프롬프트로 원본 음성 생성 완료 ({chunk_count}개 구간 분할)"
        except Exception as exc:
            return None, f"❌ 프롬프트 사용 실패: {type(exc).__name__}: {exc}"


tts_model = TTSModel()


def get_model_choices():
    return list(TTSModel.MODELS.keys())


def resolve_server_name(default_name: str = "127.0.0.1") -> str:
    return os.getenv("GRADIO_SERVER_NAME", default_name)


def resolve_server_port(default_port: int = 7862) -> int:
    port = int(os.getenv("GRADIO_SERVER_PORT", str(default_port)))
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1

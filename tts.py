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
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
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
            payload = torch.load(prompt_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or "items" not in payload:
                return None, "❌ 올바른 voice prompt 파일이 아닙니다."

            items = [VoiceClonePromptItem(**item) for item in payload["items"]]
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

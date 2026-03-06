"""
Qwen3-TTS model wrapper for local Korean-first voice cloning.
"""

from __future__ import annotations

import os
import socket
import tempfile
from dataclasses import asdict
from importlib.util import find_spec
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch

try:
    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
except ImportError:  # pragma: no cover - surfaced in UI status
    Qwen3TTSModel = None
    VoiceClonePromptItem = None


AudioTuple = Tuple[np.ndarray, int]


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
        self.output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

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

            models_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(models_dir, exist_ok=True)
            os.environ["HF_HOME"] = models_dir

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

    def _save_wav(self, wav: np.ndarray, sample_rate: int, prefix: str) -> str:
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav", dir=self.output_dir)
        os.close(fd)
        sf.write(path, wav, sample_rate)
        return path

    def generate_voice_clone(
        self,
        text: str,
        language: str,
        reference_audio,
        reference_text: str = "",
        x_vector_only_mode: bool = False,
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

            wavs, sample_rate = self.model.generate_voice_clone(
                text=text.strip(),
                language=self.LANGUAGE_MAP.get(language),
                ref_audio=audio_input,
                ref_text=reference_text.strip() or None,
                x_vector_only_mode=bool(x_vector_only_mode),
            )
            output_path = self._save_wav(wavs[0], sample_rate, "tts_clone_")
            mode_name = "x-vector only" if x_vector_only_mode else "reference text"
            return output_path, f"✅ 음성 생성 완료 ({mode_name} 모드)"
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
            wavs, sample_rate = self.model.generate_voice_clone(
                text=text.strip(),
                language=self.LANGUAGE_MAP.get(language),
                voice_clone_prompt=items,
            )
            output_path = self._save_wav(wavs[0], sample_rate, "tts_prompt_")
            return output_path, "✅ 저장된 프롬프트로 음성 생성 완료"
        except Exception as exc:
            return None, f"❌ 프롬프트 사용 실패: {type(exc).__name__}: {exc}"


tts_model = TTSModel()


def get_model_choices():
    return list(TTSModel.MODELS.keys())


def resolve_server_port(default_port: int = 7862) -> int:
    port = int(os.getenv("GRADIO_SERVER_PORT", str(default_port)))
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1


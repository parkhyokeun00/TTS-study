"""
Experimental RVC-based speech-to-speech voice conversion wrapper.
"""

from __future__ import annotations

import os
import re
import shutil
import traceback
import uuid
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from storage import configure_runtime_storage

try:
    from rvc.modules.vc.modules import VC
except Exception as exc:  # pragma: no cover - surfaced in UI status
    VC = None
    RVC_IMPORT_ERROR = exc
else:
    RVC_IMPORT_ERROR = None


STORAGE_PATHS = configure_runtime_storage()
PROJECT_DIR = os.path.dirname(__file__)
RVC_ROOT = os.path.join(PROJECT_DIR, "models", "rvc")
RVC_ASSETS_DIR = os.path.join(RVC_ROOT, "assets")
RVC_WEIGHTS_DIR = os.path.join(RVC_ASSETS_DIR, "weights")
RVC_INDICES_DIR = os.path.join(RVC_ASSETS_DIR, "indices")
RVC_HUBERT_DIR = os.path.join(RVC_ASSETS_DIR, "hubert")
RVC_RMVPE_DIR = os.path.join(RVC_ASSETS_DIR, "rmvpe")
RVC_OUTPUT_DIR = os.path.join(STORAGE_PATHS["outputs"], "voice_conversion")


def ensure_rvc_dirs() -> None:
    for path in [
        RVC_ROOT,
        RVC_ASSETS_DIR,
        RVC_WEIGHTS_DIR,
        RVC_INDICES_DIR,
        RVC_HUBERT_DIR,
        RVC_RMVPE_DIR,
        RVC_OUTPUT_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


class RVCVoiceConverter:
    F0_METHOD_CHOICES = ["rmvpe", "harvest", "pm", "crepe"]

    def __init__(self) -> None:
        ensure_rvc_dirs()
        self.vc = None
        self.loaded_model_path: Optional[str] = None
        self.last_error: Optional[str] = None

    def _resolve_file_path(self, file_input) -> Optional[str]:
        if not file_input:
            return None
        path = getattr(file_input, "name", None) or getattr(file_input, "path", None) or str(file_input)
        return path if path and os.path.exists(path) else None

    def _sanitize_filename(self, filename: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", os.path.basename(filename or "").strip())
        return cleaned or f"artifact_{uuid.uuid4().hex[:8]}"

    def _copy_if_needed(self, source_path: Optional[str], target_dir: str) -> Optional[str]:
        if not source_path:
            return None
        source_path = os.path.abspath(source_path)
        target_dir = os.path.abspath(target_dir)
        try:
            if os.path.commonpath([source_path, target_dir]) == target_dir:
                return source_path
        except ValueError:
            pass

        target_path = os.path.join(target_dir, self._sanitize_filename(source_path))
        if source_path != target_path:
            shutil.copy2(source_path, target_path)
        return target_path

    def _default_hubert_path(self) -> Optional[str]:
        for candidate in ["hubert_base.pt", "hubert.pt"]:
            path = os.path.join(RVC_HUBERT_DIR, candidate)
            if os.path.exists(path):
                return path
        return None

    def _default_rmvpe_path(self) -> Optional[str]:
        path = os.path.join(RVC_RMVPE_DIR, "rmvpe.pt")
        return path if os.path.exists(path) else None

    def _configure_environment(self, hubert_path: Optional[str]) -> None:
        ensure_rvc_dirs()
        os.environ["weight_root"] = RVC_WEIGHTS_DIR
        os.environ["index_root"] = RVC_INDICES_DIR
        os.environ["rmvpe_root"] = RVC_RMVPE_DIR
        if hubert_path:
            os.environ["hubert_path"] = hubert_path

    def _apply_runtime_compat_patches(self) -> None:
        try:
            from rvc.lib import audio as rvc_audio
            from rvc.modules.vc import modules as rvc_modules
            from fairseq import checkpoint_utils as fairseq_checkpoint_utils
        except Exception:
            return

        original_load_audio = getattr(rvc_audio, "_codex_original_load_audio", rvc_audio.load_audio)

        def patched_load_audio(file, sr):
            if not isinstance(file, str):
                return original_load_audio(file, sr)

            if not os.path.exists(file):
                raise RuntimeError(
                    "You input a wrong audio path that does not exists, please fix it!"
                )

            try:
                audio, source_sr = sf.read(file, always_2d=False)
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if source_sr != sr:
                    audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)
                return audio.astype(np.float32).flatten()
            except Exception:
                try:
                    return original_load_audio(file, sr)
                except Exception:
                    raise RuntimeError(traceback.format_exc())

        rvc_audio.load_audio = patched_load_audio
        try:
            rvc_modules.load_audio = patched_load_audio
        except Exception:
            pass

        original_torch_load = getattr(
            fairseq_checkpoint_utils,
            "_codex_original_torch_load",
            fairseq_checkpoint_utils.torch.load,
        )

        def patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        fairseq_checkpoint_utils.torch.load = patched_torch_load

        rvc_audio._codex_original_load_audio = original_load_audio
        fairseq_checkpoint_utils._codex_original_torch_load = original_torch_load
        rvc_audio._codex_audio_patch_applied = True

    def get_runtime_status(self) -> str:
        if VC is None:
            return (
                "❌ RVC 패키지가 설치되지 않았습니다.\n"
                "권장 환경: Windows에서는 Python 3.10 전용 가상환경\n"
                "설치 절차: README의 RVC 설치 절차 참고\n"
                f"상세: {type(RVC_IMPORT_ERROR).__name__}: {RVC_IMPORT_ERROR}"
            )

        hubert_path = self._default_hubert_path()
        rmvpe_path = self._default_rmvpe_path()
        return (
            "✅ RVC 런타임 사용 가능\n"
            f"모델 폴더: {RVC_WEIGHTS_DIR}\n"
            f"Index 폴더: {RVC_INDICES_DIR}\n"
            f"HuBERT: {'준비됨' if hubert_path else '없음'}\n"
            f"RMVPE: {'준비됨' if rmvpe_path else '없음'}"
        )

    def _save_output_audio(self, source_audio_path: str, audio, sample_rate: int) -> str:
        stem = os.path.splitext(os.path.basename(source_audio_path))[0]
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", stem) or "voice_conversion"
        output_path = os.path.join(RVC_OUTPUT_DIR, f"{stem}_rvc_{uuid.uuid4().hex[:8]}.wav")
        sf.write(output_path, audio, sample_rate)
        return output_path

    def _ensure_model_loaded(self, model_path: str, protect: float) -> Tuple[Optional[str], Optional[float]]:
        try:
            if self.vc is None or self.loaded_model_path != model_path:
                self.vc = VC()
                self.loaded_model_path = model_path
            _, protect_values, auto_index = self.vc.get_vc(model_path, protect, protect)
            self.last_error = None
            protect_value = float(protect_values[1] if isinstance(protect_values, (list, tuple)) else protect)
            return auto_index or None, protect_value
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None, None

    def convert_voice(
        self,
        input_audio,
        model_file,
        index_file=None,
        hubert_file=None,
        rmvpe_file=None,
        speaker_id: int = 0,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ) -> Tuple[Optional[str], str]:
        if VC is None:
            return None, self.get_runtime_status()

        input_audio_path = self._resolve_file_path(input_audio)
        if not input_audio_path:
            return None, "❌ 변조할 입력 음성을 업로드하거나 녹음해주세요."

        model_path = self._copy_if_needed(self._resolve_file_path(model_file), RVC_WEIGHTS_DIR)
        if not model_path or not model_path.endswith(".pth"):
            return None, "❌ RVC 모델(.pth) 파일이 필요합니다."

        explicit_index_path = self._copy_if_needed(self._resolve_file_path(index_file), RVC_INDICES_DIR)
        hubert_path = self._copy_if_needed(self._resolve_file_path(hubert_file), RVC_HUBERT_DIR) or self._default_hubert_path()
        rmvpe_path = self._copy_if_needed(self._resolve_file_path(rmvpe_file), RVC_RMVPE_DIR) or self._default_rmvpe_path()

        if not hubert_path:
            return None, (
                "❌ HuBERT base 모델이 없습니다.\n"
                f"파일을 업로드하거나 {RVC_HUBERT_DIR} 에 hubert_base.pt 를 배치하세요."
            )

        if f0_method == "rmvpe" and not rmvpe_path:
            return None, (
                "❌ RMVPE 모델이 없습니다.\n"
                f"파일을 업로드하거나 {RVC_RMVPE_DIR} 에 rmvpe.pt 를 배치하세요."
            )

        self._configure_environment(hubert_path)
        self._apply_runtime_compat_patches()

        try:
            auto_index_path, protect_value = self._ensure_model_loaded(model_path, float(protect))
            if protect_value is None:
                return None, f"❌ RVC 모델 로드 실패: {os.path.basename(model_path)}\n{self.last_error or ''}".strip()

            chosen_index_path = explicit_index_path or auto_index_path
            target_sr, audio_opt, times, error = self.vc.vc_inference(
                sid=int(speaker_id),
                input_audio_path=input_audio_path,
                f0_up_key=int(pitch_shift),
                f0_method=str(f0_method),
                index_file=chosen_index_path,
                index_rate=float(index_rate),
                filter_radius=int(filter_radius),
                resample_sr=int(resample_sr),
                rms_mix_rate=float(rms_mix_rate),
                protect=float(protect_value),
                hubert_path=hubert_path,
            )
            if error or target_sr is None or audio_opt is None:
                return None, f"❌ 음성 변조 실패\n{error or '알 수 없는 오류'}"

            output_path = self._save_output_audio(input_audio_path, audio_opt, int(target_sr))
            index_label = os.path.basename(chosen_index_path) if chosen_index_path else "미사용"
            timing_label = ""
            if isinstance(times, dict):
                timing_label = f"\n처리 시간 단서: npy={times.get('npy', 0):.2f}, f0={times.get('f0', 0):.2f}, infer={times.get('infer', 0):.2f}"
            return (
                output_path,
                "✅ 음성 변조 완료\n"
                f"모델: {os.path.basename(model_path)}\n"
                f"Index: {index_label}\n"
                f"샘플레이트: {target_sr}Hz"
                f"{timing_label}",
            )
        except Exception as exc:
            return None, f"❌ 음성 변조 실패: {type(exc).__name__}: {exc}"


voice_converter = RVCVoiceConverter()

"""
Qwen3-ASR 모델 로딩 및 음성 인식 모듈
qwen-asr 패키지 사용
"""

import torch
import numpy as np
import tempfile
import soundfile as sf
from typing import Optional, Tuple


class ASRModel:
    """Qwen3-ASR 음성 인식 모델 래퍼"""
    
    MODELS = {
        "Qwen3-ASR-1.7B (고정확도)": "Qwen/Qwen3-ASR-1.7B",
        "Qwen3-ASR-0.6B (경량/빠름)": "Qwen/Qwen3-ASR-0.6B",
    }
    
    LANGUAGE_MAP = {
        "korean": "Korean",
        "japanese": "Japanese",
        "english": "English",
    }
    
    def __init__(self):
        self.model = None
        self.device = None
        self.current_model_name = None
    
    def get_device(self):
        """사용 가능한 디바이스 확인"""
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"
    
    def load_model(self, model_choice: str) -> str:
        """
        모델 로드
        
        Args:
            model_choice: 모델 선택 (UI에서 선택한 값)
        
        Returns:
            상태 메시지
        """
        if model_choice == self.current_model_name and self.model is not None:
            return f"✅ {model_choice} 이미 로드됨"
        
        model_id = self.MODELS.get(model_choice)
        if not model_id:
            return f"❌ 알 수 없는 모델: {model_choice}"
        
        self.device = self.get_device()
        device_name = "GPU (CUDA)" if "cuda" in self.device else "CPU"
        
        try:
            print(f"🔄 {model_choice} 로딩 중... ({device_name})")
            
            # 이전 모델 메모리 해제
            if self.model is not None:
                del self.model
                if "cuda" in self.device:
                    torch.cuda.empty_cache()
            
            # qwen-asr 패키지에서 모델 로드
            from qwen_asr import Qwen3ASRModel
            import os
            
            # 모델 저장 경로 설정 (프로젝트 폴더 내 models/)
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(models_dir, exist_ok=True)
            os.environ["HF_HOME"] = models_dir
            
            dtype = torch.bfloat16 if "cuda" in self.device else torch.float32
            self.model = Qwen3ASRModel.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=self.device,
                max_inference_batch_size=1,
                max_new_tokens=256,
            )
            
            self.current_model_name = model_choice
            
            return f"✅ {model_choice} 로드 완료!\n📍 디바이스: {device_name}"
            
        except ImportError:
            return "❌ qwen-asr 패키지가 설치되지 않았습니다.\n💡 실행: pip install -U qwen-asr"
        except Exception as e:
            return f"❌ 모델 로드 실패: {str(e)}"
    
    def transcribe(self, audio_path: str, language: str = "korean") -> str:
        """
        오디오 파일을 텍스트로 변환
        
        Args:
            audio_path: 오디오 파일 경로
            language: 인식 언어 ("korean" 또는 "japanese")
        
        Returns:
            변환된 텍스트
        """
        if self.model is None:
            return "❌ 먼저 모델을 로드해주세요!"
        
        try:
            lang = self.LANGUAGE_MAP.get(language, None)
            
            # 변환 수행
            results = self.model.transcribe(
                audio=audio_path,
                language=lang,
            )
            
            if results and len(results) > 0:
                return results[0].text.strip()
            else:
                return "❌ 변환 결과가 없습니다."
            
        except Exception as e:
            return f"❌ 변환 오류: {str(e)}"
    
    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int, language: str = "korean") -> str:
        """
        오디오 배열을 텍스트로 변환 (마이크 입력용)
        
        Args:
            audio_array: numpy 오디오 배열
            sample_rate: 샘플레이트
            language: 인식 언어
        
        Returns:
            변환된 텍스트
        """
        if self.model is None:
            return "❌ 먼저 모델을 로드해주세요!"
        
        try:
            # 모노로 변환
            if len(audio_array.shape) > 1:
                audio = audio_array.mean(axis=1)
            else:
                audio = audio_array
            
            # float32로 변환 및 정규화
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, sample_rate)
                temp_path = f.name
            
            # 변환 수행
            result = self.transcribe(temp_path, language)
            
            # 임시 파일 삭제
            import os
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            return f"❌ 변환 오류: {str(e)}"


# 전역 모델 인스턴스
asr_model = ASRModel()


def get_model_choices():
    """모델 선택 목록 반환"""
    return list(ASRModel.MODELS.keys())


def load_model(model_choice: str) -> str:
    """모델 로드 (Gradio 콜백용)"""
    return asr_model.load_model(model_choice)


def transcribe_file(audio_path: str, language: str) -> str:
    """파일 변환 (Gradio 콜백용)"""
    if audio_path is None:
        return ""
    return asr_model.transcribe(audio_path, language)


def transcribe_mic(audio_tuple, language: str) -> str:
    """마이크 입력 변환 (Gradio 콜백용)"""
    if audio_tuple is None:
        return ""
    sample_rate, audio_array = audio_tuple
    return asr_model.transcribe_array(audio_array, sample_rate, language)

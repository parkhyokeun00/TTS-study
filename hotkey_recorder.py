"""
글로벌 단축키 녹음 기능
`+1: 녹음 시작/종료
`+2: 마지막 변환 텍스트 클립보드 복사
"""

import keyboard
import sounddevice as sd
import numpy as np
import threading
import queue
import tempfile
import soundfile as sf
import pyperclip
from datetime import datetime
from storage import configure_runtime_storage


STORAGE_PATHS = configure_runtime_storage()


class HotkeyRecorder:
    """글로벌 단축키 기반 녹음기"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recorded_audio = []
        self.stream = None
        self.callback = None  # 녹음 완료 시 호출할 콜백
        self.last_transcription = ""  # 마지막 변환 텍스트
        
    def set_callback(self, callback):
        """녹음 완료 시 호출할 콜백 설정"""
        self.callback = callback
    
    def set_last_transcription(self, text):
        """마지막 변환 텍스트 저장"""
        self.last_transcription = text
    
    def copy_last_transcription(self):
        """마지막 변환 텍스트를 클립보드에 복사"""
        if self.last_transcription:
            pyperclip.copy(self.last_transcription)
            print(f"📋 클립보드에 복사됨: {self.last_transcription[:50]}...")
        else:
            print("⚠️ 복사할 텍스트가 없습니다.")
    
    def audio_callback(self, indata, frames, time, status):
        """오디오 스트림 콜백"""
        if self.is_recording:
            self.audio_queue.put(indata.copy())
    
    def start_recording(self):
        """녹음 시작"""
        if self.is_recording:
            print("⚠️ 이미 녹음 중입니다.")
            return
        
        self.is_recording = True
        self.recorded_audio = []
        self.audio_queue = queue.Queue()
        
        # 오디오 스트림 시작
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self.audio_callback
        )
        self.stream.start()
        
        print("🔴 녹음 시작! (`+1을 눌러 종료)")
        
    def stop_recording(self):
        """녹음 종료"""
        if not self.is_recording:
            print("⚠️ 녹음 중이 아닙니다.")
            return None
        
        self.is_recording = False
        
        # 스트림 종료
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # 큐에서 오디오 데이터 수집
        while not self.audio_queue.empty():
            self.recorded_audio.append(self.audio_queue.get())
        
        if not self.recorded_audio:
            print("⚠️ 녹음된 오디오가 없습니다.")
            return None
        
        # 오디오 배열 합치기
        audio_data = np.concatenate(self.recorded_audio, axis=0)
        
        # 임시 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f"_recording_{timestamp}.wav",
            delete=False,
            dir=STORAGE_PATHS["tmp"],
        )
        sf.write(temp_file.name, audio_data, self.sample_rate)
        
        print(f"⏹️ 녹음 종료! 파일: {temp_file.name}")
        
        # 콜백 호출 (있으면)
        if self.callback:
            self.callback(temp_file.name, audio_data, self.sample_rate)
        
        return temp_file.name
    
    def toggle_recording(self):
        """녹음 토글 (시작/종료)"""
        if self.is_recording:
            return self.stop_recording()
        else:
            self.start_recording()
            return None


# 전역 녹음기 인스턴스
recorder = HotkeyRecorder()


def on_record_hotkey():
    """녹음 단축키 (`+1) 눌렸을 때 호출"""
    recorder.toggle_recording()


def on_copy_hotkey():
    """복사 단축키 (`+2) 눌렸을 때 호출"""
    recorder.copy_last_transcription()


def setup_hotkeys():
    """글로벌 단축키 설정"""
    # `+1: 녹음 시작/종료
    keyboard.add_hotkey('`+1', on_record_hotkey)
    # `+2: 텍스트 클립보드 복사
    keyboard.add_hotkey('`+2', on_copy_hotkey)
    
    print("✅ 단축키 등록 완료:")
    print("   `+1 : 녹음 시작/종료")
    print("   `+2 : 마지막 텍스트 클립보드 복사")
    print("💡 종료하려면 Ctrl+C를 누르세요.")


def run_hotkey_listener(asr_transcribe_func=None, language="korean"):
    """
    단축키 리스너 실행
    
    Args:
        asr_transcribe_func: 녹음 완료 시 호출할 ASR 함수 (optional)
        language: 인식 언어 (default: "korean")
    """
    
    def on_recording_complete(filepath, audio_data, sample_rate):
        """녹음 완료 시 자동 변환"""
        if asr_transcribe_func:
            print("🔄 음성 변환 중...")
            result = asr_transcribe_func(filepath, language)
            print(f"📝 변환 결과: {result}")
            # 마지막 변환 텍스트 저장
            recorder.set_last_transcription(result)
            print("💡 `+2를 눌러 클립보드에 복사할 수 있습니다.")
    
    if asr_transcribe_func:
        recorder.set_callback(on_recording_complete)
    
    setup_hotkeys()
    
    try:
        keyboard.wait()  # 프로그램 유지
    except KeyboardInterrupt:
        print("\n👋 프로그램 종료")


if __name__ == "__main__":
    # 독립 실행 테스트
    print("=" * 50)
    print("🎤 단축키 녹음기 테스트")
    print("=" * 50)
    run_hotkey_listener()

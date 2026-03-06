import sys
import threading
import time

# IMPORT ASR (Torch) FIRST to prevent Windows DLL initialization Error 1114 with PyQt5
import asr

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QSystemTrayIcon, QMenu, QAction, QStyle
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor

from system_audio import SystemAudioRecorder
from translator import translate

class SignalEmitter(QObject):
    update_text = pyqtSignal(str, str)
    update_status = pyqtSignal(str)

class SubtitleOverlay(QWidget):
    def __init__(self, emitter):
        super().__init__()
        self.emitter = emitter
        
        # 윈도우 스타일 설정: 테두리 없음, 항상 위, 투명 배경, 클릭 무시
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Tool |
            Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 레이아웃 생성
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter | Qt.AlignBottom)
        
        # 원본 언어 라벨
        self.original_label = QLabel(" ")
        self.original_label.setStyleSheet(
            "color: white; "
            "background-color: rgba(0, 0, 0, 150); "
            "padding: 5px; "
            "border-radius: 10px;"
        )
        original_font = QFont("Malgun Gothic", 16)
        self.original_label.setFont(original_font)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setWordWrap(True)
        
        # 번역 언어 라벨 (한국어)
        self.translated_label = QLabel("🎤 시스템 오디오 번역 대기 중...")
        self.translated_label.setStyleSheet(
            "color: #FFD700; " # 노란색 
            "background-color: rgba(0, 0, 0, 180); "
            "padding: 8px; "
            "border-radius: 10px;"
        )
        translated_font = QFont("Malgun Gothic", 24, QFont.Bold)
        self.translated_label.setFont(translated_font)
        self.translated_label.setAlignment(Qt.AlignCenter)
        self.translated_label.setWordWrap(True)
        
        self.layout.addWidget(self.original_label)
        self.layout.addWidget(self.translated_label)
        self.setLayout(self.layout)
        
        # 화면 중앙 하단 배치
        screen = QApplication.primaryScreen().geometry()
        window_width = int(screen.width() * 0.8)
        window_height = 250

        # 시스템 트레이 아이콘 추가 (종료 기능)
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_DesktopIcon))
        self.tray_menu = QMenu()
        self.quit_action = QAction("프로그램 종료 (Exit)", self)
        self.quit_action.triggered.connect(QApplication.instance().quit)
        self.tray_menu.addAction(self.quit_action)
        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.setToolTip("Qwen3-ASR 자막 번역기 (우클릭하여 종료)")
        self.tray_icon.show()
        self.setGeometry(
            int((screen.width() - window_width) / 2),
            screen.height() - window_height - 100,
            window_width,
            window_height
        )

        # 시그널 연결
        self.emitter.update_text.connect(self.on_update_text)
        self.emitter.update_status.connect(self.on_update_status)
        
        # 자막 초기화용 타이머
        self.clear_timer = QTimer()
        self.clear_timer.timeout.connect(self.clear_subtitle)
        self.clear_timer.setSingleShot(True)

    def on_update_text(self, original, translated):
        if not original:
            return
        self.original_label.setText(original)
        self.translated_label.setText(translated)
        # 자막이 10초간 업데이트 안 되면 지우기
        self.clear_timer.start(10000)

    def on_update_status(self, status):
        self.translated_label.setText(status)
        self.original_label.setText(" ")

    def clear_subtitle(self):
        self.original_label.setText(" ")
        self.translated_label.setText(" ")


def audio_processing_thread(emitter, target_language="japanese"):
    """
    백그라운드에서 오디오를 수집하고 번역하는 스레드
    """
    recorder = SystemAudioRecorder(sample_rate=16000)
    recorder.start()
    
    # 모델 로드 상태 갱신
    emitter.update_status.emit(f"⏳ Qwen3-ASR 모델 로딩 중 ({target_language})...")
    # ASR 모델 로드 (가장 가벼운 0.6B 권장, 아니면 1.7B)
    model_choice = "Qwen3-ASR-0.6B (경량/빠름)" 
    # 혹은 asr.get_model_choices()[0]
    res_msg = asr.load_model(model_choice)
    emitter.update_status.emit(f"✅ 모델 로드 완료! 영상을 재생해주세요.\n({res_msg})")
    
    while True:
        chunk = recorder.get_audio_chunk()
        if chunk:
            sample_rate, audio_data = chunk
            # 1. ASR - 텍스트 인식 (인식 대상 언어)
            try:
                original_text = asr.asr_model.transcribe_array(audio_data, sample_rate, target_language)
            except Exception as e:
                print(f"ASR Error: {e}")
                continue
            
            if not original_text or original_text.startswith("❌"):
                continue
                
            print(f"인식: {original_text}")
            
            # 2. 번역 (한국어로)
            translated_text = translate(original_text)
            print(f"번역: {translated_text}")
            
            # 3. UI 업데이트
            emitter.update_text.emit(original_text, translated_text)
            
        time.sleep(0.1)


def start_subtitle_app():
    # 콘솔에서 대상 언어 선택
    print("=" * 60)
    print("        🎬 실시간 자막 & 번역 시스템 (Qwen3-ASR)")
    print("=" * 60)
    
    target_language = "japanese"
    if len(sys.argv) > 1:
        lang_arg = sys.argv[1].lower()
        if lang_arg in ["1", "japanese", "jp", "ja"]:
            target_language = "japanese"
        elif lang_arg in ["2", "english", "en"]:
            target_language = "english"
        elif lang_arg in ["3", "korean", "ko"]:
            target_language = "korean"
            
    print(f"👉 [{target_language}] 언어로 인식 및 한국어 번역을 시작합니다.")
    print("사용법: python subtitle_app.py [japanese|english|korean]")
    print("❌ 프로그램 종료 단축키: [Ctrl + Alt + Q]")

    app = QApplication(sys.argv)
    emitter = SignalEmitter()
    
    overlay = SubtitleOverlay(emitter)
    overlay.show()
    
    # 처리 스레드 시작
    processor = threading.Thread(target=audio_processing_thread, args=(emitter, target_language), daemon=True)
    processor.start()
    
    # 글로벌 단축키로 강제 종료 (어디서든 Ctrl+Alt+Q 누르면 즉시 꺼짐)
    import keyboard
    import os
    keyboard.add_hotkey("ctrl+alt+q", lambda: os._exit(0))
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    start_subtitle_app()

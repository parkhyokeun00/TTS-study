import numpy as np
import soundcard as sc
import time
import queue
import threading

class SystemAudioRecorder:
    def __init__(self, sample_rate=16000, silence_threshold=0.001, silence_duration=1.0, max_duration=10.0):
        """
        시스템 오디오(Loopback) 캡처기
        :param sample_rate: 캡처할 샘플레이트
        :param silence_threshold: 무음 감지 임계값 (RMS)
        :param silence_duration: 이 시간(초) 동안 무음이면 하나의 청크로 자름
        :param max_duration: 최대 녹음 시간(초). 이 시간이 넘어가면 무음이 아니어도 강제로 자름.
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.thread = None

    def start(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_recording = False
        if self.thread:
            self.thread.join()

    def _record_loop(self):
        try:
            # 윈도우 환경에서 기본 스피커의 루프백(Loopback) 마이크 가져오기
            default_speaker = sc.default_speaker()
            mic = sc.get_microphone(default_speaker.id, include_loopback=True)
            print(f"🎤 오디오 캡처 시작: {mic.name}")
        except Exception as e:
            print(f"❌ 루프백 장치를 찾을 수 없습니다: {e}")
            return

        with mic.recorder(samplerate=self.sample_rate, channels=1) as recorder_:
            # 100ms 단위 청크
            chunk_size = int(self.sample_rate * 0.1)
            
            buffer = []
            silence_frames = 0
            is_speaking = False
            
            while self.is_recording:
                # 데이터 읽기 (블로킹)
                data = recorder_.record(numframes=chunk_size)
                # 모노
                if len(data.shape) > 1:
                    data = data[:, 0]
                
                rms = np.sqrt(np.mean(data**2))
                
                if rms > self.silence_threshold:
                    if not is_speaking:
                        is_speaking = True
                        silence_frames = 0
                        buffer.append(data)
                    else:
                        buffer.append(data)
                        silence_frames = 0
                else:
                    if is_speaking:
                        buffer.append(data)
                        silence_frames += 1
                        
                        current_duration = len(buffer) * 0.1
                        
                        # 무음 지속 시간이 기준을 넘었거나, 최대 녹음 시간을 초과했을 때 처리
                        if (silence_frames * 0.1) > self.silence_duration or current_duration > self.max_duration:
                            audio_chunk = np.concatenate(buffer)
                            self.audio_queue.put((self.sample_rate, audio_chunk))
                            buffer = []
                            is_speaking = False
                            silence_frames = 0
                    else:
                        # 소리가 없는 상태 유지
                        pass

    def get_audio_chunk(self):
        """큐에서 수집된 오디오 청크를 반환 (비블로킹)"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

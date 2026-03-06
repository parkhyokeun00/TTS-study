# 🎤 Qwen3-ASR 음성-텍스트 변환기

Qwen3-ASR 모델을 사용한 한국어/일본어 음성 인식 앱

## 📦 설치

```bash
# 필수 패키지 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install qwen-asr torch torchaudio sounddevice keyboard pyperclip gradio
```

## 🚀 실행 방법

### 방법 1: 웹 GUI (Gradio)

```bash
python app.py
```

- 브라우저에서 `http://localhost:7861` 접속
- 모델 선택 후 "모델 로드" 클릭
- 마이크 녹음 또는 파일 업로드로 음성 변환

### 방법 1-2: TTS / 음성 복제 GUI (Qwen3-TTS)

```bash
pip install -r requirements-tts.txt
python tts_app.py
```

- 브라우저에서 `http://localhost:7862` 또는 자동으로 잡힌 다음 포트 접속
- `Qwen3-TTS-1.7B Base` 로드 후 한국어 원고 입력
- 참조 음성과 참조 텍스트를 넣으면 정밀 voice clone 가능
- 참조 텍스트 없이 `x-vector only` 모드도 가능하지만 품질은 더 불안정할 수 있음
- `voice prompt` 파일로 저장해두면 같은 목소리로 여러 문장을 반복 생성 가능

> ⚠️ `qwen-asr` 와 `qwen-tts` 는 `transformers` 버전 충돌 가능성이 있으므로 가급적 **별도 가상환경** 사용 권장
>
> ⚠️ Windows에서는 `SoX` 실행파일이 없으면 경고가 뜰 수 있습니다. 필요 시 https://sox.sourceforge.net/ 설치 후 PATH에 추가하세요.

### 방법 2: 단축키 모드 (추천)

```bash
python hotkey_app.py
```

어디서든 단축키로 녹음하고 바로 텍스트로 변환!

## ⌨️ 단축키

| 단축키 | 기능 |
|--------|------|
| `` ` + 1 `` | 녹음 시작/종료 |
| `` ` + 2 `` | 마지막 텍스트 클립보드 복사 |

> ⚠️ Windows에서는 **관리자 권한**으로 실행 필요

## 📝 사용 예시

1. `python hotkey_app.py` 실행
2. 모델 선택 (1.7B 권장)
3. 언어 선택 (한국어/일본어)
4. `` ` + 1 `` 눌러서 🔴 녹음 시작
5. 말하기
6. `` ` + 1 `` 눌러서 ⏹️ 녹음 종료
7. 자동으로 텍스트 변환됨
8. `` ` + 2 `` 눌러서 📋 복사

## 🤖 지원 모델

| 모델 | 특징 |
|------|------|
| Qwen3-ASR-1.7B | 높은 정확도 (권장) |
| Qwen3-ASR-0.6B | 빠른 속도 |

## 🌐 지원 언어

- 🇰🇷 한국어
- 🇯🇵 일본어

## 📁 파일 구조

```
qwen-tts/
├── app.py              # Gradio 웹 UI
├── hotkey_app.py       # 단축키 모드 실행
├── hotkey_recorder.py  # 단축키 녹음 모듈
├── asr.py              # ASR 모델 래퍼
├── requirements.txt    # 의존성 패키지
├── models/             # 다운로드된 모델 (자동 생성)
└── README.md           # 사용 설명서
```

## 💡 팁

- 첫 실행 시 모델 다운로드에 시간이 걸립니다 (약 3~7GB)
- GPU (CUDA)가 있으면 훨씬 빠릅니다
- 모델은 `models/` 폴더에 저장됩니다

# Qwen3-TTS 한국어 음성 복제 스튜디오

이 프로젝트의 중심 프로그램은 [tts_app.py](/f:/abc/qwen-tts/tts_app.py) 입니다.

핵심 목적:
- 한국어 텍스트를 음성으로 생성
- 참조 음성을 기반으로 voice clone 수행
- 한 번 만든 음성 프롬프트를 저장한 뒤 여러 콘텐츠에 재사용

기존 [app.py](/f:/abc/qwen-tts/app.py) 는 ASR용 보조 프로그램이고, 현재 주력 기능은 TTS 앱입니다.

## 핵심 프로그램

### 메인 앱

- [tts_app.py](/f:/abc/qwen-tts/tts_app.py): TTS 웹 UI
- [tts.py](/f:/abc/qwen-tts/tts.py): 모델 로딩, 음성 생성, voice prompt 저장/재사용
- [TTS_GUIDE.md](/f:/abc/qwen-tts/TTS_GUIDE.md): 상세 사용 설명서

### 보조 앱

- [app.py](/f:/abc/qwen-tts/app.py): 음성 인식 ASR 웹 UI
- [hotkey_app.py](/f:/abc/qwen-tts/hotkey_app.py): 단축키 기반 ASR
- [subtitle_app.py](/f:/abc/qwen-tts/subtitle_app.py): 자막/번역용 ASR

## TTS 기능

현재 TTS 앱에서 지원하는 기능:
- 한국어 중심 텍스트 음성 변환
- 참조 음성 + 참조 텍스트 기반 정밀 voice clone
- `x-vector only` 기반 간편 voice clone
- voice prompt 저장
- 저장한 voice prompt 재사용
- 생성 음성 `.wav` 저장

## 실행 방법

### 1. 의존성 설치

```powershell
pip install -r requirements-tts.txt
```

권장:
- TTS는 별도 가상환경 사용
- CUDA 지원 PyTorch 사용

### 2. 앱 실행

```powershell
python tts_app.py
```

기본 포트:
- `7862`

이미 사용 중이면:
- `7863`, `7864` 등 다음 빈 포트를 자동 사용

## 가장 빠른 사용 순서

1. `python tts_app.py` 실행
2. 브라우저에서 TTS 페이지 열기
3. `Qwen3-TTS-1.7B Base (음성 복제)` 선택
4. `모델 로드` 클릭
5. `출력 언어`를 `한국어`로 설정
6. `합성할 원고` 입력
7. `참조 음성` 업로드 또는 녹음
8. 가능하면 `참조 음성 텍스트` 입력
9. `음성 생성` 클릭

## 음성을 저장하고 재사용하는 핵심 흐름

이 프로젝트에서 가장 중요한 기능은 `voice prompt` 저장과 재사용입니다.

흐름:
1. 참조 음성 준비
2. 필요하면 참조 텍스트 준비
3. `음성 프롬프트 저장/재사용` 탭에서 `음성 프롬프트 저장` 클릭
4. `.pt` 파일 생성
5. 이후 새 원고 입력
6. 저장한 `.pt` 파일 선택
7. `저장 프롬프트로 생성` 클릭

의미:
- 참조 음성을 매번 다시 업로드하지 않아도 됨
- 같은 목소리 계열로 여러 콘텐츠를 반복 제작 가능
- 화자별 음성 설정을 자산처럼 관리 가능

자세한 설명:
- [TTS_GUIDE.md](/f:/abc/qwen-tts/TTS_GUIDE.md)

특히 이 부분을 보면 됩니다.
- [TTS_GUIDE.md](/f:/abc/qwen-tts/TTS_GUIDE.md): `5. 음성을 저장하고 재사용하는 파이프라인`
- [TTS_GUIDE.md](/f:/abc/qwen-tts/TTS_GUIDE.md): `6. x-vector only 모드와 일반 모드 차이`
- [TTS_GUIDE.md](/f:/abc/qwen-tts/TTS_GUIDE.md): `10. 콘텐츠 제작용 추천 운영 방식`

## 저장 위치

### 모델 파일

- `models/`

설명:
- Hugging Face에서 내려받은 모델 저장 위치
- Git 업로드 대상 제외

### 생성 음성 / voice prompt

- `outputs/`

예시:
- `tts_clone_*.wav`
- `tts_prompt_*.wav`
- `voice_prompt_*.pt`

설명:
- 생성 음성 파일과 재사용 프롬프트 파일 저장 위치
- Git 업로드 대상 제외

## 주의 사항

- 본인 음성이나 사용 권한이 있는 음성만 사용하세요.
- `qwen-asr` 와 `qwen-tts` 는 의존성 충돌 가능성이 있으므로 분리 환경이 안전합니다.
- Windows에서 `SoX` 실행파일이 없으면 경고가 뜰 수 있습니다.
- 첫 모델 로드 시 다운로드 시간이 걸릴 수 있습니다.

## 파일 구조

```text
qwen-tts/
├── tts_app.py            # 메인 TTS 웹 UI
├── tts.py                # TTS 모델 래퍼
├── TTS_GUIDE.md          # TTS 상세 설명서
├── requirements-tts.txt  # TTS 의존성
├── app.py                # ASR 웹 UI
├── asr.py                # ASR 모델 래퍼
├── hotkey_app.py         # 단축키 ASR
├── subtitle_app.py       # 자막/번역 ASR
├── models/               # 모델 저장 폴더
├── outputs/              # 생성 음성/프롬프트 저장 폴더
└── README.md             # 프로젝트 요약 설명서
```

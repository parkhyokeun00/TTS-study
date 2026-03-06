"""
Qwen3-ASR 음성-텍스트 변환기 - 단축키 모드
F4+F5 단축키로 녹음하고 자동으로 텍스트 변환
"""

import sys
from asr import load_model, get_model_choices, asr_model
from hotkey_recorder import run_hotkey_listener


def main():
    print("=" * 60)
    print("🎤 Qwen3-ASR 단축키 녹음 모드")
    print("=" * 60)
    print()
    
    # 모델 선택
    choices = get_model_choices()
    print("📋 사용 가능한 모델:")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")
    print()
    
    try:
        selection = input("모델 번호를 선택하세요 (기본: 1): ").strip()
        if not selection:
            selection = "1"
        model_idx = int(selection) - 1
        model_choice = choices[model_idx]
    except (ValueError, IndexError):
        model_choice = choices[0]
    
    # 언어 선택
    print("\n🌐 인식 언어:")
    print("  1. 한국어 (기본)")
    print("  2. 일본어")
    print()
    
    lang_input = input("언어 번호를 선택하세요 (기본: 1): ").strip()
    language = "japanese" if lang_input == "2" else "korean"
    
    # 모델 로드
    print(f"\n⏳ {model_choice} 로딩 중...")
    result = load_model(model_choice)
    print(result)
    
    if "❌" in result:
        print("모델 로드에 실패했습니다.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ 준비 완료!")
    print("🎯 단축키:")
    print("   `+1 : 녹음 시작/종료")
    print("   `+2 : 마지막 텍스트 클립보드 복사")
    print("🛑 종료하려면 Ctrl+C를 누르세요.")
    print("=" * 60)
    print()
    
    # 단축키 리스너 시작
    run_hotkey_listener(
        asr_transcribe_func=asr_model.transcribe,
        language=language
    )


if __name__ == "__main__":
    main()

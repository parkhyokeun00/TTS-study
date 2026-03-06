from deep_translator import GoogleTranslator

class Translator:
    def __init__(self, source_lang='auto', target_lang='ko'):
        """
        초기화
        :param source_lang: 원본 언어 (기본값: auto)
        :param target_lang: 대상 언어 (기본값: ko)
        """
        self.translator = GoogleTranslator(source=source_lang, target=target_lang)

    def translate_to_korean(self, text: str) -> str:
        """
        주어진 텍스트를 한국어로 번역합니다.
        :param text: 원본 텍스트
        :return: 번역된 텍스트
        """
        if not text or not text.strip():
            return ""
        try:
            return self.translator.translate(text)
        except Exception as e:
            print(f"Translation Error: {str(e)}")
            return text

# 전역 번역기 인스턴스
korean_translator = Translator()

def translate(text: str) -> str:
    return korean_translator.translate_to_korean(text)

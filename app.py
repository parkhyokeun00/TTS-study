"""
Qwen3-ASR 음성-텍스트 변환기 GUI
Gradio 기반 웹 인터페이스
"""

import os
import socket
import gradio as gr
from asr import get_model_choices, load_model, transcribe_file, transcribe_mic


# CSS 스타일
CUSTOM_CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}

.title-text {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5em;
}

.subtitle-text {
    text-align: center;
    color: #666;
    margin-bottom: 1.5em;
}

.status-box {
    padding: 1em;
    border-radius: 8px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.output-text textarea {
    font-size: 1.2em !important;
    line-height: 1.6 !important;
}
"""


def create_app():
    """Gradio 앱 생성"""
    
    with gr.Blocks(title="Qwen3 음성-텍스트 변환기") as app:
        
        # 헤더
        gr.HTML('<div class="title-text">🎙️ Qwen3 음성-텍스트 변환기</div>')
        gr.HTML('<div class="subtitle-text">Qwen3-ASR 모델로 음성을 텍스트로 변환합니다</div>')
        
        # 모델 설정 영역
        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=get_model_choices(),
                    value=get_model_choices()[0],
                    label="🤖 모델 선택",
                    info="1.7B는 정확도가 높고, 0.6B는 빠릅니다"
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("📥 모델 로드", variant="primary", size="lg")
        
        # 상태 표시
        status_text = gr.Textbox(
            label="📊 상태",
            value="⏳ 모델을 로드해주세요...",
            interactive=False,
            elem_classes=["status-box"]
        )
        
        gr.Markdown("---")
        
        # 언어 선택
        language_radio = gr.Radio(
            choices=[("🇰🇷 한국어", "korean"), ("🇯🇵 일본어", "japanese")],
            value="korean",
            label="🌐 인식 언어",
            interactive=True
        )
        
        # 입력 탭
        with gr.Tabs():
            
            # 마이크 입력 탭
            with gr.TabItem("🎤 마이크 녹음"):
                mic_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="마이크로 녹음하세요"
                )
                mic_btn = gr.Button("✨ 변환하기", variant="primary")
                mic_output = gr.Textbox(
                    label="📝 변환 결과",
                    lines=5,
                    placeholder="변환된 텍스트가 여기에 표시됩니다...",
                    elem_classes=["output-text"],

                )
            
            # 파일 업로드 탭
            with gr.TabItem("📁 파일 업로드"):
                file_input = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="오디오 파일을 업로드하세요 (mp3, wav, m4a 등)"
                )
                file_btn = gr.Button("✨ 변환하기", variant="primary")
                file_output = gr.Textbox(
                    label="📝 변환 결과",
                    lines=5,
                    placeholder="변환된 텍스트가 여기에 표시됩니다...",
                    elem_classes=["output-text"],

                )
        
        # 이벤트 연결
        load_btn.click(
            fn=load_model,
            inputs=[model_dropdown],
            outputs=[status_text]
        )
        
        mic_btn.click(
            fn=transcribe_mic,
            inputs=[mic_input, language_radio],
            outputs=[mic_output]
        )
        
        file_btn.click(
            fn=transcribe_file,
            inputs=[file_input, language_radio],
            outputs=[file_output]
        )
        
        # 푸터
        gr.Markdown("---")
        gr.Markdown(
            "💡 **팁**: GPU가 없으면 CPU로 실행됩니다 (느릴 수 있음). "
            "CUDA 설치 시 훨씬 빠른 변환이 가능합니다."
        )
    
    return app


def resolve_server_port():
    """Return the requested port, or the next free port if it's busy."""
    port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=resolve_server_port(),
        share=False,
        inbrowser=True,
        css=CUSTOM_CSS
    )

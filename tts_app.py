"""
Qwen3-TTS Gradio app for Korean-first local voice cloning.
"""

import gradio as gr

from tts import get_model_choices, resolve_server_port, tts_model


CUSTOM_CSS = """
.gradio-container {
    max-width: 980px !important;
    margin: auto !important;
}

.title-text {
    text-align: center;
    font-size: 2.4em;
    font-weight: bold;
    color: #153243;
    margin-bottom: 0.4em;
}

.subtitle-text {
    text-align: center;
    color: #5c6b73;
    margin-bottom: 1.4em;
}

.status-box {
    padding: 1em;
    border-radius: 12px;
    background: linear-gradient(135deg, #f3f7f0 0%, #d8e2dc 100%);
}
"""


def create_app():
    with gr.Blocks(title="Qwen3-TTS 음성 복제 스튜디오") as app:
        gr.HTML('<div class="title-text">Qwen3-TTS 음성 복제 스튜디오</div>')
        gr.HTML('<div class="subtitle-text">한국어 원고를 로컬에서 음성으로 생성하고, 참조 음성으로 목소리를 복제합니다</div>')

        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=get_model_choices(),
                    value=get_model_choices()[0],
                    label="모델 선택",
                    info="음성 복제는 Base 모델을 우선 권장합니다.",
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("모델 로드", variant="primary", size="lg")

        status_text = gr.Textbox(
            label="상태",
            value="⏳ TTS 모델을 로드해주세요...",
            interactive=False,
            elem_classes=["status-box"],
            lines=4,
        )

        language_dropdown = gr.Dropdown(
            choices=[("한국어", "korean"), ("일본어", "japanese"), ("영어", "english"), ("중국어", "chinese")],
            value="korean",
            label="출력 언어",
        )

        with gr.Tabs():
            with gr.Tab("실시간 음성 복제"):
                text_input = gr.Textbox(
                    label="합성할 원고",
                    lines=6,
                    placeholder="예: 안녕하세요. 오늘 영상에서는 로컬에서 한국어 음성 복제를 하는 방법을 소개합니다.",
                )
                reference_audio = gr.Audio(
                    label="참조 음성",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                reference_text = gr.Textbox(
                    label="참조 음성 텍스트",
                    lines=3,
                    placeholder="정밀 복제 모드에서는 이 칸을 채우는 것이 좋습니다.",
                )
                x_vector_only = gr.Checkbox(
                    label="x-vector only 모드 사용",
                    value=False,
                    info="참조 텍스트 없이 화자 특성만 복제합니다. 정확도는 낮아질 수 있습니다.",
                )
                clone_btn = gr.Button("음성 생성", variant="primary")
                clone_status = gr.Textbox(label="생성 결과", interactive=False, lines=3)
                clone_audio = gr.Audio(label="생성된 음성", type="filepath")

            with gr.Tab("음성 프롬프트 저장/재사용"):
                prompt_audio = gr.Audio(
                    label="프롬프트용 참조 음성",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                prompt_text = gr.Textbox(
                    label="프롬프트용 참조 텍스트",
                    lines=3,
                    placeholder="x-vector only를 끄면 이 텍스트가 필요합니다.",
                )
                prompt_xvec = gr.Checkbox(
                    label="x-vector only 모드 사용",
                    value=False,
                )
                save_prompt_btn = gr.Button("음성 프롬프트 저장")
                prompt_status = gr.Textbox(label="프롬프트 상태", interactive=False, lines=3)
                prompt_file = gr.File(label="저장된 voice prompt", type="filepath")

                prompt_target_text = gr.Textbox(
                    label="재사용할 원고",
                    lines=5,
                    placeholder="저장한 voice prompt로 읽을 새 문장을 입력하세요.",
                )
                prompt_generate_btn = gr.Button("저장 프롬프트로 생성", variant="primary")
                prompt_gen_status = gr.Textbox(label="재사용 결과", interactive=False, lines=3)
                prompt_audio_out = gr.Audio(label="생성된 음성", type="filepath")

        gr.Markdown(
            "주의: 본인 음성이나 사용 권한이 있는 음성만 사용하세요. "
            "ASR와 TTS는 의존성 충돌 가능성이 있어 별도 가상환경 사용을 권장합니다."
        )

        load_btn.click(
            fn=tts_model.load_model,
            inputs=[model_dropdown],
            outputs=[status_text],
        )

        clone_btn.click(
            fn=tts_model.generate_voice_clone,
            inputs=[text_input, language_dropdown, reference_audio, reference_text, x_vector_only],
            outputs=[clone_audio, clone_status],
        )

        save_prompt_btn.click(
            fn=tts_model.save_voice_prompt,
            inputs=[prompt_audio, prompt_text, prompt_xvec],
            outputs=[prompt_file, prompt_status],
        )

        prompt_generate_btn.click(
            fn=tts_model.generate_from_prompt_file,
            inputs=[prompt_file, prompt_target_text, language_dropdown],
            outputs=[prompt_audio_out, prompt_gen_status],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=resolve_server_port(7862),
        share=False,
        inbrowser=True,
        css=CUSTOM_CSS,
    )

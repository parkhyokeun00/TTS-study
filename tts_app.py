"""
Qwen3-TTS Gradio app for Korean-first local voice cloning.
"""

import gradio as gr

from storage import configure_runtime_storage
from tts import get_model_choices, resolve_server_name, resolve_server_port, tts_model


configure_runtime_storage()


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


def save_postprocess_preset_ui(preset_name, speed, pitch_semitones, ending_style, ending_length_ms):
    preset_names, status = tts_model.save_postprocess_preset(
        preset_name=preset_name,
        speed=speed,
        pitch_semitones=pitch_semitones,
        ending_style=ending_style,
        ending_length_ms=ending_length_ms,
    )
    selected_name = preset_names[-1] if preset_names else None
    return gr.Dropdown(choices=preset_names, value=selected_name), status


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
                clone_audio = gr.Audio(label="생성된 원본 음성", type="filepath")
                with gr.Accordion("원본 생성 후 후처리", open=True):
                    gr.Markdown("원본 음성을 먼저 듣고 아래 옵션을 조절한 뒤, `후처리 미리듣기`를 누르면 적용된 결과를 바로 듣고 다운로드할 수 있습니다.")
                    with gr.Row():
                        clone_preset_name = gr.Textbox(
                            label="프리셋 이름",
                            lines=1,
                            placeholder="예: 유튜브 나레이션",
                            scale=2,
                        )
                        clone_preset_dropdown = gr.Dropdown(
                            choices=tts_model.get_postprocess_preset_names(),
                            value="기본",
                            label="저장된 프리셋",
                            scale=2,
                        )
                        clone_load_preset_btn = gr.Button("프리셋 불러오기", scale=1)
                        clone_save_preset_btn = gr.Button("현재값 저장", scale=1)
                    clone_preset_status = gr.Textbox(label="프리셋 상태", interactive=False, lines=2)
                    with gr.Row():
                        clone_speed_slider = gr.Slider(
                            minimum=0.7,
                            maximum=1.4,
                            value=1.0,
                            step=0.05,
                            label="속도",
                            info="1.0이 기본값입니다. 값이 클수록 더 빠르게 재생됩니다.",
                        )
                        clone_pitch_slider = gr.Slider(
                            minimum=-4.0,
                            maximum=4.0,
                            value=0.0,
                            step=0.5,
                            label="높낮이",
                            info="반음 단위입니다. 양수는 높게, 음수는 낮게 조정합니다.",
                        )
                    with gr.Row():
                        clone_ending_style = gr.Dropdown(
                            choices=[
                                ("기본", "default"),
                                ("부드럽게 마침", "soften"),
                                ("빠르게 감쇠", "fade"),
                                ("여운 추가", "hold"),
                                ("자연스럽게 마침", "natural"),
                            ],
                            value="default",
                            label="끝음 처리",
                        )
                        clone_ending_length = gr.Slider(
                            minimum=80,
                            maximum=1200,
                            value=180,
                            step=20,
                            label="끝음 길이(ms)",
                            info="끝부분에 적용할 길이 또는 여운 길이입니다.",
                        )
                    clone_postprocess_btn = gr.Button("후처리 미리듣기", variant="secondary")
                    clone_postprocess_status = gr.Textbox(label="후처리 결과", interactive=False, lines=2)
                    clone_processed_audio = gr.Audio(label="후처리 적용 음성", type="filepath")

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
                prompt_audio_out = gr.Audio(label="생성된 원본 음성", type="filepath")
                with gr.Accordion("원본 생성 후 후처리", open=True):
                    gr.Markdown("저장된 프롬프트로 만든 원본 음성을 확인한 뒤, 후처리 결과를 별도로 만들어 듣고 다운로드할 수 있습니다.")
                    with gr.Row():
                        prompt_preset_name = gr.Textbox(
                            label="프리셋 이름",
                            lines=1,
                            placeholder="예: 차분한 끝맺음",
                            scale=2,
                        )
                        prompt_preset_dropdown = gr.Dropdown(
                            choices=tts_model.get_postprocess_preset_names(),
                            value="기본",
                            label="저장된 프리셋",
                            scale=2,
                        )
                        prompt_load_preset_btn = gr.Button("프리셋 불러오기", scale=1)
                        prompt_save_preset_btn = gr.Button("현재값 저장", scale=1)
                    prompt_preset_status = gr.Textbox(label="프리셋 상태", interactive=False, lines=2)
                    with gr.Row():
                        prompt_speed_slider = gr.Slider(
                            minimum=0.7,
                            maximum=1.4,
                            value=1.0,
                            step=0.05,
                            label="속도",
                            info="1.0이 기본값입니다. 값이 클수록 더 빠르게 재생됩니다.",
                        )
                        prompt_pitch_slider = gr.Slider(
                            minimum=-4.0,
                            maximum=4.0,
                            value=0.0,
                            step=0.5,
                            label="높낮이",
                            info="반음 단위입니다. 양수는 높게, 음수는 낮게 조정합니다.",
                        )
                    with gr.Row():
                        prompt_ending_style = gr.Dropdown(
                            choices=[
                                ("기본", "default"),
                                ("부드럽게 마침", "soften"),
                                ("빠르게 감쇠", "fade"),
                                ("여운 추가", "hold"),
                                ("자연스럽게 마침", "natural"),
                            ],
                            value="default",
                            label="끝음 처리",
                        )
                        prompt_ending_length = gr.Slider(
                            minimum=80,
                            maximum=1200,
                            value=180,
                            step=20,
                            label="끝음 길이(ms)",
                            info="끝부분에 적용할 길이 또는 여운 길이입니다.",
                        )
                    prompt_postprocess_btn = gr.Button("후처리 미리듣기", variant="secondary")
                    prompt_postprocess_status = gr.Textbox(label="후처리 결과", interactive=False, lines=2)
                    prompt_processed_audio = gr.Audio(label="후처리 적용 음성", type="filepath")

        gr.Markdown(
            "주의: 본인 음성이나 사용 권한이 있는 음성만 사용하세요. "
            "ASR와 TTS는 의존성 충돌 가능성이 있어 별도 가상환경 사용을 권장합니다. "
            "마이크 녹음이 안 되면 반드시 localhost 또는 127.0.0.1 주소로 접속하세요."
        )

        load_btn.click(
            fn=tts_model.load_model,
            inputs=[model_dropdown],
            outputs=[status_text],
        )

        clone_save_preset_btn.click(
            fn=save_postprocess_preset_ui,
            inputs=[clone_preset_name, clone_speed_slider, clone_pitch_slider, clone_ending_style, clone_ending_length],
            outputs=[clone_preset_dropdown, clone_preset_status],
        )

        clone_load_preset_btn.click(
            fn=tts_model.load_postprocess_preset,
            inputs=[clone_preset_dropdown],
            outputs=[clone_speed_slider, clone_pitch_slider, clone_ending_style, clone_ending_length, clone_preset_status],
        )

        clone_btn.click(
            fn=tts_model.generate_voice_clone,
            inputs=[
                text_input,
                language_dropdown,
                reference_audio,
                reference_text,
                x_vector_only,
            ],
            outputs=[clone_audio, clone_status],
        )

        clone_postprocess_btn.click(
            fn=tts_model.apply_postprocess_to_file,
            inputs=[clone_audio, clone_speed_slider, clone_pitch_slider, clone_ending_style, clone_ending_length],
            outputs=[clone_processed_audio, clone_postprocess_status],
        )

        save_prompt_btn.click(
            fn=tts_model.save_voice_prompt,
            inputs=[prompt_audio, prompt_text, prompt_xvec],
            outputs=[prompt_file, prompt_status],
        )

        prompt_save_preset_btn.click(
            fn=save_postprocess_preset_ui,
            inputs=[prompt_preset_name, prompt_speed_slider, prompt_pitch_slider, prompt_ending_style, prompt_ending_length],
            outputs=[prompt_preset_dropdown, prompt_preset_status],
        )

        prompt_load_preset_btn.click(
            fn=tts_model.load_postprocess_preset,
            inputs=[prompt_preset_dropdown],
            outputs=[prompt_speed_slider, prompt_pitch_slider, prompt_ending_style, prompt_ending_length, prompt_preset_status],
        )

        prompt_generate_btn.click(
            fn=tts_model.generate_from_prompt_file,
            inputs=[
                prompt_file,
                prompt_target_text,
                language_dropdown,
            ],
            outputs=[prompt_audio_out, prompt_gen_status],
        )

        prompt_postprocess_btn.click(
            fn=tts_model.apply_postprocess_to_file,
            inputs=[prompt_audio_out, prompt_speed_slider, prompt_pitch_slider, prompt_ending_style, prompt_ending_length],
            outputs=[prompt_processed_audio, prompt_postprocess_status],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name=resolve_server_name("127.0.0.1"),
        server_port=resolve_server_port(7862),
        share=False,
        inbrowser=True,
        css=CUSTOM_CSS,
    )

"""
Qwen3-TTS Gradio app for Korean-first local voice cloning.
"""

import gradio as gr

from storage import configure_runtime_storage
from tts import get_model_choices, resolve_server_name, resolve_server_port, tts_model
from voice_conversion import voice_converter


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


def convert_voice_ui(
    input_audio,
    model_file,
    index_file,
    hubert_file,
    rmvpe_file,
    speaker_id,
    pitch_shift,
    f0_method,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):
    output_path, status = voice_converter.convert_voice(
        input_audio=input_audio,
        model_file=model_file,
        index_file=index_file,
        hubert_file=hubert_file,
        rmvpe_file=rmvpe_file,
        speaker_id=speaker_id,
        pitch_shift=pitch_shift,
        f0_method=f0_method,
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
    )
    return output_path, status, voice_converter.get_runtime_status()


def extract_multi_speaker_rows_ui(script_text):
    rows, status = tts_model.build_multi_speaker_rows(script_text)
    return rows, status


def create_app():
    with gr.Blocks(title="Qwen3-TTS 음성 복제 스튜디오") as app:
        gr.HTML('<div class="title-text">Qwen3-TTS 음성 복제 스튜디오</div>')
        gr.HTML('<div class="subtitle-text">한국어 TTS, voice clone, 그리고 실험적 RVC 음성 변조를 로컬에서 다룹니다</div>')

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

            with gr.Tab("다화자 대본 생성"):
                gr.Markdown(
                    "긴 대본을 `화자: 대사` 형식으로 입력한 뒤, 화자를 추출해서 각 화자에 맞는 "
                    "`voice prompt(.pt)` 경로를 지정하면 줄별 음성과 최종 합본을 로컬에 저장합니다."
                )
                multi_script_input = gr.Textbox(
                    label="대본 입력",
                    lines=12,
                    placeholder=(
                        "예:\n"
                        "나레이터: 오늘은 새로운 기능을 소개합니다.\n"
                        "여자1: 첫 번째 장면을 시작하겠습니다.\n"
                        "남자1: 다음 내용을 이어서 설명하겠습니다."
                    ),
                )
                with gr.Row():
                    multi_extract_btn = gr.Button("화자 추출", variant="secondary")
                    multi_generate_btn = gr.Button("다화자 음성 생성", variant="primary")
                multi_status = gr.Textbox(
                    label="상태 / 안내",
                    interactive=False,
                    lines=5,
                    value="대본을 입력하고 `화자 추출`을 누르세요.",
                )
                with gr.Row():
                    multi_job_name = gr.Textbox(
                        label="작업 이름",
                        value="multi_speaker_demo",
                        placeholder="예: story_episode_01",
                    )
                    multi_silence_ms = gr.Slider(
                        minimum=0,
                        maximum=1500,
                        value=120,
                        step=20,
                        label="줄 사이 무음(ms)",
                    )
                with gr.Row():
                    multi_save_lines = gr.Checkbox(label="줄별 파일 목록에 포함", value=True)
                    multi_make_mix = gr.Checkbox(label="최종 합본 생성", value=True)

                multi_speaker_table = gr.Dataframe(
                    headers=["speaker", "prompt_path", "speed", "pitch", "ending_style", "ending_length_ms"],
                    datatype=["str", "str", "number", "number", "str", "number"],
                    row_count=(1, "dynamic"),
                    column_count=(6, "fixed"),
                    interactive=True,
                    wrap=True,
                    label="화자별 설정 표",
                    value=[["나레이터", "", 1.0, 0.0, "default", 180]],
                )
                gr.Markdown(
                    "`prompt_path`에는 로컬의 `voice prompt(.pt)` 경로를 직접 넣어주세요. "
                    "`ending_style`은 `default`, `soften`, `fade`, `hold`, `natural` 중 하나를 사용합니다."
                )
                multi_result_table = gr.Dataframe(
                    headers=["line_id", "speaker", "text", "status", "audio_path", "chunks"],
                    datatype=["str", "str", "str", "str", "str", "number"],
                    row_count=(0, "dynamic"),
                    column_count=(6, "fixed"),
                    interactive=False,
                    wrap=True,
                    label="줄별 생성 결과",
                )
                multi_final_audio = gr.Audio(label="최종 합본", type="filepath")
                multi_output_files = gr.Files(label="생성된 파일")

            with gr.Tab("음성 대 음성 변조 (RVC 실험)"):
                gr.Markdown(
                    "입력 음성의 내용은 유지한 채, RVC 모델의 목소리로 바꾸는 실험 탭입니다. "
                    "Windows에서는 Python 3.10 가상환경을 권장합니다. "
                    "설치 절차는 README의 RVC 섹션을 따르고, "
                    "`hubert_base.pt`, `rmvpe.pt`, 대상 `*.pth` 모델을 준비해야 합니다."
                )
                vc_runtime_status = gr.Textbox(
                    label="RVC 런타임 상태",
                    value=voice_converter.get_runtime_status(),
                    interactive=False,
                    lines=6,
                )
                vc_refresh_btn = gr.Button("RVC 상태 새로고침")
                vc_input_audio = gr.Audio(
                    label="변조할 입력 음성",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                with gr.Row():
                    vc_model_file = gr.File(
                        label="RVC 모델 (.pth)",
                        file_types=[".pth"],
                        type="filepath",
                    )
                    vc_index_file = gr.File(
                        label="Feature Index (.index, 선택)",
                        file_types=[".index"],
                        type="filepath",
                    )
                with gr.Row():
                    vc_hubert_file = gr.File(
                        label="HuBERT base (.pt, 선택)",
                        file_types=[".pt"],
                        type="filepath",
                    )
                    vc_rmvpe_file = gr.File(
                        label="RMVPE (.pt, 선택)",
                        file_types=[".pt"],
                        type="filepath",
                    )
                with gr.Row():
                    vc_speaker_id = gr.Number(label="화자 ID", value=0, precision=0)
                    vc_pitch_shift = gr.Slider(
                        minimum=-24,
                        maximum=24,
                        value=0,
                        step=1,
                        label="키 변경 (반음)",
                    )
                    vc_f0_method = gr.Dropdown(
                        choices=voice_converter.F0_METHOD_CHOICES,
                        value="rmvpe",
                        label="F0 추출 방식",
                    )
                with gr.Row():
                    vc_index_rate = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.75,
                        step=0.05,
                        label="Index 비율",
                    )
                    vc_filter_radius = gr.Slider(
                        minimum=0,
                        maximum=7,
                        value=3,
                        step=1,
                        label="Filter Radius",
                    )
                with gr.Row():
                    vc_resample_sr = gr.Dropdown(
                        choices=[("모델 기본값", 0), ("32000 Hz", 32000), ("40000 Hz", 40000), ("48000 Hz", 48000)],
                        value=0,
                        label="출력 샘플레이트",
                    )
                    vc_rms_mix_rate = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="RMS Mix Rate",
                    )
                    vc_protect = gr.Slider(
                        minimum=0.0,
                        maximum=0.5,
                        value=0.33,
                        step=0.01,
                        label="Protect",
                    )
                vc_convert_btn = gr.Button("음성 변조 실행", variant="primary")
                vc_status = gr.Textbox(label="변조 결과", interactive=False, lines=5)
                vc_output_audio = gr.Audio(label="변조된 음성", type="filepath")

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

        multi_extract_btn.click(
            fn=extract_multi_speaker_rows_ui,
            inputs=[multi_script_input],
            outputs=[multi_speaker_table, multi_status],
        )

        multi_generate_btn.click(
            fn=tts_model.generate_multi_speaker_script,
            inputs=[
                multi_script_input,
                multi_speaker_table,
                language_dropdown,
                multi_job_name,
                multi_save_lines,
                multi_make_mix,
                multi_silence_ms,
            ],
            outputs=[multi_final_audio, multi_output_files, multi_status, multi_result_table],
        )

        vc_refresh_btn.click(
            fn=voice_converter.get_runtime_status,
            outputs=[vc_runtime_status],
        )

        vc_convert_btn.click(
            fn=convert_voice_ui,
            inputs=[
                vc_input_audio,
                vc_model_file,
                vc_index_file,
                vc_hubert_file,
                vc_rmvpe_file,
                vc_speaker_id,
                vc_pitch_shift,
                vc_f0_method,
                vc_index_rate,
                vc_filter_radius,
                vc_resample_sr,
                vc_rms_mix_rate,
                vc_protect,
            ],
            outputs=[vc_output_audio, vc_status, vc_runtime_status],
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

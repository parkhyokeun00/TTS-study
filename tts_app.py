"""
Qwen3-TTS Gradio app for Korean-first local voice cloning.
"""

import os

import gradio as gr

from storage import configure_runtime_storage
from tts import get_model_choices, resolve_server_name, resolve_server_port, tts_model
from voice_conversion import voice_converter


configure_runtime_storage()


CUSTOM_CSS = """
.gradio-container {
    max-width: 96vw !important;
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

.workspace-card {
    border: 1px solid #d9e3d6;
    border-radius: 16px;
    padding: 16px;
    background: #fbfdf9;
}

.workspace-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #153243;
    margin-bottom: 0.6rem;
}

.speaker-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin-bottom: 12px;
}

.speaker-card {
    border: 1px solid #d6ddd2;
    border-radius: 14px;
    padding: 12px;
    background: #ffffff;
}

.speaker-card.active {
    border-color: #153243;
    box-shadow: 0 0 0 1px #153243 inset;
    background: #f4f8f9;
}

.speaker-card-top {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
}

.speaker-index {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 999px;
    background: #153243;
    color: #fff;
    font-size: 12px;
    font-weight: 700;
}

.speaker-name {
    font-weight: 700;
    color: #153243;
}

.speaker-badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 8px;
}

.speaker-badge.assigned {
    background: #d8efe0;
    color: #1d6b41;
}

.speaker-badge.empty {
    background: #f7dfdf;
    color: #9a2c2c;
}

.speaker-prompt {
    font-size: 12px;
    color: #55626b;
    line-height: 1.4;
    word-break: break-all;
}
"""

CARD_LIMIT = 10
ENDING_STYLE_CHOICES = [
    ("기본", "default"),
    ("부드럽게 마침", "soften"),
    ("빠르게 감쇠", "fade"),
    ("여운 추가", "hold"),
    ("자연스럽게 마침", "natural"),
]


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


def convert_voice_folder_ui(
    input_folder,
    output_subfolder_name,
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
    output_files, status, output_dir = voice_converter.convert_voice_folder(
        input_folder=input_folder,
        output_subfolder_name=output_subfolder_name,
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
    return output_files, status, output_dir, voice_converter.get_runtime_status()


def extract_multi_speaker_rows_ui(script_text):
    rows, status = tts_model.build_multi_speaker_rows(script_text)
    return rows, status


def clear_multi_line_preview_ui():
    return None, "", "", "줄을 선택하면 여기서 미리듣기할 수 있습니다."


def paragraph_cards_from_script_ui(script_text):
    cards = tts_model.build_paragraph_card_rows(script_text, max_blocks=CARD_LIMIT)
    prompt_choices = tts_model.get_voice_prompt_choices()
    outputs = [
        f"✅ 문단 카드 {len(cards)}개 준비 완료\n문단은 빈 줄 기준으로 나뉩니다.",
        gr.update(value=[]),
    ]

    for index in range(CARD_LIMIT):
        if index < len(cards):
            card = cards[index]
            outputs.extend(
                [
                    gr.update(label=f"문단 카드 {index + 1}", open=index < 2),
                    card["text"],
                    card["speaker"],
                    gr.update(choices=prompt_choices, value=None),
                    "",
                    float(card["speed"]),
                    float(card["pitch"]),
                    str(card["ending_style"]),
                    int(card["ending_length_ms"]),
                ]
            )
        else:
            outputs.extend(
                [
                    gr.update(label=f"문단 카드 {index + 1}", open=False),
                    "",
                    "",
                    gr.update(choices=prompt_choices, value=None),
                    "",
                    1.0,
                    0.0,
                    "default",
                    180,
                ]
            )
    return tuple(outputs)


def refresh_card_prompt_choices_ui():
    prompt_choices = tts_model.get_voice_prompt_choices()
    outputs = []
    for _ in range(CARD_LIMIT):
        outputs.append(gr.update(choices=prompt_choices))
    outputs.append(f"✅ voice prompt {len(prompt_choices)}개 검색 완료")
    return tuple(outputs)


def set_card_prompt_from_library_ui(selected_prompt_path):
    return selected_prompt_path or ""


def set_card_prompt_from_upload_ui(uploaded_file):
    return tts_model._resolve_file_path(uploaded_file) or ""


def generate_multi_speaker_from_cards_ui(*args):
    expected_card_fields = CARD_LIMIT * 8
    card_args = args[:expected_card_fields]
    language, job_name, save_line_files, merge_output, silence_ms = args[expected_card_fields:]

    paragraph_cards = []
    for index in range(CARD_LIMIT):
        offset = index * 8
        text = str(card_args[offset]).strip() if card_args[offset] is not None else ""
        speaker = str(card_args[offset + 1]).strip() if card_args[offset + 1] is not None else ""
        prompt_path = str(card_args[offset + 3]).strip() if card_args[offset + 3] is not None else ""
        if not text:
            continue
        paragraph_cards.append(
            {
                "card_index": index + 1,
                "text": text,
                "speaker": speaker,
                "prompt_path": prompt_path,
                "speed": float(card_args[offset + 4]),
                "pitch": float(card_args[offset + 5]),
                "ending_style": str(card_args[offset + 6]),
                "ending_length_ms": int(card_args[offset + 7]),
            }
        )

    return tts_model.generate_multi_speaker_paragraphs(
        paragraph_cards=paragraph_cards,
        language=language,
        job_name=job_name,
        save_line_files=save_line_files,
        merge_output=merge_output,
        silence_ms=silence_ms,
    )


def build_cards_summary_ui(*args):
    rows = []
    for index in range(CARD_LIMIT):
        offset = index * 8
        text = str(args[offset]).strip() if args[offset] is not None else ""
        speaker = str(args[offset + 1]).strip() if args[offset + 1] is not None else ""
        prompt_path = str(args[offset + 3]).strip() if args[offset + 3] is not None else ""
        if not text:
            continue
        rows.append(
            [
                speaker,
                prompt_path,
                float(args[offset + 4]),
                float(args[offset + 5]),
                str(args[offset + 6]),
                int(args[offset + 7]),
            ]
        )
    return rows


def _normalize_multi_rows(rows):
    if hasattr(rows, "fillna") and hasattr(rows, "values"):
        rows = rows.fillna("").values.tolist()
    elif isinstance(rows, tuple):
        rows = list(rows)
    return rows if isinstance(rows, list) else []

def build_speaker_cards_html(rows, selected_speaker=""):
    normalized = _normalize_multi_rows(rows)
    if not normalized:
        return "<div class='speaker-cards-empty'>화자 추출 후 카드가 표시됩니다.</div>"

    parts = ["<div class='speaker-cards'>"]
    for index, row in enumerate(normalized, start=1):
        speaker = str(row[0]).strip() if len(row) > 0 and row[0] is not None else f"speaker_{index}"
        prompt_path = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
        assigned = bool(prompt_path)
        card_class = "speaker-card active" if speaker == selected_speaker else "speaker-card"
        badge = "연결됨" if assigned else "미지정"
        badge_class = "assigned" if assigned else "empty"
        prompt_name = os.path.basename(prompt_path) if prompt_path else "voice prompt 없음"
        parts.append(
            f"<div class='{card_class}'>"
            f"<div class='speaker-card-top'><span class='speaker-index'>{index}</span><span class='speaker-name'>{speaker}</span></div>"
            f"<div class='speaker-badge {badge_class}'>{badge}</div>"
            f"<div class='speaker-prompt'>{prompt_name}</div>"
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _first_speaker_name(rows):
    normalized = _normalize_multi_rows(rows)
    for row in normalized:
        if row and len(row) > 0 and str(row[0]).strip():
            return str(row[0]).strip()
    return None


def extract_multi_speaker_workspace_ui(script_text):
    rows, status = tts_model.build_multi_speaker_rows(script_text)
    if status.startswith("❌"):
        return rows, status, build_speaker_cards_html([]), gr.update(choices=[], value=None), gr.update(choices=[], value=None), "", 1.0, 0.0, "default", 180, "화자 추출에 실패했습니다."

    speaker_selector = tts_model.build_speaker_selector(rows)
    selected_speaker = _first_speaker_name(rows)
    prompt_path, speed, pitch, ending_style, ending_length, editor_status = tts_model.get_speaker_editor_values(rows, selected_speaker)
    prompt_choices = tts_model.get_voice_prompt_choices()
    merged_status = f"{status}\n✅ voice prompt {len(prompt_choices)}개 검색 완료"
    return (
        rows,
        merged_status,
        build_speaker_cards_html(rows, selected_speaker),
        gr.update(choices=speaker_selector.choices, value=selected_speaker),
        gr.update(choices=prompt_choices, value=prompt_path or None),
        prompt_path,
        speed,
        pitch,
        ending_style,
        ending_length,
        editor_status,
    )


def refresh_prompt_library_ui(rows, selected_speaker):
    prompt_choices = tts_model.get_voice_prompt_choices()
    prompt_path, speed, pitch, ending_style, ending_length, status = tts_model.get_speaker_editor_values(rows, selected_speaker)
    return gr.update(choices=prompt_choices, value=prompt_path or None), f"✅ voice prompt {len(prompt_choices)}개 검색 완료"


def select_speaker_card_ui(rows, selected_speaker):
    prompt_choices = tts_model.get_voice_prompt_choices()
    prompt_path, speed, pitch, ending_style, ending_length, status = tts_model.get_speaker_editor_values(rows, selected_speaker)
    return (
        build_speaker_cards_html(rows, selected_speaker),
        gr.update(choices=prompt_choices, value=prompt_path or None),
        prompt_path,
        speed,
        pitch,
        ending_style,
        ending_length,
        status,
    )


def apply_selected_speaker_ui(rows, selected_speaker, selected_prompt_path, uploaded_prompt_file, speed, pitch, ending_style, ending_length_ms):
    updated_rows, prompt_path, status, next_speaker = tts_model.update_speaker_row(
        rows,
        selected_speaker,
        selected_prompt_path,
        "",
        uploaded_prompt_file,
        speed,
        pitch,
        ending_style,
        ending_length_ms,
    )
    speaker_selector = tts_model.build_speaker_selector(updated_rows)
    target_speaker = next_speaker or selected_speaker or _first_speaker_name(updated_rows)
    prompt_choices = tts_model.get_voice_prompt_choices()
    current_prompt_path, current_speed, current_pitch, current_ending_style, current_ending_length, _ = tts_model.get_speaker_editor_values(updated_rows, target_speaker)
    return (
        updated_rows,
        build_speaker_cards_html(updated_rows, target_speaker),
        gr.update(choices=speaker_selector.choices, value=target_speaker),
        gr.update(choices=prompt_choices, value=current_prompt_path or None),
        current_prompt_path,
        current_speed,
        current_pitch,
        current_ending_style,
        current_ending_length,
        status,
    )


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
                multi_job_dir_state = gr.State("")
                gr.Markdown(
                    "문단은 사용자가 직접 입력하고, 각 문단 카드마다 화자와 voice prompt를 직접 지정합니다. "
                    "빈 줄을 기준으로 문단 카드를 만들고, 최대 10개 카드까지 편집할 수 있습니다."
                )
                with gr.Row():
                    with gr.Column(scale=7):
                        gr.HTML('<div class="workspace-card"><div class="workspace-title">대본 작업 공간</div></div>')
                        multi_script_input = gr.Textbox(
                            label="대본 입력",
                            lines=18,
                            placeholder=(
                                "예:\n"
                                "원래 안 사려고 했던 물건도\n"
                                "마지막 1개 남음이라는 문구를 보는 순간\n"
                                "갑자기 마음이 급해질 때가 있습니다.\n\n"
                                "조금 전까지만 해도 괜찮았는데,\n"
                                "그 한 줄이 뜨는 순간\n"
                                "지금 안 사면 놓칠 것 같다는 감정이 밀려오죠."
                            ),
                        )
                        with gr.Row():
                            multi_extract_btn = gr.Button("문단 카드 만들기", variant="secondary")
                            multi_generate_btn = gr.Button("다화자 음성 생성", variant="primary")
                        multi_status = gr.Textbox(
                            label="상태 / 안내",
                            interactive=False,
                            lines=5,
                            value="대본을 입력한 뒤 `문단 카드 만들기`를 누르세요.",
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

                        multi_result_table = gr.Dataframe(
                            headers=["line_id", "speaker", "text", "status", "audio_path", "chunks", "selected_version", "version_count"],
                            datatype=["str", "str", "str", "str", "str", "number", "str", "number"],
                            row_count=(0, "dynamic"),
                            column_count=(8, "fixed"),
                            interactive=False,
                            wrap=True,
                            label="줄별 생성 결과",
                        )
                        multi_final_audio = gr.Audio(label="최종 합본", type="filepath")
                        multi_output_files = gr.Files(label="생성된 파일")

                    with gr.Column(scale=4):
                        gr.HTML('<div class="workspace-card"><div class="workspace-title">문단 카드 편집기</div></div>')
                        multi_refresh_prompts_btn = gr.Button("저장된 voice prompt 목록 새로고침")
                        multi_editor_status = gr.Textbox(
                            label="문단 카드 상태",
                            interactive=False,
                            lines=4,
                            value="문단 카드를 만든 뒤 각 카드에 화자와 voice prompt를 지정하세요.",
                        )
                        multi_card_components = []
                        for index in range(CARD_LIMIT):
                            with gr.Accordion(f"문단 카드 {index + 1}", open=index < 2) as card_box:
                                card_text = gr.Textbox(label="문단 텍스트", lines=5)
                                card_speaker = gr.Textbox(label="화자 이름", placeholder=f"예: 화자{index + 1}")
                                card_prompt_library = gr.Dropdown(
                                    choices=tts_model.get_voice_prompt_choices(),
                                    value=None,
                                    label="저장된 voice prompt",
                                    allow_custom_value=True,
                                )
                                card_prompt_upload = gr.File(
                                    label="이 카드에 .pt 드래그 앤 드롭",
                                    file_types=[".pt"],
                                    type="filepath",
                                )
                                card_prompt_path = gr.Textbox(
                                    label="현재 연결된 voice prompt 경로",
                                    lines=2,
                                )
                                with gr.Row():
                                    card_speed = gr.Slider(minimum=0.7, maximum=1.4, value=1.0, step=0.05, label="속도")
                                    card_pitch = gr.Slider(minimum=-4.0, maximum=4.0, value=0.0, step=0.5, label="피치")
                                with gr.Row():
                                    card_ending_style = gr.Dropdown(choices=ENDING_STYLE_CHOICES, value="default", label="끝음 처리")
                                    card_ending_length = gr.Slider(minimum=80, maximum=1200, value=180, step=20, label="끝음 길이(ms)")

                                card_prompt_library.change(
                                    fn=set_card_prompt_from_library_ui,
                                    inputs=[card_prompt_library],
                                    outputs=[card_prompt_path],
                                )
                                card_prompt_upload.change(
                                    fn=set_card_prompt_from_upload_ui,
                                    inputs=[card_prompt_upload],
                                    outputs=[card_prompt_path],
                                )

                                multi_card_components.append(
                                    {
                                        "accordion": card_box,
                                        "text": card_text,
                                        "speaker": card_speaker,
                                        "prompt_library": card_prompt_library,
                                        "prompt_path": card_prompt_path,
                                        "speed": card_speed,
                                        "pitch": card_pitch,
                                        "ending_style": card_ending_style,
                                        "ending_length": card_ending_length,
                                    }
                                )

                        multi_speaker_table = gr.Dataframe(
                            headers=["speaker", "prompt_path", "speed", "pitch", "ending_style", "ending_length_ms"],
                            datatype=["str", "str", "number", "number", "str", "number"],
                            row_count=(0, "dynamic"),
                            column_count=(6, "fixed"),
                            interactive=False,
                            wrap=True,
                            label="현재 카드 설정 요약",
                            value=[],
                        )

                with gr.Accordion("줄별 미리듣기 / 다시 생성", open=True):
                    with gr.Row():
                        multi_line_selector = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="선택 줄",
                            info="생성 후 확인하거나 다시 만들 줄을 선택합니다.",
                        )
                        multi_preview_btn = gr.Button("선택 줄 미리듣기", variant="secondary")
                        multi_regenerate_btn = gr.Button("선택 줄 다시 생성", variant="primary")
                    multi_line_status = gr.Textbox(
                        label="줄 작업 상태",
                        interactive=False,
                        lines=4,
                        value="줄을 선택하면 여기서 미리듣기할 수 있습니다.",
                    )
                    multi_line_audio = gr.Audio(label="선택 줄 오디오", type="filepath")
                    with gr.Row():
                        multi_line_speaker = gr.Textbox(label="선택 줄 화자", interactive=False)
                        multi_line_text = gr.Textbox(
                            label="선택 줄 대사",
                            interactive=True,
                            lines=3,
                            info="여기서 대사를 수정한 뒤 다시 생성할 수 있습니다.",
                        )
                    with gr.Row():
                        multi_regen_speed = gr.Slider(
                            minimum=0.7,
                            maximum=1.4,
                            value=1.0,
                            step=0.05,
                            label="재생성 속도",
                        )
                        multi_regen_pitch = gr.Slider(
                            minimum=-4.0,
                            maximum=4.0,
                            value=0.0,
                            step=0.5,
                            label="재생성 피치",
                        )
                    with gr.Row():
                        multi_regen_ending_style = gr.Dropdown(
                            choices=ENDING_STYLE_CHOICES,
                            value="default",
                            label="재생성 끝음 처리",
                        )
                        multi_regen_ending_length = gr.Slider(
                            minimum=80,
                            maximum=1200,
                            value=180,
                            step=20,
                            label="재생성 끝음 길이(ms)",
                        )

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
                with gr.Accordion("폴더 일괄 변조", open=False):
                    vc_input_folder = gr.Textbox(
                        label="변조할 음성 폴더 경로",
                        placeholder=r"예: F:\abc\qwen-tts\inputs\rvc_batch",
                        info="이 폴더 바로 아래의 오디오 파일들을 한 번에 변조합니다.",
                    )
                    vc_output_subfolder = gr.Textbox(
                        label="저장 하위 폴더 이름",
                        placeholder="비워두면 날짜_입력폴더명으로 자동 생성",
                        info=r"저장 위치: F:\abc\qwen-tts\outputs\voice_conversion\<여기에 입력한 이름>",
                    )
                    vc_batch_convert_btn = gr.Button("폴더 일괄 변조 실행", variant="primary")
                    vc_batch_status = gr.Textbox(label="일괄 변조 결과", interactive=False, lines=6)
                    vc_batch_output_dir = gr.Textbox(label="일괄 저장 폴더", interactive=False)
                    vc_batch_output_files = gr.File(label="일괄 변조 파일", file_count="multiple")

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

        card_extract_outputs = [multi_status, multi_speaker_table]
        for card in multi_card_components:
            card_extract_outputs.extend(
                [
                    card["accordion"],
                    card["text"],
                    card["speaker"],
                    card["prompt_library"],
                    card["prompt_path"],
                    card["speed"],
                    card["pitch"],
                    card["ending_style"],
                    card["ending_length"],
                ]
            )

        multi_extract_btn.click(
            fn=paragraph_cards_from_script_ui,
            inputs=[multi_script_input],
            outputs=card_extract_outputs,
        )

        multi_extract_btn.click(
            fn=clear_multi_line_preview_ui,
            outputs=[multi_line_audio, multi_line_speaker, multi_line_text, multi_line_status],
        )

        multi_extract_btn.click(
            fn=lambda: (1.0, 0.0, "default", 180),
            outputs=[multi_regen_speed, multi_regen_pitch, multi_regen_ending_style, multi_regen_ending_length],
        )

        multi_refresh_prompts_btn.click(
            fn=refresh_card_prompt_choices_ui,
            outputs=[*[card["prompt_library"] for card in multi_card_components], multi_editor_status],
        )

        summary_inputs = []
        for card in multi_card_components:
            summary_inputs.extend(
                [
                    card["text"],
                    card["speaker"],
                    card["prompt_library"],
                    card["prompt_path"],
                    card["speed"],
                    card["pitch"],
                    card["ending_style"],
                    card["ending_length"],
                ]
            )

        for card in multi_card_components:
            card["text"].change(fn=build_cards_summary_ui, inputs=summary_inputs, outputs=[multi_speaker_table])
            card["speaker"].change(fn=build_cards_summary_ui, inputs=summary_inputs, outputs=[multi_speaker_table])
            card["prompt_path"].change(fn=build_cards_summary_ui, inputs=summary_inputs, outputs=[multi_speaker_table])
            card["speed"].release(fn=build_cards_summary_ui, inputs=summary_inputs, outputs=[multi_speaker_table])
            card["pitch"].release(fn=build_cards_summary_ui, inputs=summary_inputs, outputs=[multi_speaker_table])
            card["ending_style"].change(fn=build_cards_summary_ui, inputs=summary_inputs, outputs=[multi_speaker_table])
            card["ending_length"].release(fn=build_cards_summary_ui, inputs=summary_inputs, outputs=[multi_speaker_table])

        multi_generate_btn.click(
            fn=generate_multi_speaker_from_cards_ui,
            inputs=[*summary_inputs, language_dropdown, multi_job_name, multi_save_lines, multi_make_mix, multi_silence_ms],
            outputs=[multi_final_audio, multi_output_files, multi_status, multi_result_table, multi_job_dir_state, multi_line_selector],
        )

        multi_generate_btn.click(
            fn=clear_multi_line_preview_ui,
            outputs=[multi_line_audio, multi_line_speaker, multi_line_text, multi_line_status],
        )

        multi_preview_btn.click(
            fn=tts_model.preview_multi_speaker_line,
            inputs=[multi_job_dir_state, multi_line_selector],
            outputs=[multi_line_audio, multi_line_speaker, multi_line_text, multi_line_status],
        )

        multi_line_selector.change(
            fn=tts_model.get_multi_speaker_line_editor_values,
            inputs=[multi_job_dir_state, multi_line_selector],
            outputs=[multi_line_speaker, multi_line_text, multi_line_status],
        )

        multi_line_selector.change(
            fn=tts_model.get_regenerate_line_defaults,
            inputs=[multi_job_dir_state, multi_line_selector, multi_speaker_table],
            outputs=[multi_regen_speed, multi_regen_pitch, multi_regen_ending_style, multi_regen_ending_length, multi_line_status],
        )

        multi_regenerate_btn.click(
            fn=tts_model.regenerate_multi_speaker_line,
            inputs=[
                multi_job_dir_state,
                multi_line_selector,
                multi_speaker_table,
                language_dropdown,
                multi_line_text,
                multi_make_mix,
                multi_silence_ms,
                multi_regen_speed,
                multi_regen_pitch,
                multi_regen_ending_style,
                multi_regen_ending_length,
            ],
            outputs=[
                multi_line_audio,
                multi_final_audio,
                multi_output_files,
                multi_line_status,
                multi_result_table,
                multi_line_selector,
            ],
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

        vc_batch_convert_btn.click(
            fn=convert_voice_folder_ui,
            inputs=[
                vc_input_folder,
                vc_output_subfolder,
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
            outputs=[vc_batch_output_files, vc_batch_status, vc_batch_output_dir, vc_runtime_status],
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

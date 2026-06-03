from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
from pipeline import DrivingScenePipeline
from src.common.config import load_app_config


st.set_page_config(page_title='驾驶场景语义描述系统', layout='wide')


def format_percent(v, digits=1):
    try:
        return f"{float(v) * 100:.{digits}f}%"
    except Exception:
        return "-"


def format_ms(v, digits=2):
    try:
        return f"{float(v):.{digits}f} ms"
    except Exception:
        return "-"


def collect_dataset_options(default_root: Path):
    options = []
    default_root = Path(default_root)

    if default_root.exists():
        options.append(default_root)

    project_root = Path(__file__).resolve().parent
    candidates = [
        project_root / "data" / "raw" / "nuScenes",
        project_root / "data" / "raw" / "nuscenes",
        Path("D:/python/driving_scene_desc/data/raw/nuScenes"),
        Path("D:/python/driving_scene_desc/data/raw/nuscenes"),
    ]

    for p in candidates:
        if p.exists() and p not in options:
            options.append(p)

    if not options:
        options.append(default_root)

    return [str(p) for p in options]


def collect_model_options(default_ckpt: Path):
    options = []
    default_ckpt = Path(default_ckpt)

    if default_ckpt.exists():
        options.append(default_ckpt)

    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "models" / "fusion"

    if model_dir.exists():
        for p in sorted(model_dir.glob("*.pt")):
            if p not in options:
                options.append(p)

    if not options:
        options.append(default_ckpt)

    return [str(p) for p in options]


def get_camera_mean_f1(res):
    """平均 F1"""
    metrics = res.get('metrics', {})

    if 'camera_mean_f1' in metrics:
        return float(metrics.get('camera_mean_f1', 0.0))

    direction_results = res.get('direction_results', [])
    f1s = []
    for item in direction_results:
        m = item.get('metrics', {})
        f1s.append(float(m.get('f1', 0.0)))

    if f1s:
        return sum(f1s) / len(f1s)

    return float(metrics.get('f1', 0.0))


def build_full_direction_targets(direction_results):
    """方向语义"""
    order = ['前方', '左前方', '右前方', '后方', '左后方', '右后方']
    out = {}

    for item in direction_results:
        direction = item.get('direction', '')
        pred_classes = item.get('pred_classes', []) or []

        keep = []
        for c in pred_classes:
            if c in ['vehicle', 'pedestrian', 'obstacle'] and c not in keep:
                keep.append(c)

        if keep:
            out[direction] = keep

    return {d: out[d] for d in order if d in out}


def compact_direction_results(direction_results):
    """各方向"""
    return direction_results


def get_previous_structured(pipe, frame_idx: int):
    """用于相邻帧动态描述"""
    if frame_idx <= 0:
        return None

    try:
        prev_scene = pipe.reader.scene_token_to_index[
            pipe.reader.get_sample_scene_token(pipe.reader.samples[frame_idx - 1])
        ]
        curr_scene = pipe.reader.scene_token_to_index[
            pipe.reader.get_sample_scene_token(pipe.reader.samples[frame_idx])
        ]

        if prev_scene != curr_scene:
            return None

        return pipe.process_frame(
            frame_idx - 1,
            method='ours',
            save_outputs=False,
            save_vis=False,
            save_result_json=False,
        )
    except Exception:
        return None


def run_current_frame(pipe, frame_idx: int):
    prev = get_previous_structured(pipe, frame_idx)
    with st.spinner('生成中...'):
        return pipe.process_frame(
            frame_idx,
            method='ours',
            save_outputs=True,
            save_vis=True,
            save_result_json=True,
            prev_structured=prev,
        )


def render_result(pipe, frame_idx: int, res):
    stem = f'ours_frame_{frame_idx:04d}'

    top_left, top_right = st.columns([2.2, 1.0])

    with top_left:
        st.subheader('可视化结果')

        tab1, tab2, tab3, tab4 = st.tabs([
            '六相机原图',
            'LiDAR投影叠加',
            '融合BEV',
            '原始BEV'
        ])

        with tab1:
            p = pipe.cfg.vis_dir / f'{stem}_sixcam.jpg'
            if p.exists():
                st.image(str(p), width='stretch')
            else:
                st.warning('未找到六相机原图。')

        with tab2:
            p = pipe.cfg.vis_dir / f'{stem}_sixcam_projected.jpg'
            if p.exists():
                st.image(str(p), width='stretch')
            else:
                st.warning('未找到 LiDAR 投影叠加图。')

        with tab3:
            p = pipe.cfg.vis_dir / f'{stem}_bev_fused.jpg'
            if p.exists():
                st.image(str(p), width='stretch')
            else:
                st.warning('未找到融合 BEV 图。')

        with tab4:
            p = pipe.cfg.vis_dir / f'{stem}_bev_raw.jpg'
            if p.exists():
                st.image(str(p), width='stretch')
            else:
                st.warning('未找到原始 BEV 图。')

    with top_right:
        st.subheader('语义描述')
        st.write(res.get('description_text', ''))

        st.subheader('评价指标')

        metrics = res.get('metrics', {})
        runtime = res.get('runtime_ms', {})

        camera_mean_f1 = get_camera_mean_f1(res)
        coherence_value = float(metrics.get('coherence_score', 0.0))
        runtime_value = float(runtime.get('core_runtime_ms', 0.0))

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                label='单帧识别准确率',
                value=format_percent(camera_mean_f1),
            )

        with c2:
            st.metric(
                label='描述连贯性',
                value=format_percent(coherence_value),
            )

        st.metric(
            label='实时性',
            value=format_ms(runtime_value),
        )

    st.markdown('---')

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.subheader('方向语义')

        full_direction_targets = build_full_direction_targets(
            res.get('direction_results', [])
        )

        direction_semantics = {
            'scene_idx': res.get('scene_idx'),
            'scene_condition': res.get('scene_condition'),
            'direction_targets': full_direction_targets,
            'focus_regions': res.get('focus_regions', []),
        }

        st.json(direction_semantics)

    with bottom_right:
        st.subheader('各方向概率')
        st.json(compact_direction_results(res.get('direction_results', [])))


# -----------------------------------------------------------------------------
# 主界面
# -----------------------------------------------------------------------------

default_cfg = load_app_config()


@st.cache_resource(show_spinner=False)
def load_pipe(dataset_root: str, fusion_ckpt: str):
    return DrivingScenePipeline(
        device='auto',
        use_clip=True,
        load_model=True,
        nuscenes_root=dataset_root,
        fusion_ckpt=fusion_ckpt,
    )


st.title('基于视觉语言大模型的驾驶场景语义描述系统')

with st.sidebar:
    st.header('运行设置')

    dataset_options = collect_dataset_options(default_cfg.nuscenes_root)
    model_options = collect_model_options(default_cfg.fusion_ckpt)

    dataset_root = st.selectbox(
        '选择数据集',
        dataset_options,
        index=0,
    )

    fusion_ckpt = st.selectbox(
        '选择FusionMLP模型',
        model_options,
        index=0,
    )

    try:
        pipe = load_pipe(dataset_root, fusion_ckpt)
        total_frames = len(pipe.reader)
    except Exception as e:
        st.error(f'加载失败：{e}')
        st.stop()

    frame_idx = st.number_input(
        '帧序号',
        min_value=0,
        max_value=total_frames - 1,
        value=0,
        step=1,
        key='frame_idx',
    )
    frame_idx = int(frame_idx)

    run_clicked = st.button(
        '语义描述生成',
        type='primary',
    )

    if run_clicked:
        st.session_state['auto_generate'] = True

# 点击过一次“语义描述生成”后，后续点击帧序号 + / - 会自动重新生成当前帧结果。
should_run = st.session_state.get('auto_generate', False)

if should_run:
    res = run_current_frame(pipe, frame_idx)
    render_result(pipe, frame_idx, res)

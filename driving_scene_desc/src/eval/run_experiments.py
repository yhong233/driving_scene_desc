from __future__ import annotations
import json
import pandas as pd
from tqdm import tqdm

from src.datasets.nuscenes_reader import TARGET_CLASSES
from src.eval.metrics import multilabel_counts, metrics_from_counts, frame_prf, safe_mean

CAMERA_ORDER = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


def _as_int(v, default=-1):
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _safe_get(res, key, default=''):
    if isinstance(res, dict):
        return res.get(key, default)
    return default


def _event_to_text(event):
    if event is None:
        return ''
    if isinstance(event, str):
        return event
    if isinstance(event, dict):
        cls = event.get('class', '')
        ev_type = event.get('type', '')
        from_dir = event.get('from', '')
        to_dir = event.get('to', '')
        parts = []
        if cls:
            parts.append(str(cls))
        if ev_type:
            parts.append(str(ev_type))
        if from_dir or to_dir:
            parts.append(f'{from_dir}->{to_dir}')
        if parts:
            return ':'.join(parts)
        try:
            return json.dumps(event, ensure_ascii=False)
        except Exception:
            return str(event)
    try:
        return json.dumps(event, ensure_ascii=False)
    except Exception:
        return str(event)


def _events_to_text(events):
    if events is None:
        return ''
    if isinstance(events, str):
        return events
    if isinstance(events, dict):
        return _event_to_text(events)
    if isinstance(events, (list, tuple, set)):
        return ';'.join([s for s in (_event_to_text(e) for e in events) if s])
    return _event_to_text(events)


def flatten_result(res):
    """保留帧级结果，主要用于 GUI 或单帧调试。"""
    metrics = res.get('metrics', {}) or {}
    runtime = res.get('runtime_ms', {}) or {}
    coherence_detail = metrics.get('coherence_detail', {}) or {}

    target_classes = res.get('target_classes', res.get('predicted_classes', [])) or []

    row = {
        'frame_idx': _as_int(res.get('frame_idx', res.get('sample_idx', -1))),
        'sample_token': res.get('sample_token', ''),
        'scene_idx': _as_int(res.get('scene_idx', -1)),
        'scene_name': res.get('scene_name', ''),
        'condition': res.get('scene_condition', res.get('condition', 'unknown')),
        'method': res.get('method', 'ours'),
        'predicted_classes': ','.join(target_classes),
        'reference_classes': ','.join(res.get('reference_classes', []) or []),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'f1': metrics.get('f1', 0),
        'spatial_score': metrics.get('spatial_score', 0),
        'coherence_score': metrics.get('coherence_score', 0),
        'description_text': res.get('description_text', ''),
        'dynamic_description': res.get('dynamic_description', ''),
        'temporal_events': _events_to_text(res.get('temporal_events', res.get('dynamic_events', []))),
        'core_runtime_ms': runtime.get('core_runtime_ms', 0),

        # 描述连贯性细项，便于后续论文分析或调试。
        'coherence_scene_score': coherence_detail.get('scene_score', 0),
        'coherence_key_semantic_score': coherence_detail.get('key_semantic_score', 0),
        'coherence_semantic_consistency_score': coherence_detail.get('semantic_consistency_score', 0),
        'coherence_dynamic_score': coherence_detail.get('dynamic_score', 0),
        'coherence_language_score': coherence_detail.get('language_score', 0),
    }
    for c in TARGET_CLASSES:
        row[f'pred_{c}'] = int(c in target_classes)
        row[f'true_{c}'] = int(c in (res.get('reference_classes', []) or []))
        row[f'prob_{c}'] = float(res.get('confidence_scores', {}).get(c, 0.0))
    for k, v in runtime.items():
        row[f'time_{k}'] = v
    return row


def flatten_direction_results(res):
    """将一帧结果展开成六个方向级评价样本。"""
    rows = []
    runtime = res.get('runtime_ms', {}) or {}
    metrics_all = res.get('metrics', {}) or {}
    coherence_detail = metrics_all.get('coherence_detail', {}) or {}

    frame_idx = _as_int(res.get('frame_idx', res.get('sample_idx', -1)))
    sample_token = res.get('sample_token', '')
    scene_idx = _as_int(res.get('scene_idx', -1))
    scene_name = res.get('scene_name', '')
    condition = res.get('scene_condition', res.get('condition', 'unknown'))
    method = res.get('method', 'ours')

    for item in res.get('direction_results', []) or []:
        true_classes = item.get('true_classes', []) or []
        pred_classes = item.get('pred_classes', []) or []
        p, r, f1 = frame_prf(true_classes, pred_classes)

        row = {
            'frame_idx': frame_idx,
            'sample_token': sample_token,
            'scene_idx': scene_idx,
            'scene_name': scene_name,
            'condition': condition,
            'scene_condition': condition,
            'method': method,
            'camera_name': item.get('camera_name', ''),
            'direction': item.get('direction', ''),
            'predicted_classes': ','.join(pred_classes),
            'reference_classes': ','.join(true_classes),
            'precision': p,
            'recall': r,
            'f1': f1,
            'frame_precision': metrics_all.get('frame_precision', 0),
            'frame_recall': metrics_all.get('frame_recall', 0),
            'frame_f1': metrics_all.get('frame_f1', 0),
            'camera_mean_precision': metrics_all.get('camera_mean_precision', metrics_all.get('precision', 0)),
            'camera_mean_recall': metrics_all.get('camera_mean_recall', metrics_all.get('recall', 0)),
            'camera_mean_f1': metrics_all.get('camera_mean_f1', metrics_all.get('f1', 0)),
            'spatial_score': metrics_all.get('spatial_score', 0),
            'coherence_score': metrics_all.get('coherence_score', 0),
            'description_text': res.get('description_text', ''),
            'dynamic_description': res.get('dynamic_description', ''),
            'temporal_events': _events_to_text(res.get('temporal_events', res.get('dynamic_events', []))),
            'core_runtime_ms': runtime.get('core_runtime_ms', 0),

            # 描述连贯性细项。
            'coherence_scene_score': coherence_detail.get('scene_score', 0),
            'coherence_key_semantic_score': coherence_detail.get('key_semantic_score', 0),
            'coherence_semantic_consistency_score': coherence_detail.get('semantic_consistency_score', 0),
            'coherence_dynamic_score': coherence_detail.get('dynamic_score', 0),
            'coherence_language_score': coherence_detail.get('language_score', 0),
        }
        for c in TARGET_CLASSES:
            row[f'pred_{c}'] = int(c in pred_classes)
            row[f'true_{c}'] = int(c in true_classes)
            row[f'prob_{c}'] = float(item.get('confidence_scores', {}).get(c, 0.0))
        for k, v in runtime.items():
            row[f'time_{k}'] = v
        rows.append(row)
    return rows

def _empty_summary(method: str):
    return {
        'method': method,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'support': 0,
        'pred_positive': 0,
        'spatial_score': 0.0,
        'coherence_score': 0.0,
        'core_runtime_ms': 0.0,
        'num_samples': 0,
        'num_frames': 0,
    }


def evaluate_prediction_df(df: pd.DataFrame, method: str):
    """把方向级样本混合在一起，计算整体方向级 micro-F1 与 per-class 指标。"""
    if len(df) == 0:
        return _empty_summary(method), []
    y_true = df[[f'true_{c}' for c in TARGET_CLASSES]].values.astype(int)
    y_pred = df[[f'pred_{c}' for c in TARGET_CLASSES]].values.astype(int)
    counts = multilabel_counts(y_true, y_pred, TARGET_CLASSES)
    metrics = metrics_from_counts(counts)
    summary = {
        'method': method,
        'precision': metrics['overall']['precision'],
        'recall': metrics['overall']['recall'],
        'f1': metrics['overall']['f1'],
        'support': metrics['overall']['support'],
        'pred_positive': metrics['overall']['pred_positive'],
        'spatial_score': df['spatial_score'].mean() if 'spatial_score' in df else 0,
        'coherence_score': df['coherence_score'].mean() if 'coherence_score' in df else 0,
        'core_runtime_ms': df['core_runtime_ms'].mean() if 'core_runtime_ms' in df else 0,
        'num_samples': len(df),
        'num_frames': int(df['frame_idx'].nunique()) if 'frame_idx' in df else len(df),
    }
    per_class = []
    for c in TARGET_CLASSES:
        per_class.append({'method': method, 'class': c, **metrics[c]})
    return summary, per_class


def evaluate_by_camera(df: pd.DataFrame, method: str):
    """分别计算六个相机方向上的整体方向级指标与 per-class 指标。"""
    camera_rows = []
    camera_class_rows = []
    for camera_name, g in df.groupby('camera_name'):
        summary, per_class = evaluate_prediction_df(g, method)
        direction = g['direction'].iloc[0] if 'direction' in g and len(g) else ''
        camera_rows.append({'method': method, 'camera_name': camera_name, 'direction': direction, **summary})
        for item in per_class:
            camera_class_rows.append({'method': method, 'camera_name': camera_name, 'direction': direction, **item})
    return camera_rows, camera_class_rows


def evaluate_camera_mean(camera_rows: list[dict], camera_class_rows: list[dict], method: str):
    """计算六相机平均指标。"""
    cam_df = pd.DataFrame(camera_rows)
    if len(cam_df) == 0:
        method_mean = {'method': method, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                       'spatial_score': 0.0, 'coherence_score': 0.0, 'core_runtime_ms': 0.0,
                       'num_cameras': 0, 'num_samples': 0, 'num_frames': 0,
                       'metric_type': 'camera_mean_overall'}
    else:
        method_mean = {
            'method': method,
            'precision': safe_mean(cam_df['precision']),
            'recall': safe_mean(cam_df['recall']),
            'f1': safe_mean(cam_df['f1']),
            'spatial_score': safe_mean(cam_df['spatial_score']),
            'coherence_score': safe_mean(cam_df['coherence_score']),
            'core_runtime_ms': safe_mean(cam_df['core_runtime_ms']),
            'num_cameras': int(cam_df['camera_name'].nunique()),
            'num_samples': int(cam_df['num_samples'].sum()),
            'num_frames': int(cam_df['num_frames'].max()) if 'num_frames' in cam_df else 0,
            'metric_type': 'camera_mean_overall',
        }

    class_mean_rows = []
    cc_df = pd.DataFrame(camera_class_rows)
    if len(cc_df):
        for c, g in cc_df.groupby('class'):
            valid = g[g['support'] > 0]
            src = valid if len(valid) else g
            class_mean_rows.append({
                'method': method,
                'class': c,
                'precision': safe_mean(src['precision']),
                'recall': safe_mean(src['recall']),
                'f1': safe_mean(src['f1']),
                'support': int(g['support'].sum()),
                'pred_positive': int(g['pred_positive'].sum()),
                'num_valid_cameras': int(valid['camera_name'].nunique()),
                'num_cameras': int(g['camera_name'].nunique()),
                'metric_type': 'camera_mean_per_class_support_gt0',
            })
    return method_mean, class_mean_rows


def evaluate_by_condition_camera_mean(df: pd.DataFrame, method: str):
    rows = []
    if 'condition' not in df:
        return rows
    for cond, g in df.groupby('condition'):
        cam_rows, cam_class_rows = evaluate_by_camera(g, method)
        mean_row, _ = evaluate_camera_mean(cam_rows, cam_class_rows, method)
        rows.append({'condition': cond, **mean_row})
    return rows


def _prediction_csv_matches_frames(csv_path, frame_indices):
    """检查已有 predictions CSV 是否覆盖当前要评价的帧。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False, None, '无法读取已有 CSV'

    if 'frame_idx' not in df.columns:
        return False, df, '已有 CSV 缺少 frame_idx 列'

    expected = set(int(i) for i in frame_indices)
    got = set(int(i) for i in df['frame_idx'].dropna().unique())

    if expected != got:
        missing = sorted(expected - got)[:10]
        extra = sorted(got - expected)[:10]
        msg = f'帧范围不一致：expected={len(expected)}帧, got={len(got)}帧, missing_sample={missing}, extra_sample={extra}'
        return False, df, msg

    return True, df, '帧范围一致'


def run_methods(
    pipeline,
    methods,
    frame_indices=None,
    save_csv=True,
    skip_existing=False,
    save_vis=False,
    save_json=False,
    reuse_ours_from_04=True,
):
    """运行多方法对比实验。"""
    if frame_indices is None:
        frame_indices = list(range(len(pipeline.reader)))

    all_rows = []
    summaries, per_classes = [], []
    condition_rows, scene_rows = [], []
    camera_rows_all, camera_class_rows_all = [], []
    camera_mean_rows, per_class_camera_mean_rows = [], []
    condition_camera_mean_rows = []

    for method in methods:
        out = pipeline.cfg.csv_dir / f'predictions_{method}.csv'

        should_reuse_existing = (skip_existing or (reuse_ours_from_04 and method == 'ours')) and out.exists()
        if should_reuse_existing:
            matched, df_existing, msg = _prediction_csv_matches_frames(out, frame_indices)
            if matched:
                if method == 'ours' and reuse_ours_from_04:
                    print(f'[REUSE] ours: 复用 04_run_inference.py 已生成的 {out}')
                else:
                    print(f'[SKIP] {method}: 已存在 {out}，直接读取并汇总。')
                df = df_existing
            else:
                print(f'[WARN] {method}: 已有 {out} 但不能复用，原因：{msg}。将重新运行该方法。')
                rows = []
                prev_structured = None
                prev_scene_idx = None

                for idx in tqdm(frame_indices, desc=f'run {method}'):
                    scene_idx = pipeline.reader.scene_token_to_index[
                        pipeline.reader.get_sample_scene_token(pipeline.reader.samples[idx])
                    ]
                    if prev_scene_idx is not None and scene_idx != prev_scene_idx:
                        prev_structured = None

                    should_save_outputs = (method == 'ours') and (save_vis or save_json)
                    res = pipeline.process_frame(
                        idx,
                        method=method,
                        save_outputs=should_save_outputs,
                        save_vis=save_vis,
                        save_result_json=save_json,
                        prev_structured=prev_structured,
                    )
                    rows.extend(flatten_direction_results(res))
                    prev_structured = res
                    prev_scene_idx = scene_idx

                df = pd.DataFrame(rows)
                if save_csv:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out, index=False, encoding='utf-8-sig')
                    print('saved', out)
        else:
            rows = []
            prev_structured = None
            prev_scene_idx = None

            for idx in tqdm(frame_indices, desc=f'run {method}'):
                scene_idx = pipeline.reader.scene_token_to_index[
                    pipeline.reader.get_sample_scene_token(pipeline.reader.samples[idx])
                ]
                if prev_scene_idx is not None and scene_idx != prev_scene_idx:
                    prev_structured = None

                # 对比实验阶段默认不保存可视化，避免大量图片保存。
                should_save_outputs = (method == 'ours') and (save_vis or save_json)
                res = pipeline.process_frame(
                    idx,
                    method=method,
                    save_outputs=should_save_outputs,
                    save_vis=save_vis,
                    save_result_json=save_json,
                    prev_structured=prev_structured,
                )
                rows.extend(flatten_direction_results(res))
                prev_structured = res
                prev_scene_idx = scene_idx

            df = pd.DataFrame(rows)
            if save_csv:
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, index=False, encoding='utf-8-sig')
                print('saved', out)

        if len(df) == 0:
            print(f'[WARN] {method}: 预测结果为空，跳过汇总。')
            continue

        summary, pc = evaluate_prediction_df(df, method)
        summaries.append(summary)
        per_classes.extend(pc)
        all_rows.append(df)

        # 按条件、按 scene 的方向级整体指标。
        if 'condition' in df:
            for cond, g in df.groupby('condition'):
                s, _ = evaluate_prediction_df(g, method)
                condition_rows.append({'condition': cond, **s})
        if 'scene_idx' in df:
            for scene, g in df.groupby('scene_idx'):
                s, _ = evaluate_prediction_df(g, method)
                condition = g['condition'].iloc[0] if 'condition' in g and len(g) else ''
                scene_rows.append({'scene_idx': scene, 'condition': condition, **s})

        # 六相机方向单独指标与六相机平均指标。
        cam_rows, cam_class_rows = evaluate_by_camera(df, method)
        camera_rows_all.extend(cam_rows)
        camera_class_rows_all.extend(cam_class_rows)
        cam_mean, class_cam_mean = evaluate_camera_mean(cam_rows, cam_class_rows, method)
        camera_mean_rows.append(cam_mean)
        per_class_camera_mean_rows.extend(class_cam_mean)
        condition_camera_mean_rows.extend(evaluate_by_condition_camera_mean(df, method))

    if save_csv:
        pipeline.cfg.csv_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(pipeline.cfg.csv_dir / 'summary_by_method.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(per_classes).to_csv(pipeline.cfg.csv_dir / 'summary_per_class.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(condition_rows).to_csv(pipeline.cfg.csv_dir / 'summary_by_condition.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(scene_rows).to_csv(pipeline.cfg.csv_dir / 'summary_by_scene.csv', index=False, encoding='utf-8-sig')

        pd.DataFrame(camera_rows_all).to_csv(pipeline.cfg.csv_dir / 'summary_by_camera.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(camera_class_rows_all).to_csv(pipeline.cfg.csv_dir / 'summary_by_camera_class.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(camera_mean_rows).to_csv(pipeline.cfg.csv_dir / 'summary_by_method_camera_mean.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(per_class_camera_mean_rows).to_csv(pipeline.cfg.csv_dir / 'summary_per_class_camera_mean.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(condition_camera_mean_rows).to_csv(pipeline.cfg.csv_dir / 'summary_by_condition_camera_mean.csv', index=False, encoding='utf-8-sig')

    if all_rows:
        return pd.concat(all_rows, ignore_index=True), pd.DataFrame(camera_mean_rows)
    return pd.DataFrame(), pd.DataFrame(camera_mean_rows)

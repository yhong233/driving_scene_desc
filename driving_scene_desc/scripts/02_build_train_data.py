from pathlib import Path
import sys, argparse
import pandas as pd
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common.config import load_app_config, camera_names, camera_to_zh_direction
from src.datasets.nuscenes_reader import MiniNuScenesReader, TARGET_CLASSES
from src.geometry.projection import project_frame
from src.semantics.clip_adapter import ClipAdapter
from src.features.feature_extract import make_direction_feature_row, ALL_FEATURE_NAMES

parser = argparse.ArgumentParser()
parser.add_argument('--use-clip', action='store_true', help='启用真实 CLIP；不加则 CLIP 分数置零。')
parser.add_argument('--device', default='auto')
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--max-points', type=int, default=None)
args = parser.parse_args()

cfg = load_app_config()
reader = MiniNuScenesReader(cfg.nuscenes_root, cfg.nuscenes_version, verbose=False)
clipper = ClipAdapter(cfg.clip_ckpt, cfg.class_cfg, device=args.device, enabled=args.use_clip)
rows = []
N = min(len(reader), args.limit) if args.limit else len(reader)
for i in tqdm(range(N), desc='build direction train features'):
    frame = reader.get_frame(i, load_images=True, max_points=args.max_points)
    projections = project_frame(frame, reader.nusc)
    camera_scores = clipper.score_images(frame.images)
    direction_labels = reader.get_direction_labels(frame)

    for cam in camera_names():
        row = make_direction_feature_row(frame, projections, camera_scores.get(cam, {}), cam)
        row.update({
            'frame_idx': frame.frame_idx,
            'sample_token': frame.sample_token,
            'scene_idx': frame.scene_idx,
            'scene_name': frame.scene_name,
            'condition': frame.scene_condition,
            'camera_name': cam,
            'direction': camera_to_zh_direction(cam),
        })
        for c in TARGET_CLASSES:
            row[f'label_{c}'] = direction_labels[cam][c]
        rows.append(row)

df = pd.DataFrame(rows)
meta = ['frame_idx','sample_token','scene_idx','scene_name','condition','camera_name','direction']
labels = [f'label_{c}' for c in TARGET_CLASSES]
df = df[meta + ALL_FEATURE_NAMES + labels]
out = cfg.csv_dir / 'train_features.csv'
df.to_csv(out, index=False, encoding='utf-8-sig')
print('saved:', out)
print('sample count:', len(df), '(约等于 frame_count × 6)')
print('labels mean by direction sample:')
print(df[labels].mean())
print('labels sum by direction sample:')
print(df[labels].sum())

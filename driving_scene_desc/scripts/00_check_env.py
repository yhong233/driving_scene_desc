from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common.config import load_app_config

cfg = load_app_config()
print('project_root:', cfg.project_root)
print('nuscenes_root:', cfg.nuscenes_root, cfg.nuscenes_root.exists())
print('version dir:', cfg.nuscenes_root / cfg.nuscenes_version, (cfg.nuscenes_root / cfg.nuscenes_version).exists())
print('clip_ckpt:', cfg.clip_ckpt, cfg.clip_ckpt.exists())
try:
    import torch
    print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
except Exception as e:
    print('torch error:', e)
try:
    import clip
    print('clip package: OK')
except Exception as e:
    print('clip package missing:', e)
try:
    from src.datasets.nuscenes_reader import MiniNuScenesReader
    reader = MiniNuScenesReader(cfg.nuscenes_root, cfg.nuscenes_version, verbose=True)
    print('nuScenes samples:', len(reader), 'scenes:', len(reader.nusc.scene))
except Exception as e:
    print('nuScenes load error:', e)

from pathlib import Path
import sys
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common.config import load_app_config
from src.datasets.nuscenes_reader import MiniNuScenesReader

cfg = load_app_config()
reader = MiniNuScenesReader(cfg.nuscenes_root, cfg.nuscenes_version, verbose=True)
rows = reader.make_index_rows()
df = pd.DataFrame(rows)
out = cfg.csv_dir / 'frame_index.csv'
df.to_csv(out, index=False, encoding='utf-8-sig')
print('saved:', out)
print(df.groupby(['scene_idx','condition']).size())

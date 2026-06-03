from pathlib import Path
import sys, argparse
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common.config import load_app_config
from src.models.train_utils import train_fusion_model

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', default='auto')
parser.add_argument('--early-stop-patience', type=int, default=15)
parser.add_argument('--max-pos-weight', type=float, default=6.0)
parser.add_argument('--threshold-min', type=float, default=0.20)
parser.add_argument('--distill-weight', type=float, default=0.02)
args = parser.parse_args()

cfg = load_app_config()
features_csv = cfg.csv_dir / 'train_features.csv'
train_fusion_model(
    features_csv,
    cfg.fusion_ckpt,
    cfg.scene_cfg,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    device=args.device,
    early_stop_patience=args.early_stop_patience,
    max_pos_weight=args.max_pos_weight,
    threshold_min=args.threshold_min,
    distill_weight=args.distill_weight,
)

from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from PIL import Image

from src.common.config import camera_names
from src.features.feature_extract import CLIP_CLASSES

class ClipAdapter:
    """OpenAI CLIP 适配器。"""
    def __init__(self, ckpt_path: str | Path, class_cfg: dict, device: str = 'auto', enabled: bool = True):
        self.ckpt_path = Path(str(ckpt_path).replace('\\', '/'))
        self.class_cfg = class_cfg
        self.enabled = enabled
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
        self.model = None
        self.preprocess = None
        self.text_features = None
        self.available = False
        if enabled:
            self._load()

    def _load(self):
        try:
            import clip  # type: ignore
        except Exception as e:
            raise ImportError('未安装 OpenAI CLIP。请先运行：pip install git+https://github.com/openai/CLIP.git') from e
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f'CLIP 权重文件不存在：{self.ckpt_path}。请确认 models/clip/ViT-B-32.pt 路径正确。')
        try:
            self.model, self.preprocess = clip.load(str(self.ckpt_path), device=self.device, jit=False)
            self.model.eval()
            self._build_text_features(clip)
            self.available = True
        except Exception as e:
            raise RuntimeError(f'CLIP 加载失败，已停止运行，避免使用非真实 CLIP 分数。原因：{e}') from e

    def _build_text_features(self, clip_module):
        prompts_by_class = self.class_cfg.get('clip_classes', {})
        class_feats = []
        with torch.no_grad():
            for c in CLIP_CLASSES:
                prompts = prompts_by_class.get(c, {}).get('prompts', [c])
                tokens = clip_module.tokenize(prompts).to(self.device)
                feat = self.model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feat = feat.mean(dim=0, keepdim=True)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                class_feats.append(feat)
            self.text_features = torch.cat(class_feats, dim=0)

    def score_images(self, images: Dict[str, Image.Image]) -> Dict[str, Dict[str, float]]:
        if not self.enabled:
            return {cam: {c: 0.0 for c in CLIP_CLASSES} for cam in camera_names()}
        if not self.available or self.model is None or self.preprocess is None or self.text_features is None:
            raise RuntimeError('CLIP 已启用但不可用。请检查权重路径、clip 包和运行设备。')
        scores = {}
        with torch.no_grad():
            for cam in camera_names():
                img_tensor = self.preprocess(images[cam]).unsqueeze(0).to(self.device)
                img_feat = self.model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                logits = 100.0 * img_feat @ self.text_features.T
                probs = logits.softmax(dim=-1).squeeze(0).detach().cpu().numpy()
                scores[cam] = {c: float(probs[i]) for i, c in enumerate(CLIP_CLASSES)}
        return scores

    def aggregate_scores(self, camera_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if not camera_scores:
            return {c: 0.0 for c in CLIP_CLASSES}
        out = {}
        for c in CLIP_CLASSES:
            vals = [camera_scores[cam].get(c, 0.0) for cam in camera_scores]
            out[c] = float(np.max(vals) * 0.55 + np.mean(vals) * 0.45)
        return out

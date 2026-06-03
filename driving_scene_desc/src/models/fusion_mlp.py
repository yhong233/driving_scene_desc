from __future__ import annotations
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

from src.datasets.nuscenes_reader import TARGET_CLASSES
from src.features.feature_extract import (
    LIDAR_FEATURE_NAMES,
    IMAGE_FEATURE_NAMES,
    CLIP_FEATURE_NAMES,
    EVIDENCE_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
)


class FusionMLP(nn.Module):
    """轻量化多模态 FusionMLP 融合网络"""

    def __init__(
        self,
        lidar_dim: Optional[int] = None,
        image_dim: Optional[int] = None,
        clip_dim: Optional[int] = None,
        evidence_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        hidden: int = 64,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.lidar_dim = lidar_dim or len(LIDAR_FEATURE_NAMES)
        self.image_dim = image_dim or len(IMAGE_FEATURE_NAMES)
        self.clip_dim = clip_dim or len(CLIP_FEATURE_NAMES)
        self.evidence_dim = evidence_dim or len(EVIDENCE_FEATURE_NAMES)
        self.context_dim = context_dim or len(CONTEXT_FEATURE_NAMES)
        self.num_classes = int(num_classes or len(TARGET_CLASSES))
        self.hidden = hidden

        self.lidar_branch = nn.Sequential(
            nn.Linear(self.lidar_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.image_branch = nn.Sequential(
            nn.Linear(self.image_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.clip_branch = nn.Sequential(
            nn.Linear(self.clip_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.evidence_branch = nn.Sequential(
            nn.Linear(self.evidence_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.context_branch = nn.Sequential(
            nn.Linear(self.context_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        # 拼接五个分支特征，输入普通 MLP 融合层。
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(self.num_classes)])

    def forward(self, x_lidar, x_image, x_clip, x_evidence, x_context):
        z_lidar = self.lidar_branch(x_lidar)
        z_image = self.image_branch(x_image)
        z_clip = self.clip_branch(x_clip)
        z_evi = self.evidence_branch(x_evidence)
        z_ctx = self.context_branch(x_context)

        z_cat = torch.cat([z_lidar, z_image, z_clip, z_evi, z_ctx], dim=1)
        z = self.fusion(z_cat)
        logits = torch.cat([head(z) for head in self.heads], dim=1)

        return logits, {
            'z_lidar': z_lidar,
            'z_image': z_image,
            'z_clip': z_clip,
            'z_evidence': z_evi,
            'z_context': z_ctx,
            'z_fused': z,
        }


def fusion_loss(logits, y, aux, distill_weight: float = 0.02, pos_weight=None):
    """FusionMLP 损失函数。"""
    cls_loss = F.binary_cross_entropy_with_logits(
        logits,
        y.float(),
        pos_weight=pos_weight,
    )
    cos = F.cosine_similarity(aux['z_lidar'], aux['z_image'], dim=1)
    distill = (1.0 - cos).mean()
    total = cls_loss + distill_weight * distill
    return total, {
        'cls_loss': float(cls_loss.detach().cpu()),
        'distill_loss': float(distill.detach().cpu()),
    }


def make_tensor_batch(df, device='cpu', zero_clip=False, zero_image=False):
    def t(cols):
        return torch.tensor(df[cols].values, dtype=torch.float32, device=device)

    x_lidar = t(LIDAR_FEATURE_NAMES)
    x_image = t(IMAGE_FEATURE_NAMES)
    x_clip = t(CLIP_FEATURE_NAMES)
    x_evi = t(EVIDENCE_FEATURE_NAMES)
    x_ctx = t(CONTEXT_FEATURE_NAMES)

    if zero_clip:
        x_clip = torch.zeros_like(x_clip)
    if zero_image:
        x_image = torch.zeros_like(x_image)

    return x_lidar, x_image, x_clip, x_evi, x_ctx

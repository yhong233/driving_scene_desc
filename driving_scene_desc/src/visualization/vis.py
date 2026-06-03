from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from src.common.config import camera_names, camera_to_zh_direction


PAPER_CAMERA_GRID: List[List[str]] = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
]


def paper_camera_order() -> List[str]:
    """返回六相机顺序。"""
    return [cam for row in PAPER_CAMERA_GRID for cam in row]


def _fit_image_to_cell(img: Image.Image, cell_size: tuple[int, int], keep_aspect: bool = True) -> Image.Image:
    cell_w, cell_h = cell_size
    img = img.convert('RGB')
    if not keep_aspect:
        return img.resize((cell_w, cell_h), Image.BILINEAR)

    src_w, src_h = img.size
    scale = min(cell_w / max(src_w, 1), cell_h / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new('RGB', (cell_w, cell_h), (0, 0, 0))
    x = (cell_w - new_w) // 2
    y = (cell_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _draw_camera_label(img: Image.Image, cam: str, show_zh: bool = True) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    zh = camera_to_zh_direction(cam)
    text = f'{cam}' if not show_zh else f'{zh}  {cam}'
    try:
        bbox = draw.textbbox((0, 0), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = len(text) * 8, 14
    bg_w = min(out.size[0], max(120, tw + 14))
    bg_h = min(out.size[1], max(24, th + 10))
    draw.rectangle((0, 0, bg_w, bg_h), fill=(0, 0, 0))
    draw.text((6, max(4, (bg_h - th) // 2 - 1)), text, fill=(255, 255, 255))
    return out


def _make_camera_grid(images_by_cam: Dict[str, Image.Image], cell_size: tuple[int, int]) -> Image.Image:
    """按照 PAPER_CAMERA_GRID 拼接六相机图像。"""
    cell_w, cell_h = cell_size
    canvas = Image.new('RGB', (cell_w * 3, cell_h * 2), (0, 0, 0))
    for r, row in enumerate(PAPER_CAMERA_GRID):
        for c, cam in enumerate(row):
            im = images_by_cam.get(cam)
            if im is None:
                im = Image.new('RGB', (cell_w, cell_h), (20, 20, 20))
                draw = ImageDraw.Draw(im)
                draw.text((10, 10), f'Missing {cam}', fill=(255, 255, 255))
            canvas.paste(im, (c * cell_w, r * cell_h))
    return canvas


def draw_projection_on_image(img: Image.Image, proj, max_points=1800) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    pts = proj.pixel_coords
    dep = proj.depth_values
    if len(pts) == 0:
        return out
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts)-1, max_points).astype(int)
        pts = pts[idx]; dep = dep[idx]
    dmin, dmax = np.percentile(dep, 5), np.percentile(dep, 95)
    for (u, v), d in zip(pts, dep):
        t = float((d - dmin) / max(dmax - dmin, 1e-6))
        # 不指定 matplotlib 样式
        color = (int(255*(1-t)), int(255*t), 80)
        draw.ellipse((u-1, v-1, u+1, v+1), fill=color)
    return out


def save_six_camera_projection(
    frame,
    projections,
    path: str | Path,
    cell_size: tuple[int, int] = (520, 292),
    keep_aspect: bool = False,
):
    """保存六相机原图拼接。 """
    images_by_cam: Dict[str, Image.Image] = {}
    for cam in paper_camera_order():
        im = _fit_image_to_cell(frame.images[cam], cell_size, keep_aspect=keep_aspect)
        im = _draw_camera_label(im, cam, show_zh=True)
        images_by_cam[cam] = im

    canvas = _make_camera_grid(images_by_cam, cell_size)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path



def _ensure_parent(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _robust_normalize(values: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """鲁棒归一化到 [0, 1]，避免极少数异常点影响色彩范围。"""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    v = values[finite]
    lo = float(np.percentile(v, low_pct))
    hi = float(np.percentile(v, high_pct))
    if hi - lo < 1e-6:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return out


def _downsample_points(xyz: np.ndarray, values: np.ndarray | None, max_points: int = 35000):
    """固定间隔降采样，保证 raw / fused 图形稳定可复现。"""
    n = len(xyz)
    if n <= max_points:
        return xyz, values
    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    xyz2 = xyz[idx]
    values2 = values[idx] if values is not None and len(values) == n else values
    return xyz2, values2


def _paper_bev_limits(xyz: np.ndarray, xlim=None, ylim=None):
    """返回统一 BEV 范围。默认采用固定坐标范围，保证不同帧可比。"""
    if xlim is None:
        xlim = (-60.0, 85.0)
    if ylim is None:
        ylim = (-85.0, 85.0)
    return xlim, ylim


def _draw_ego_vehicle(ax):
    """在原点位置绘制自车位置标记"""
    ax.scatter([0], [0], s=56, c='black', edgecolors='white', linewidths=0.9, zorder=5)
    ax.text(1.8, 1.8, 'ego', fontsize=8, color='black', zorder=6)


def _draw_paper_bev(
    points: np.ndarray,
    values: np.ndarray,
    path: str | Path,
    title: str,
    cbar_label: str,
    value_range: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    point_size: float = 1.0,
    dpi: int = 200,
):
    """ BEV 散点图。横轴为 x / forward，纵轴为 y / left。"""
    _ensure_parent(path)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x / forward', fontsize=9)
        ax.set_ylabel('y / left', fontsize=9)
        ax.text(0.5, 0.5, 'No LiDAR points', ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return path

    xyz = points[:, :3]
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) != len(xyz):
        raise ValueError(f'BEV value length mismatch: {len(values)} vs points {len(xyz)}')

    xyz, values = _downsample_points(xyz, values, max_points=35000)
    xlim, ylim = _paper_bev_limits(xyz, xlim=xlim, ylim=ylim)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    scatter_kwargs = dict(
        x=xyz[:, 0],
        y=xyz[:, 1],
        c=values,
        s=point_size,
        cmap='turbo',
        linewidths=0,
        alpha=0.95,
    )
    if value_range is not None:
        scatter_kwargs['vmin'] = float(value_range[0])
        scatter_kwargs['vmax'] = float(value_range[1])
    sc = ax.scatter(**scatter_kwargs)

    _draw_ego_vehicle(ax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x / forward', fontsize=9)
    ax.set_ylabel('y / left', fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.tick_params(labelsize=8)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path



def _image_edge_map(img: Image.Image) -> np.ndarray:
    """计算图像局部纹理/边缘强度，返回 [H, W] 的 0~1 矩阵。"""
    arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    # 用一阶梯度近似 Sobel
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    return _robust_normalize(grad.reshape(-1), low_pct=5.0, high_pct=99.0).reshape(gray.shape)


def _sample_image_edge(edge_map: np.ndarray, pixel_coords: np.ndarray) -> np.ndarray:
    """在投影像素处采样图像边缘/纹理强度。"""
    h, w = edge_map.shape[:2]
    uv = np.asarray(pixel_coords, dtype=np.float32).reshape(-1, 2)
    u = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, h - 1)
    return edge_map[v, u].astype(np.float32)


def _camera_center_reliability(img: Image.Image, pixel_coords: np.ndarray, margin_ratio: float = 0.10) -> np.ndarray:
    """计算投影点位于相机图像中的可靠性，靠近图像边界的点降权。"""
    w, h = img.size
    uv = np.asarray(pixel_coords, dtype=np.float32).reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    margin = np.minimum.reduce([u, v, (w - 1) - u, (h - 1) - v])
    ref = max(8.0, min(w, h) * float(margin_ratio))
    rel = np.clip(margin / ref, 0.0, 1.0)
    return (0.35 + 0.65 * rel).astype(np.float32)


def _semantic_reliability(score_dict: Dict[str, float] | None) -> float:
    """从 CLIP 语义分数中得到轻量语义可靠性，只做辅助，不直接主导点云强度。"""
    if not score_dict:
        return 0.0
    semantic_keys = [
        'vehicle', 'pedestrian', 'obstacle',
        'road', 'building', 'vegetation', 'traffic_light', 'traffic_sign',
        'barrier', 'traffic_cone',
    ]
    vals = [float(score_dict.get(k, 0.0)) for k in semantic_keys if k in score_dict]
    if not vals:
        return 0.0
    # CLIP 分数不是严格概率，这里作为弱权重。
    return float(np.clip(max(vals), 0.0, 1.0))



def _bev_grid_structure_strength(points: np.ndarray, grid_size: float = 0.55) -> np.ndarray:
    """基于 BEV 网格计算点云结构显著性。"""
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    if n == 0 or pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((n,), dtype=np.float32)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    x_min, x_max = -60.0, 85.0
    y_min, y_max = -85.0, 85.0
    nx = int(np.ceil((x_max - x_min) / grid_size))
    ny = int(np.ceil((y_max - y_min) / grid_size))

    ix = np.floor((x - x_min) / grid_size).astype(np.int64)
    iy = np.floor((y - y_min) / grid_size).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = iy * nx + ix

    count = np.zeros(nx * ny, dtype=np.float32)
    z_min_arr = np.full(nx * ny, np.inf, dtype=np.float32)
    z_max_arr = np.full(nx * ny, -np.inf, dtype=np.float32)

    flat_v = flat[valid]
    z_v = z[valid]
    np.add.at(count, flat_v, 1.0)
    np.minimum.at(z_min_arr, flat_v, z_v)
    np.maximum.at(z_max_arr, flat_v, z_v)

    z_range = z_max_arr - z_min_arr
    z_range[~np.isfinite(z_range)] = 0.0
    z_range = np.clip(z_range, 0.0, 4.0)

    count_grid = count.reshape(ny, nx)
    zrange_grid = z_range.reshape(ny, nx)

    density_norm_grid = _robust_normalize(np.log1p(count_grid).reshape(-1), low_pct=5.0, high_pct=98.0).reshape(ny, nx)
    height_norm_grid = _robust_normalize(zrange_grid.reshape(-1), low_pct=5.0, high_pct=98.0).reshape(ny, nx)

    # BEV 边缘强度：密度变化 + 高度变化。结构边界会被突出。
    dgy, dgx = np.gradient(density_norm_grid)
    hgy, hgx = np.gradient(height_norm_grid)
    edge_grid = np.sqrt(dgx * dgx + dgy * dgy) + 0.7 * np.sqrt(hgx * hgx + hgy * hgy)
    edge_grid = _robust_normalize(edge_grid.reshape(-1), low_pct=5.0, high_pct=99.0).reshape(ny, nx)

    cell_strength = 0.42 * edge_grid + 0.34 * height_norm_grid + 0.24 * density_norm_grid
    cell_strength = np.clip(cell_strength, 0.0, 1.0)

    out = np.zeros(n, dtype=np.float32)
    out[valid] = cell_strength[iy[valid], ix[valid]]
    return out


def _gaussian_kernel(size: int = 5, sigma: float = 1.1) -> np.ndarray:
    """生成二维高斯核，不依赖 scipy。"""
    size = int(size)
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1
    r = size // 2
    ax = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) * float(sigma) + 1e-8))
    k = k / (np.sum(k) + 1e-8)
    return k.astype(np.float32)


def _masked_smooth_grid(grid: np.ndarray, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """只在有点的网格邻域内做归一化平滑，避免空白区域稀释结构。"""
    grid = np.asarray(grid, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    h, w = grid.shape
    gpad = np.pad(grid * mask, ((ph, ph), (pw, pw)), mode='constant')
    mpad = np.pad(mask, ((ph, ph), (pw, pw)), mode='constant')
    out = np.zeros_like(grid, dtype=np.float32)
    weight = np.zeros_like(grid, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            kv = float(kernel[i, j])
            out += kv * gpad[i:i+h, j:j+w]
            weight += kv * mpad[i:i+h, j:j+w]
    out = out / (weight + 1e-8)
    out[mask <= 0] = 0.0
    return out.astype(np.float32)


def _structured_bev_display_strength(
    points: np.ndarray,
    base_strength: np.ndarray,
    image_response: np.ndarray | None = None,
    grid_size: float = 0.55,
    topk: int = 4,
    smooth_kernel: int = 5,
    smooth_sigma: float = 1.1,
    sparse_density_thr: float = 4.5,
) -> np.ndarray:
    """将逐点融合响应整理成更适合论文展示的 BEV 特征强度。"""
    pts = np.asarray(points, dtype=np.float32)
    base = np.asarray(base_strength, dtype=np.float32).reshape(-1)
    n = len(pts)
    if n == 0 or len(base) != n or pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((n,), dtype=np.float32)

    img = np.zeros(n, dtype=np.float32) if image_response is None else np.asarray(image_response, dtype=np.float32).reshape(-1)
    if len(img) != n:
        img = np.zeros(n, dtype=np.float32)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    x_min, x_max = -60.0, 85.0
    y_min, y_max = -85.0, 85.0
    nx = int(np.ceil((x_max - x_min) / grid_size))
    ny = int(np.ceil((y_max - y_min) / grid_size))
    ix = np.floor((x - x_min) / grid_size).astype(np.int64)
    iy = np.floor((y - y_min) / grid_size).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & np.isfinite(base)

    out = np.zeros(n, dtype=np.float32)
    if not valid.any():
        return out

    # 收集每个网格的点，用 top-k 而不是 mean，避免小目标响应被周围低值点稀释。
    flat = (iy[valid] * nx + ix[valid]).astype(np.int64)
    valid_indices = np.where(valid)[0]
    cell_values: dict[int, list[float]] = {}
    cell_z: dict[int, list[float]] = {}
    cell_img: dict[int, list[float]] = {}
    for pidx, cidx in zip(valid_indices, flat):
        key = int(cidx)
        cell_values.setdefault(key, []).append(float(base[pidx]))
        cell_z.setdefault(key, []).append(float(z[pidx]))
        cell_img.setdefault(key, []).append(float(img[pidx]))

    bg_grid = np.zeros((ny, nx), dtype=np.float32)
    count_grid = np.zeros((ny, nx), dtype=np.float32)
    relief_grid = np.zeros((ny, nx), dtype=np.float32)
    img_grid = np.zeros((ny, nx), dtype=np.float32)
    occ_grid = np.zeros((ny, nx), dtype=np.float32)

    for key, vals in cell_values.items():
        r = key // nx
        c = key % nx
        vals_arr = np.asarray(vals, dtype=np.float32)
        k = min(int(topk), len(vals_arr))
        top_vals = np.sort(vals_arr)[-k:]
        bg_grid[r, c] = float(np.mean(top_vals))
        count_grid[r, c] = float(len(vals_arr))
        zz = np.asarray(cell_z[key], dtype=np.float32)
        relief_grid[r, c] = float(zz.max() - zz.min()) if len(zz) > 1 else 0.0
        ii = np.asarray(cell_img[key], dtype=np.float32)
        img_grid[r, c] = float(ii.max()) if len(ii) > 0 else 0.0
        occ_grid[r, c] = 1.0

    # 背景结构层只轻平滑，保持结构连续但不过度模糊。
    kernel = _gaussian_kernel(size=smooth_kernel, sigma=smooth_sigma)
    bg_smooth = _masked_smooth_grid(bg_grid, occ_grid, kernel)

    occ = occ_grid > 0
    relief_norm = np.zeros_like(relief_grid, dtype=np.float32)
    img_norm = np.zeros_like(img_grid, dtype=np.float32)
    if occ.any():
        relief_norm[occ] = _robust_normalize(relief_grid[occ], low_pct=3.0, high_pct=98.0)
        img_norm[occ] = _robust_normalize(img_grid[occ], low_pct=5.0, high_pct=99.0)

    # 回填到点
    v_ix = ix[valid]
    v_iy = iy[valid]
    cell_bg = bg_smooth[v_iy, v_ix]
    cell_count = count_grid[v_iy, v_ix]
    cell_relief = relief_norm[v_iy, v_ix]
    cell_img = img_norm[v_iy, v_ix]
    base_v = base[valid]
    img_v = img[valid]

    # 稀疏小目标保护：点数少但局部高度/图像响应明显时，保留点级高响应。
    sparse_factor = np.clip((sparse_density_thr - cell_count) / (sparse_density_thr + 1e-8), 0.0, 1.0)
    sparse_saliency = 0.50 * base_v + 0.25 * np.maximum(img_v, cell_img) + 0.25 * cell_relief
    sparse_protect = sparse_saliency + 0.22 * sparse_factor * np.maximum(sparse_saliency - 0.38, 0.0)

    # 结构背景层 + 点级细节层；最终取 max，防止小目标被平滑层抹掉。
    mixed = 0.62 * cell_bg + 0.38 * base_v
    final_v = np.maximum(mixed, sparse_protect)
    final_v = np.clip(final_v, 0.0, 1.0)

    out[valid] = final_v
    # 做一次温和归一化和压暗低响应，减少全图彩色噪声。
    out = _robust_normalize(out, low_pct=2.0, high_pct=99.2)
    out = np.power(np.clip(out, 0.0, 1.0), 1.10).astype(np.float32)
    return out


def compute_fused_bev_strength(frame, projections, camera_scores: Dict[str, Dict[str, float]] | None = None) -> np.ndarray:
    """计算点级融合特征强度。"""
    pts = np.asarray(frame.lidar_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
        return np.zeros((0,), dtype=np.float32)

    n = len(pts)
    xyz = pts[:, :3]
    planar = np.linalg.norm(xyz[:, :2], axis=1)

    # 1) 点云结构显著性：用于突出边界、立面和局部几何变化。
    structure = _bev_grid_structure_strength(xyz, grid_size=0.55)

    # 2) 反射强度只是弱辅助，不能主导融合图。
    if pts.shape[1] > 3:
        intensity = _robust_normalize(np.clip(pts[:, 3], 0.0, None), low_pct=3.0, high_pct=98.5)
    else:
        intensity = np.zeros(n, dtype=np.float32)

    # 3) 图像纹理/边缘响应：多相机取最大可信响应，不累加可见次数。
    image_response = np.zeros(n, dtype=np.float32)
    visible_count = np.zeros(n, dtype=np.float32)
    camera_scores = camera_scores or {}
    edge_maps: Dict[str, np.ndarray] = {}

    if isinstance(projections, dict):
        for cam, proj in projections.items():
            idx = getattr(proj, 'valid_indices', None)
            pix = getattr(proj, 'pixel_coords', None)
            if idx is None or pix is None or cam not in getattr(frame, 'images', {}):
                continue

            idx = np.asarray(idx, dtype=np.int64).reshape(-1)
            pix = np.asarray(pix, dtype=np.float32).reshape(-1, 2)
            m = min(len(idx), len(pix))
            idx = idx[:m]
            pix = pix[:m]
            valid = (idx >= 0) & (idx < n)
            if not valid.any():
                continue
            idx = idx[valid]
            pix = pix[valid]

            if cam not in edge_maps:
                edge_maps[cam] = _image_edge_map(frame.images[cam])
            edge = _sample_image_edge(edge_maps[cam], pix)
            center_rel = _camera_center_reliability(frame.images[cam], pix, margin_ratio=0.12)
            sem_rel = _semantic_reliability(camera_scores.get(cam, {}))

            # 语义只做弱增强，并且必须依附图像纹理/边界响应。
            response = edge * center_rel * (0.88 + 0.12 * sem_rel)
            np.maximum.at(image_response, idx, response.astype(np.float32))
            np.add.at(visible_count, idx, 1.0)

    image_response = _robust_normalize(image_response, low_pct=5.0, high_pct=99.0)

    # 4) 距离可靠性只做弱辅助，避免近处天然高亮。
    distance_reliability = np.clip(1.0 - planar / 95.0, 0.0, 1.0).astype(np.float32)

    # 5) 重叠区域抑制：多相机重叠不加分，轻微降权。
    overlap_penalty = np.ones(n, dtype=np.float32)
    multi = visible_count > 1.0
    overlap_penalty[multi] = 1.0 / np.sqrt(visible_count[multi])
    overlap_penalty = np.clip(0.80 + 0.20 * overlap_penalty, 0.80, 1.0)

    # 6) 自车附近扫描环抑制。
    ego_suppression = np.ones(n, dtype=np.float32)
    ego_suppression[planar < 3.0] = 0.28
    ring = (planar >= 3.0) & (planar < 9.0)
    ego_suppression[ring] = 0.45 + 0.55 * ((planar[ring] - 3.0) / 6.0)

    # 基础逐点融合强度：结构和图像为主，反射/距离为弱辅助。
    base_strength = (
        0.50 * structure.astype(np.float32)
        + 0.34 * image_response.astype(np.float32)
        + 0.10 * intensity.astype(np.float32)
        + 0.06 * distance_reliability.astype(np.float32)
    )
    base_strength = base_strength * overlap_penalty * ego_suppression
    base_strength = _robust_normalize(base_strength, low_pct=2.0, high_pct=98.5)

    # 双层 BEV 显示：结构层平滑 + 小目标点级保留。
    display_strength = _structured_bev_display_strength(
        points=xyz,
        base_strength=base_strength,
        image_response=image_response,
        grid_size=0.55,
        topk=4,
        smooth_kernel=5,
        smooth_sigma=1.1,
        sparse_density_thr=4.5,
    )
    return np.clip(display_strength, 0.0, 1.0).astype(np.float32)

def save_bev(
    points: np.ndarray,
    path: str | Path,
    title: str = 'LiDAR BEV',
    strength: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    point_size: float = 1.0,
    dpi: int = 200,
):
    """保存 BEV 图。 """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError('points should have shape [N, >=3]')

    if strength is None:
        xyz = pts[:, :3]
        distance = np.linalg.norm(xyz[:, :2], axis=1).astype(np.float32)
        if title in ['LiDAR BEV', 'Raw LiDAR BEV']:
            title = 'Raw LiDAR top-down view'
        return _draw_paper_bev(
            points=pts,
            values=distance,
            path=path,
            title=title,
            cbar_label='distance / m',
            value_range=None,
            xlim=xlim,
            ylim=ylim,
            point_size=point_size,
            dpi=dpi,
        )

    strength = np.asarray(strength, dtype=np.float32).reshape(-1)
    finite = np.isfinite(strength)
    if finite.any() and float(np.nanmin(strength)) >= 0.0 and float(np.nanmax(strength)) <= 1.0001:
        # compute_fused_bev_strength 已经输出 0~1，避免二次鲁棒归一化破坏图中色彩含义。
        strength = np.clip(strength, 0.0, 1.0)
    else:
        strength = _robust_normalize(strength, low_pct=1.0, high_pct=99.0)
    if title in ['Fusion Feature BEV', 'Fused LiDAR BEV', 'LiDAR BEV']:
        title = 'Fused LiDAR feature strength'
    return _draw_paper_bev(
        points=pts,
        values=strength,
        path=path,
        title=title,
        cbar_label='feature strength',
        value_range=(0.0, 1.0),
        xlim=xlim,
        ylim=ylim,
        point_size=point_size,
        dpi=dpi,
    )




def _projection_stats(proj) -> dict:
    """统计单相机有效投影点数量与深度范围。"""
    if proj is None:
        return {
            'valid_points': 0,
            'drawn_points': 0,
            'mean_depth': 0.0,
            'min_depth': 0.0,
            'max_depth': 0.0,
        }
    dep = getattr(proj, 'depth_values', None)
    if dep is None and isinstance(proj, dict):
        dep = proj.get('depth_values', proj.get('depth', []))
    dep = np.asarray(dep, dtype=np.float32).reshape(-1)
    dep = dep[np.isfinite(dep) & (dep > 0)]
    if len(dep) == 0:
        return {
            'valid_points': 0,
            'drawn_points': 0,
            'mean_depth': 0.0,
            'min_depth': 0.0,
            'max_depth': 0.0,
        }
    return {
        'valid_points': int(len(dep)),
        'drawn_points': int(len(dep)),
        'mean_depth': float(np.mean(dep)),
        'min_depth': float(np.min(dep)),
        'max_depth': float(np.max(dep)),
    }


def _depth_color(depth_value: float, dmin: float, dmax: float) -> tuple[int, int, int]:
    """按深度生成颜色。近处偏红/黄，远处偏蓝，便于观察投影对齐。"""
    t = float((depth_value - dmin) / max(dmax - dmin, 1e-6))
    t = float(np.clip(t, 0.0, 1.0))
    r = int(255 * (1.0 - t))
    g = int(180 * (1.0 - abs(t - 0.5) * 2.0))
    b = int(255 * t)
    return r, g, b


def draw_projection_on_image_with_stats(
    img: Image.Image,
    proj,
    max_points: int = 2500,
    point_radius: int = 2,
) -> tuple[Image.Image, dict]:
    """在相机图像上叠加 LiDAR 投影点，并返回有效投影点统计。"""
    out = img.convert('RGB').copy() if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img)).convert('RGB')
    draw = ImageDraw.Draw(out)

    pts = getattr(proj, 'pixel_coords', None)
    dep = getattr(proj, 'depth_values', None)
    if pts is None and isinstance(proj, dict):
        pts = proj.get('pixel_coords', proj.get('pixels', None))
    if dep is None and isinstance(proj, dict):
        dep = proj.get('depth_values', proj.get('depth', None))

    if pts is None or dep is None:
        return out, _projection_stats(None)

    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    dep = np.asarray(dep, dtype=np.float32).reshape(-1)
    n = min(len(pts), len(dep))
    pts = pts[:n]
    dep = dep[:n]
    valid = np.isfinite(dep) & (dep > 0)
    pts = pts[valid]
    dep = dep[valid]

    stats = _projection_stats({'depth_values': dep})
    if len(pts) == 0:
        return out, stats

    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts_draw = pts[idx]
        dep_draw = dep[idx]
    else:
        pts_draw = pts
        dep_draw = dep

    stats['drawn_points'] = int(len(pts_draw))
    dmin = float(np.percentile(dep_draw, 5))
    dmax = float(np.percentile(dep_draw, 95))
    w, h = out.size
    for (u, v), d in zip(pts_draw, dep_draw):
        u = int(round(float(u)))
        v = int(round(float(v)))
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
        color = _depth_color(float(d), dmin, dmax)
        r = int(point_radius)
        draw.ellipse((u - r, v - r, u + r, v + r), fill=color, outline=color)

    return out, stats


def save_six_camera_projected_with_stats(
    frame,
    projections,
    image_path: str | Path,
    stats_path: str | Path | None = None,
    cell_size: tuple[int, int] = (520, 292),
    max_points_per_camera: int = 2500,
    keep_aspect: bool = False,
):
    """保存六相机 LiDAR 投影叠加图，并保存有效投影统计 CSV。"""
    images_by_cam: Dict[str, Image.Image] = {}
    stats_rows = []

    for cam in paper_camera_order():
        proj = projections[cam] if isinstance(projections, dict) and cam in projections else None
        im, stats = draw_projection_on_image_with_stats(
            frame.images[cam],
            proj,
            max_points=max_points_per_camera,
            point_radius=2,
        )
        im = _fit_image_to_cell(im, cell_size, keep_aspect=keep_aspect)

        # 标签信息
        draw = ImageDraw.Draw(im)
        title = f'{camera_to_zh_direction(cam)}  {cam}  valid={stats["valid_points"]}'
        try:
            bbox = draw.textbbox((0, 0), title)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(title) * 8, 14
        bg_w = min(im.size[0], max(160, tw + 14))
        bg_h = min(im.size[1], max(26, th + 10))
        draw.rectangle((0, 0, bg_w, bg_h), fill=(0, 0, 0))
        draw.text((6, max(4, (bg_h - th) // 2 - 1)), title, fill=(255, 255, 255))
        images_by_cam[cam] = im

        stats_rows.append({
            'camera_name': cam,
            'direction': camera_to_zh_direction(cam),
            'valid_points': stats['valid_points'],
            'drawn_points': stats['drawn_points'],
            'mean_depth': stats['mean_depth'],
            'min_depth': stats['min_depth'],
            'max_depth': stats['max_depth'],
        })

    canvas = _make_camera_grid(images_by_cam, cell_size)

    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(image_path)

    if stats_path is not None:
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with stats_path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'camera_name',
                    'direction',
                    'valid_points',
                    'drawn_points',
                    'mean_depth',
                    'min_depth',
                    'max_depth',
                ],
            )
            writer.writeheader()
            writer.writerows(stats_rows)

    return image_path, stats_rows

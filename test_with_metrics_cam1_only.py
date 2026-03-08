import os
import re
import json
import math
import csv
import itertools
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ArgParser import Argument
from dataset.dataset import *
from model.Metasurface import Metasurface
from utils.Camera import FisheyeCam, PanoramaCam
from Image_formation.renderer import ActiveStereoRenderer
from model.StereoMatching import DepthEstimator
from model.e2e import E2E


# ----------------------------
# basic helpers
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    name = str(name)
    name = os.path.basename(name)
    name = re.sub(r"\.(png|jpg|jpeg|npy|exr)$", "", name, flags=re.IGNORECASE)
    name = name.replace(" ", "_")
    return name


def save_uint8_png(path: Path, img_uint8: np.ndarray):
    ensure_dir(path.parent)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(path), img_uint8)
    except Exception:
        from PIL import Image
        Image.fromarray(img_uint8).save(str(path))


def save_uint16_png(path: Path, img_uint16: np.ndarray):
    ensure_dir(path.parent)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(path), img_uint16)
    except Exception:
        from PIL import Image
        Image.fromarray(img_uint16, mode="I;16").save(str(path))


def save_colormap_png(path: Path, img_float01: np.ndarray, cmap: str = "inferno"):
    ensure_dir(path.parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(str(path), img_float01, cmap=cmap)


def to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.ndim == 2:
        return x[None, None]
    if x.ndim == 3:
        # BxHxW
        return x[:, None]
    if x.ndim == 4:
        # maybe BxHxWxC
        if x.shape[-1] == 3 and x.shape[1] != 3:
            return x.permute(0, 3, 1, 2).contiguous()
        return x
    raise ValueError(f"Unsupported shape: {tuple(x.shape)}")


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if mask.shape[1] == 1 and x.shape[1] != 1:
        mask = mask.repeat(1, x.shape[1], 1, 1)
    s = (x * mask).sum(dim=(1, 2, 3))
    d = mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    return s / d


def _safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def depth_grad_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred, gt: (B,1,H,W) ; mask: (B,1,H,W)
    gradient L1 on depth differences (multi-shift), only where both pixels valid.
    """

    def one_shift(shift: int) -> torch.Tensor:
        p1 = pred[:, :, :, shift:]
        p0 = pred[:, :, :, :-shift]
        g1 = gt[:, :, :, shift:]
        g0 = gt[:, :, :, :-shift]
        m = mask[:, :, :, shift:] * mask[:, :, :, :-shift]
        gx = (p1 - p0) - (g1 - g0)

        p1y = pred[:, :, shift:, :]
        p0y = pred[:, :, :-shift, :]
        g1y = gt[:, :, shift:, :]
        g0y = gt[:, :, :-shift, :]
        my = mask[:, :, shift:, :] * mask[:, :, :-shift, :]
        gy = (p1y - p0y) - (g1y - g0y)

        loss_x = masked_mean(gx.abs(), m).mean()
        loss_y = masked_mean(gy.abs(), my).mean()
        return loss_x + loss_y

    return (one_shift(1) + one_shift(2) + one_shift(3)) / 3.0


@torch.no_grad()
def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    valid = mask * (gt > 0).float() * (pred > 0).float()
    if valid.sum().item() < 10:
        return {
            k: float("nan")
            for k in ["absrel", "sqrel", "mse", "rmse", "rmse_log", "log10", "silog", "delta1", "delta2", "delta3", "mae", "grad_l1"]
        }

    diff = (pred - gt).abs()
    mae = masked_mean(diff, valid).mean().item()

    absrel = masked_mean(diff / gt.clamp_min(1e-6), valid).mean().item()
    sqrel = masked_mean((pred - gt) ** 2 / gt.clamp_min(1e-6), valid).mean().item()

    mse = masked_mean((pred - gt) ** 2, valid).mean().item()
    rmse = math.sqrt(max(mse, 0.0))

    gradl1 = float(depth_grad_l1(pred, gt, valid).item())

    log_diff = _safe_log(pred) - _safe_log(gt)
    rmse_log = torch.sqrt(masked_mean(log_diff ** 2, valid)).mean().item()

    log10 = masked_mean((torch.log10(pred.clamp_min(1e-6)) - torch.log10(gt.clamp_min(1e-6))).abs(), valid).mean().item()

    d = log_diff
    Ed = masked_mean(d, valid).mean()
    Ed2 = masked_mean(d ** 2, valid).mean()
    silog = torch.sqrt((Ed2 - Ed * Ed).clamp_min(0.0)).item() * 100.0

    ratio = torch.maximum(pred / gt.clamp_min(1e-6), gt / pred.clamp_min(1e-6))
    delta1 = masked_mean((ratio < 1.25).float(), valid).mean().item()
    delta2 = masked_mean((ratio < (1.25 ** 2)).float(), valid).mean().item()
    delta3 = masked_mean((ratio < (1.25 ** 3)).float(), valid).mean().item()

    return {
        "absrel": absrel,
        "sqrel": sqrel,
        "mse": mse,
        "rmse": rmse,
        "grad_l1": gradl1,
        "rmse_log": rmse_log,
        "log10": log10,
        "silog": silog,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "mae": mae,
    }


def to_depth_m_from_gt(depth_gt: torch.Tensor, depth_scale_m: float, gt_unit: str = "auto") -> torch.Tensor:
    d = depth_gt
    if gt_unit == "auto":
        mx = float(d.max().item())
        gt_unit = "normalized" if mx <= 1.5 else "meters"
    if gt_unit == "normalized":
        return d * depth_scale_m
    if gt_unit == "meters":
        return d
    raise ValueError(f"Unknown gt_unit: {gt_unit}")


def to_depth_m_from_inv_pred(inv_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 1.0 / inv_pred.clamp_min(eps)


def align_depth_scale(pred_m: torch.Tensor, gt_m: torch.Tensor, mask: torch.Tensor, mode: str = "median") -> torch.Tensor:
    if mode == "none":
        return pred_m
    valid = (mask > 0.5) & (pred_m > 1e-6) & (gt_m > 1e-6)
    if valid.sum().item() < 10:
        return pred_m
    p = pred_m[valid].detach()
    g = gt_m[valid].detach()

    if mode == "median":
        s = torch.median(g) / torch.median(p).clamp_min(1e-8)
        return pred_m * s
    if mode == "ls":
        s = (p * g).sum() / (p * p).sum().clamp_min(1e-8)
        return pred_m * s
    raise ValueError(f"Unknown align mode: {mode}")


# ----------------------------
# Normal from depth (strict with camera rays)
# ----------------------------
def depth_to_normal_from_cam(depth_m: torch.Tensor, cam, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    depth_m: (1,1,H,W) range (meters)
    cam: must have get_whole_pts() -> (3, H*W) unit rays
    valid_mask: (1,1,H,W) 0/1

    returns:
      n_unit: (1,3,H,W)
      n_valid: (1,1,H,W)
    """
    B, _, H, W = depth_m.shape
    assert B == 1, "当前实现默认 batch=1（你测试就是1）"

    rays = cam.get_whole_pts()  # 3 x (H*W)
    rays = rays.reshape(3, H, W).to(depth_m.device).float()  # 3xHxW
    P = depth_m[0, 0] * rays  # 3xHxW
    P = P[None, ...]  # 1x3xHxW

    v = (valid_mask > 0.5).float()
    v_l = F.pad(v[:, :, :, :-1], (1, 0, 0, 0))
    v_r = F.pad(v[:, :, :, 1:], (0, 1, 0, 0))
    v_u = F.pad(v[:, :, :-1, :], (0, 0, 1, 0))
    v_d = F.pad(v[:, :, 1:, :], (0, 0, 0, 1))
    n_valid = v * v_l * v_r * v_u * v_d

    P_l = F.pad(P[:, :, :, :-1], (1, 0, 0, 0))
    P_r = F.pad(P[:, :, :, 1:], (0, 1, 0, 0))
    P_u = F.pad(P[:, :, :-1, :], (0, 0, 1, 0))
    P_d = F.pad(P[:, :, 1:, :], (0, 0, 0, 1))

    Px = (P_r - P_l) * n_valid
    Py = (P_d - P_u) * n_valid

    n = torch.cross(Px, Py, dim=1)
    n_norm = torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(1e-8)
    n_unit = n / n_norm
    n_unit = n_unit * n_valid.repeat(1, 3, 1, 1)
    return n_unit, n_valid


def normalize_gt_normal(n: torch.Tensor, valid_thr: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    n = to_nchw(n).float()
    if n.shape[1] != 3:
        n = n.repeat(1, 3, 1, 1)

    nmin, nmax = n.min().item(), n.max().item()
    if nmax > 1.5:
        n = n / 127.5 - 1.0
    elif nmin >= 0.0 and nmax <= 1.0:
        n = n * 2.0 - 1.0

    norm = torch.linalg.norm(n, dim=1, keepdim=True)
    valid = (norm > valid_thr).float()
    n_unit = n / norm.clamp_min(1e-8)
    n_unit = n_unit * valid.repeat(1, 3, 1, 1)
    return n_unit, valid


def find_best_normal_transform(pred_n: torch.Tensor, gt_n: torch.Tensor, valid_mask: torch.Tensor):
    """
    Search over axis permutation (6) x sign flips (8) => 48 transforms.
    """
    best = ((0, 1, 2), (1.0, 1.0, 1.0))
    best_dot = -1e9
    if (valid_mask > 0.5).sum().item() < 10:
        return best

    for perm in itertools.permutations([0, 1, 2], 3):
        p = pred_n[:, perm, :, :]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            s = torch.tensor(signs, device=pred_n.device, dtype=pred_n.dtype)[None, :, None, None]
            pp = p * s
            dot = (pp * gt_n).sum(dim=1, keepdim=True).clamp(-1, 1)
            mdot = masked_mean(dot, valid_mask).mean().item()
            if mdot > best_dot:
                best_dot = mdot
                best = (perm, signs)
    return best


def apply_normal_transform(n: torch.Tensor, perm, signs):
    n2 = n[:, perm, :, :]
    s = torch.tensor(signs, device=n.device, dtype=n.dtype)[None, :, None, None]
    return n2 * s


@torch.no_grad()
def normal_metrics(pred_n: torch.Tensor, gt_n: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    if valid.sum().item() < 10:
        return {k: float("nan") for k in ["ang_mean", "ang_median", "ang_rmse", "acc_11", "acc_22", "acc_30", "cos_loss"]}

    dot = (pred_n * gt_n).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    ang = torch.acos(dot) * (180.0 / math.pi)
    ang_mean = masked_mean(ang, valid).mean().item()

    ang_np = ang[valid > 0.5].detach().cpu().numpy()
    ang_median = float(np.median(ang_np)) if ang_np.size else float("nan")
    ang_rmse = float(np.sqrt(np.mean(ang_np ** 2))) if ang_np.size else float("nan")

    acc_11 = float((ang_np < 11.25).mean()) if ang_np.size else float("nan")
    acc_22 = float((ang_np < 22.5).mean()) if ang_np.size else float("nan")
    acc_30 = float((ang_np < 30.0).mean()) if ang_np.size else float("nan")

    cos_loss = masked_mean(1.0 - dot, valid).mean().item()

    return {
        "ang_mean": ang_mean,
        "ang_median": ang_median,
        "ang_rmse": ang_rmse,
        "acc_11": acc_11,
        "acc_22": acc_22,
        "acc_30": acc_30,
        "cos_loss": cos_loss,
    }


# ----------------------------
# Occlusion (no cam2 data needed): forward splat + z-buffer
# ----------------------------
def normalize_gt_occ(o: torch.Tensor) -> torch.Tensor:
    o = to_nchw(o).float()
    if o.shape[1] != 1:
        o = o[:, :1]
    if o.max().item() > 1.5:
        o = (o > 127.0).float()
    else:
        o = (o > 0.5).float()
    return o


def bin_stats(pred_bool: torch.Tensor, gt_bool: torch.Tensor) -> Dict[str, float]:
    tp = (pred_bool & gt_bool).sum().item()
    fp = (pred_bool & (~gt_bool)).sum().item()
    fn = ((~pred_bool) & gt_bool).sum().item()
    tn = ((~pred_bool) & (~gt_bool)).sum().item()

    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1e-8)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "acc": acc}


@torch.no_grad()
def bce_loss(pred_prob: torch.Tensor, gt_prob: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> float:
    pred = pred_prob.clamp(eps, 1 - eps)
    gt = gt_prob
    loss = -(gt * torch.log(pred) + (1 - gt) * torch.log(1 - pred))
    return masked_mean(loss, mask).mean().item()


def relative_pose_cam1_to_cam2(cam1, cam2, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Your renderer uses: world = T_wc @ cam_point (so get_extrinsic is camera->world).
    Then: T_c2_c1 = inv(T_wc2) @ T_wc1.
    """
    T1 = cam1.get_extrinsic()
    T2 = cam2.get_extrinsic()
    if not torch.is_tensor(T1):
        T1 = torch.as_tensor(T1)
    if not torch.is_tensor(T2):
        T2 = torch.as_tensor(T2)
    T1 = T1.to(device=device, dtype=torch.float32)
    T2 = T2.to(device=device, dtype=torch.float32)
    T2_inv = torch.linalg.inv(T2)
    T_21 = T2_inv @ T1  # cam1 -> cam2
    R = T_21[:3, :3].contiguous()
    t = T_21[:3, 3].contiguous()
    return R, t


@torch.no_grad()
def occlusion_cam1_seen_from_cam2_no_cam2_depth(
    depth_cam1_m: torch.Tensor,
    cam1,
    cam2,
    R_21: torch.Tensor,
    t_21: torch.Tensor,
    valid_mask_cam1: torch.Tensor,
    outside_as_occluded: bool = True,
    tau: float = 0.05,
    margin: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute occlusion for cam1 pixels: whether the 3D point is NOT visible from cam2.
    Uses only cam1 depth + relative pose, via forward splat z-buffer in cam2.

    Returns:
      occ_prob: (1,1,H,W)
      occ_bin : (1,1,H,W) 1=occluded
      eval_mask: (1,1,H,W) valid for metrics
    """
    B, _, H, W = depth_cam1_m.shape
    assert B == 1, "batch=1 expected"

    rays1 = cam1.get_whole_pts().reshape(3, H, W).to(depth_cam1_m.device).float()  # 3xHxW
    P1 = (depth_cam1_m[0, 0] * rays1).reshape(3, -1)  # 3xN

    P2 = (R_21 @ P1) + t_21.reshape(3, 1)  # 3xN in cam2 coords
    rng2 = torch.linalg.norm(P2, dim=0).clamp_min(1e-8)  # N

    pix = cam2.world2pixel(P2)  # 3xN: u,v,valid
    u = pix[0]
    v = pix[1]
    vproj = pix[2]  # 1 if within cam2 FOV and image bounds

    # discretize to pixels for raster/zbuffer
    ui = torch.round(u).long()
    vi = torch.round(v).long()
    ui = ui.clamp(0, W - 1)
    vi = vi.clamp(0, H - 1)
    idx = (vi * W + ui).long()  # N

    # only consider cam1-valid points for z-buffering
    m1 = (valid_mask_cam1[0, 0].reshape(-1) > 0.5)
    mproj = (vproj > 0.5)
    m = m1 & mproj

    # z-buffer: per target pixel minimal range
    min_rng = torch.full((H * W, ), float("inf"), device=depth_cam1_m.device, dtype=torch.float32)
    if m.any():
        min_rng.scatter_reduce_(0, idx[m], rng2[m], reduce="amin", include_self=True)

    # visibility test: visible if rng is the min at that projected pixel (within eps/margin)
    rng_min_at = min_rng[idx].clamp_min(1e-8)
    diff = rng2 - rng_min_at  # >=0
    visible = (diff <= margin)

    occ = (~visible)  # bool N, only meaningful when projected-valid

    # build occ map in cam1 image (source)
    occ_map = torch.zeros((H * W, ), device=depth_cam1_m.device, dtype=torch.float32)

    if outside_as_occluded:
        # if point cannot be projected into cam2 (no overlap), treat as "not visible"
        occ_map[m1 & (~mproj)] = 1.0

    occ_map[m] = occ[m].float()
    occ_map = occ_map.reshape(1, 1, H, W)

    # soft prob (for BCE)
    # for projected-valid pixels: use sigmoid on (diff - margin)/tau
    occ_prob = torch.zeros_like(occ_map)
    if m.any():
        p = torch.sigmoid((diff - margin) / max(tau, 1e-6)).float()
        occ_prob_flat = torch.zeros((H * W, ), device=depth_cam1_m.device, dtype=torch.float32)
        occ_prob_flat[m] = p[m]
        if outside_as_occluded:
            occ_prob_flat[m1 & (~mproj)] = 1.0
        occ_prob = occ_prob_flat.reshape(1, 1, H, W)

    eval_mask = valid_mask_cam1.clone()
    return occ_prob, occ_map, eval_mask


# ----------------------------
# checkpoint loader
# ----------------------------
def load_checkpoint_robust(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    model_is_dp = any(k.startswith("module.") for k in model.state_dict().keys())
    sd_keys = list(sd.keys())
    if len(sd_keys) > 0 and sd_keys[0].startswith("module.") and (not model_is_dp):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if len(sd_keys) > 0 and (not sd_keys[0].startswith("module.")) and model_is_dp:
        sd = {"module." + k: v for k, v in sd.items()}

    model.load_state_dict(sd, strict=False)
    model.to(device)
    return model


# ----------------------------
# main eval
# ----------------------------
@torch.no_grad()
def test_with_metrics(opt, model: nn.Module):
    device = torch.device(opt.device)
    dataset_test = CreateSyntheticDataset(opt.test_path, opt.split)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=getattr(opt, "num_workers", 4),
        pin_memory=True,
    )

    ckpt_path = Path(opt.ckpt_path)
    ckpt_dir = ckpt_path.parent
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    model = load_checkpoint_robust(model, str(ckpt_path), device)
    model.eval()

    depth_root = ckpt_dir / "Depth"
    normal_root = ckpt_dir / "Normal"
    occ_root = ckpt_dir / "Occlusion"
    for r in [depth_root, normal_root, occ_root]:
        ensure_dir(r)

    # fisheye mask: if missing, fallback to all-ones
    fisheye_mask = None
    if opt.fisheye_mask_path and os.path.exists(opt.fisheye_mask_path):
        fisheye_mask = torch.from_numpy(np.load(opt.fisheye_mask_path)).float()
    else:
        print("[WARN] fisheye_mask_path not found -> use all-ones mask at runtime")

    depth_scale_m = float(getattr(opt, "depth_scale_m", 10.0))
    gt_unit = getattr(opt, "gt_depth_unit", "auto")
    depth_align = getattr(opt, "depth_align", "median")  # default median for robustness
    min_depth = float(getattr(opt, "min_depth_m", 0.05))
    max_depth = float(getattr(opt, "max_depth_m", 1e6))

    occ_gt_sem = getattr(opt, "occ_gt_semantics", "auto")  # auto|visible|occluded

    # cameras from model (cam1/cam2 always exist in renderer, even if cam2 data empty)
    cam1 = model.renderer.cam_calib[0]
    cam2 = model.renderer.cam_calib[1]
    R_21, t_21 = relative_pose_cam1_to_cam2(cam1, cam2, device=device)
    print(f"[INFO] pose cam1->cam2 t={t_21.detach().cpu().numpy().tolist()}")

    normal_transform_set = False
    normal_perm, normal_signs = (0, 1, 2), (1.0, 1.0, 1.0)

    depth_rows: List[Dict] = []
    normal_rows: List[Dict] = []
    occ_rows: List[Dict] = []

    # create output dirs
    for cn in ["cam1", "cam2"]:
        for sub in ["pred", "gt", "error"]:
            ensure_dir(depth_root / cn / sub)
            ensure_dir(normal_root / cn / sub)
            ensure_dir(occ_root / cn / sub)

    for i, data in enumerate(dataloader_test):
        name = data.get("name", f"{i:06d}")
        if isinstance(name, (list, tuple)):
            name = name[0]
        name = sanitize_name(name)

        ref_im_list = data["ref_im_list"]
        depth_im_list = data["depth_im_list"]
        occ_im_list = data.get("occ_im_list", [])
        normal_im_list = data.get("normal_im_list", [])

        inv_depth_pred, aux = model(ref_im_list, depth_im_list, occ_im_list, normal_im_list)

        # make inv_preds list
        if isinstance(inv_depth_pred, (list, tuple)):
            inv_preds = list(inv_depth_pred)
        elif torch.is_tensor(inv_depth_pred):
            if inv_depth_pred.ndim == 4 and inv_depth_pred.shape[1] in (1, 2):
                inv_preds = [inv_depth_pred[:, c:c + 1] for c in range(inv_depth_pred.shape[1])]
            else:
                inv_preds = [inv_depth_pred]
        else:
            raise TypeError(f"Unsupported model output: {type(inv_depth_pred)}")

        if not isinstance(depth_im_list, (list, tuple)):
            depth_im_list = [depth_im_list]
        if not isinstance(occ_im_list, (list, tuple)):
            occ_im_list = [occ_im_list]
        if not isinstance(normal_im_list, (list, tuple)):
            normal_im_list = [normal_im_list]

        # what we actually can evaluate (GT exists)
        n_gt = len(depth_im_list)
        n_pred = len(inv_preds)
        n_eval = min(n_gt, n_pred)  # cam2 empty => n_gt likely 1

        if i == 0:
            print(
                f"[DEBUG] n_gt(depth)={n_gt}, n_pred={n_pred}, n_eval={n_eval}, len(occ_gt)={len(occ_im_list)}, len(normal_gt)={len(normal_im_list)}"
            )

        # build mask
        depth_gt0 = to_nchw(depth_im_list[0]).float()
        H, W = depth_gt0.shape[-2:]
        if fisheye_mask is None:
            mask_fish = torch.ones((1, 1, H, W), device=device, dtype=torch.float32)
        else:
            mask_fish = fisheye_mask.to(device=device, dtype=torch.float32)[None, None]
            if mask_fish.shape[-2:] != (H, W):
                mask_fish = F.interpolate(mask_fish, size=(H, W), mode="nearest")

        # evaluate each available GT camera (usually only cam1)
        for ci in range(n_eval):
            cn = "cam1" if ci == 0 else f"cam{ci+1}"

            depth_gt = to_nchw(depth_im_list[ci].to(device)).float()
            inv_pred = to_nchw(inv_preds[ci].to(device)).float()

            depth_gt_m = to_depth_m_from_gt(depth_gt, depth_scale_m, gt_unit=gt_unit)
            depth_pred_m = to_depth_m_from_inv_pred(inv_pred)

            valid = mask_fish * (depth_gt_m > min_depth).float() * (depth_gt_m < max_depth).float()

            # align scale (robust)
            depth_pred_m = align_depth_scale(depth_pred_m, depth_gt_m, valid, mode=depth_align)
            valid = valid * (depth_pred_m > min_depth).float()

            if (i < 3) or ((i + 1) % 20 == 0):
                with torch.no_grad():
                    v = (valid > 0.5) & torch.isfinite(depth_gt_m) & torch.isfinite(depth_pred_m)
                    n = int(v.sum().item())
                    if n > 0:
                        gt_vals = depth_gt_m[v]
                        pr_vals = depth_pred_m[v]

                        frac_pr_gt_100 = float((pr_vals > 100.0).float().mean().item())
                        frac_pr_gt_1000 = float((pr_vals > 1000.0).float().mean().item())
                        inv_min = float(inv_pred[v].min().item()) if "inv_pred" in locals() else float("nan")
                        print(
                            f"[DEPTH DBG] idx={i:05d} name={name} cam={cn} valid={n} | "
                            f"GT(m) min/med/max={gt_vals.min().item():.4f}/{gt_vals.median().item():.4f}/{gt_vals.max().item():.4f} | "
                            f"Pred(m) min/med/max={pr_vals.min().item():.4f}/{pr_vals.median().item():.4f}/{pr_vals.max().item():.4f} | "
                            f"Pred>100m={frac_pr_gt_100*100:.2f}% Pred>1000m={frac_pr_gt_1000*100:.2f}% | "
                            f"inv_min={inv_min:.6g}"
                        )
                    else:
                        print(f"[DEPTH DBG] idx={i:05d} name={name} cam={cn} valid=0")

            # depth metrics
            dmet = depth_metrics(depth_pred_m, depth_gt_m, valid)
            depth_rows.append({"name": name, "cam": cn, **dmet})

            # save depth viz (normalize by GT valid range per-image)
            gt_vals = depth_gt_m[valid > 0.5]
            dmn = gt_vals.min().item() if gt_vals.numel() else 0.0
            dmx = gt_vals.max().item() if gt_vals.numel() else 1.0
            denom = max(dmx - dmn, 1e-8)

            dp = depth_pred_m[0, 0].detach().cpu().numpy()
            dg = depth_gt_m[0, 0].detach().cpu().numpy()
            vm = valid[0, 0].detach().cpu().numpy().astype(np.float32)

            dp_n = np.clip((dp - dmn) / denom, 0.0, 1.0) * vm
            dg_n = np.clip((dg - dmn) / denom, 0.0, 1.0) * vm
            de_n = np.clip(np.abs(dp - dg) / denom, 0.0, 1.0) * vm

            save_colormap_png(depth_root / cn / "pred" / f"{name}.png", dp_n, cmap="Spectral_r")
            save_colormap_png(depth_root / cn / "gt" / f"{name}.png", dg_n, cmap="Spectral_r")
            save_colormap_png(depth_root / cn / "error" / f"{name}.png", de_n, cmap="magma")
            save_uint16_png(depth_root / cn / "pred" / f"{name}_raw16.png", (dp_n * 65535.0).astype(np.uint16))
            save_uint16_png(depth_root / cn / "gt" / f"{name}_raw16.png", (dg_n * 65535.0).astype(np.uint16))

            # normal metrics (cam1 only usually)
            if len(normal_im_list) > ci and normal_im_list[ci] is not None:
                # pred normal from predicted depth using camera rays (strict)
                cam = cam1 if ci == 0 else cam2
                n_pred, n_valid = depth_to_normal_from_cam(depth_pred_m, cam, valid)

                gt_n, gt_nv = normalize_gt_normal(normal_im_list[ci].to(device), valid_thr=opt.normal_valid_thr)
                gt_n = F.interpolate(gt_n, size=n_pred.shape[-2:], mode="bilinear", align_corners=False)
                gt_nv = F.interpolate(gt_nv, size=n_pred.shape[-2:], mode="nearest")

                if (not normal_transform_set) and ci == 0:
                    # set transform once using GT depth->normal vs GT normal
                    n_from_gt_depth, n_v2 = depth_to_normal_from_cam(depth_gt_m, cam1, valid)
                    v_fit = n_v2 * gt_nv
                    normal_perm, normal_signs = find_best_normal_transform(n_from_gt_depth, gt_n, v_fit)
                    normal_transform_set = True
                    print(f"[INFO] Normal transform: perm={normal_perm}, signs={normal_signs}")

                n_pred = apply_normal_transform(n_pred, normal_perm, normal_signs)

                n_eval_valid = n_valid * gt_nv
                nmet = normal_metrics(n_pred, gt_n, n_eval_valid)
                normal_rows.append({"name": name, "cam": cn, **nmet})

                # save normals
                n_pred_np = (n_pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
                n_gt_np = (gt_n[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
                save_uint8_png(normal_root / cn / "pred" / f"{name}.png", (np.clip(n_pred_np, 0, 1) * 255).astype(np.uint8))
                save_uint8_png(normal_root / cn / "gt" / f"{name}.png", (np.clip(n_gt_np, 0, 1) * 255).astype(np.uint8))

                # angular error map
                dot = (n_pred * gt_n).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
                ang = (torch.acos(dot) * (180.0 / math.pi)) * n_eval_valid
                ang_np = ang[0, 0].detach().cpu().numpy()
                ang_vis = np.clip(ang_np / 60.0, 0.0, 1.0)
                save_colormap_png(normal_root / cn / "error" / f"{name}_angular.png", ang_vis, cmap="viridis")

            # occlusion metrics: cam2 GT 缺失也照样算 (cam1->cam2 zbuffer, no cam2 depth)
            if ci == 0 and len(occ_im_list) > 0 and occ_im_list[0] is not None:
                gt_occ = normalize_gt_occ(occ_im_list[0].to(device))
                if gt_occ.shape[-2:] != (H, W):
                    gt_occ = F.interpolate(gt_occ, size=(H, W), mode="nearest")

                occ_prob, occ_bin, occ_mask = occlusion_cam1_seen_from_cam2_no_cam2_depth(
                    depth_cam1_m=depth_pred_m,
                    cam1=cam1,
                    cam2=cam2,
                    R_21=R_21,
                    t_21=t_21,
                    valid_mask_cam1=valid,
                    outside_as_occluded=opt.occ_outside_as_occluded,
                    tau=opt.occ_tau,
                    margin=opt.occ_margin,
                )

                # auto detect semantics on first sample if needed
                if occ_gt_sem == "auto" and i == 0:
                    # try both: GT=visible vs GT=occluded, pick higher IoU
                    gt_bool = (gt_occ > 0.5) & (occ_mask > 0.5)
                    pred_occ_bool = (occ_bin > 0.5) & (occ_mask > 0.5)
                    pred_vis_bool = ((1.0 - occ_bin) > 0.5) & (occ_mask > 0.5)
                    iou_occ = bin_stats(pred_occ_bool, gt_bool)["iou"]
                    iou_vis = bin_stats(pred_vis_bool, gt_bool)["iou"]
                    occ_gt_sem = "occluded" if iou_occ >= iou_vis else "visible"
                    print(f"[INFO] Occlusion GT semantics: {occ_gt_sem} (IoU occ={iou_occ:.3f}, vis={iou_vis:.3f})")

                if occ_gt_sem == "visible":
                    pred_prob = (1.0 - occ_prob) * occ_mask
                    pred_bin2 = (1.0 - occ_bin) * occ_mask
                else:
                    pred_prob = occ_prob * occ_mask
                    pred_bin2 = occ_bin * occ_mask

                gt_bool = (gt_occ > 0.5) & (occ_mask > 0.5)
                pred_bool = (pred_bin2 > 0.5) & (occ_mask > 0.5)

                st = bin_stats(pred_bool, gt_bool)
                loss_bce = bce_loss(pred_prob, gt_occ, occ_mask)

                occ_rows.append({"name": name, "cam": "cam1", **st, "bce": loss_bce})

                # save occ viz
                gt_np = (gt_occ[0, 0].detach().cpu().numpy()).astype(np.float32)
                pr_np = (pred_bin2[0, 0].detach().cpu().numpy()).astype(np.float32)
                mk_np = (occ_mask[0, 0].detach().cpu().numpy()).astype(np.float32)
                gt_np = gt_np * mk_np
                pr_np = pr_np * mk_np
                err_np = np.abs(pr_np - gt_np) * mk_np

                save_uint8_png(occ_root / "cam1" / "gt" / f"{name}.png", (gt_np * 255).astype(np.uint8))
                save_uint8_png(occ_root / "cam1" / "pred" / f"{name}.png", (pr_np * 255).astype(np.uint8))
                save_uint8_png(occ_root / "cam1" / "error" / f"{name}.png", (err_np * 255).astype(np.uint8))

        if (i + 1) % getattr(opt, "print_every", 20) == 0:
            last = depth_rows[-1]
            print(
                f"[{i+1}/{len(dataloader_test)}] {name} | absrel={last['absrel']:.4f} rmse={last['rmse']:.3f} delta1={last['delta1']:.3f}"
            )

    def write_csv(path: Path, rows: List[Dict]):
        if not rows:
            return
        ensure_dir(path.parent)
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(ckpt_dir / "metrics_depth.csv", depth_rows)
    write_csv(ckpt_dir / "metrics_normal.csv", normal_rows)
    write_csv(ckpt_dir / "metrics_occlusion.csv", occ_rows)

    def summarize(rows: List[Dict]) -> Dict[str, float]:
        if not rows:
            return {}
        keys = [k for k in rows[0].keys() if k not in ("name", "cam")]
        out = {}
        for k in keys:
            vals = [r[k] for r in rows if isinstance(r.get(k, None), (int, float)) and not (isinstance(r[k], float) and math.isnan(r[k]))]
            out[k] = float(np.mean(vals)) if vals else float("nan")
        return out

    summary = {
        "avg_depth": summarize(depth_rows),
        "avg_normal": summarize(normal_rows),
        "avg_occlusion": summarize(occ_rows),
        "count_depth_rows": len(depth_rows),
        "count_normal_rows": len(normal_rows),
        "count_occ_rows": len(occ_rows),
        "depth_scale_m": depth_scale_m,
        "gt_depth_unit": gt_unit,
        "depth_align": depth_align,
        "occ_gt_semantics": occ_gt_sem,
        "pose_cam1_to_cam2": {
            "R": R_21.detach().cpu().numpy().tolist(),
            "t": t_21.detach().cpu().numpy().tolist()
        },
        "normal_transform": {
            "perm": list(normal_perm),
            "signs": list(normal_signs)
        } if normal_transform_set else None,
    }
    with open(ckpt_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] Saved metrics into ckpt dir.")


# ----------------------------
# Build your model exactly as before
# ----------------------------
def build_model(args):
    device = torch.device(args.device)
    metasurface = Metasurface(args, device)

    ckpt_path = Path(args.ckpt_path)
    phase_name = "phase_" + ckpt_path.name.replace(".pth", ".npy")
    phase_path = ckpt_path.parent / phase_name

    if phase_path.exists():
        print(f"[INFO] Loading Metasurface Phase from: {phase_path}")
        optimized_phase = np.load(str(phase_path))
        metasurface.update_phase(torch.from_numpy(optimized_phase).float().to(device))
    else:
        print(f"[WARN] Phase file not found: {phase_path}. Results may be incorrect!")

    radian_90 = math.radians(90)
    cam1 = FisheyeCam(args, (0.05, 0.05, 0), (radian_90, 0, 0), "cam1", device, args.cam_config_path)
    cam2 = FisheyeCam(args, (-0.05, 0.05, 0), (radian_90, 0, 0), "cam2", device, args.cam_config_path)
    cam_calib = nn.ModuleList([cam1, cam2])

    renderer = ActiveStereoRenderer(args, metasurface, cam_calib, device)
    pano_cam = PanoramaCam(args, (0, 0, 0), (radian_90, 0, 0), "pano", device)
    estimator = DepthEstimator(pano_cam, cam_calib, device, args)
    e2e_model = E2E(metasurface, renderer, estimator)
    return e2e_model


if __name__ == "__main__":
    parser = Argument()

    # paths
    parser.parser.add_argument(
        "--ckpt_path", type=str, default="/data/wudelong/result/360-sl-metasurface/runs/251229-165719/best_epoch_988.pth"
    )
    # /data/wudelong/result/360-sl-metasurface/runs/251229-165705/best_epoch_436.pth
    # /data/wudelong/result/360-sl-metasurface/runs/251229-165719/best_epoch_988.pth
    parser.parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.parser.add_argument("--fisheye_mask_path", type=str, default="./fisheye_mask.npy")

    # depth units/scales
    parser.parser.add_argument("--depth_scale_m", type=float, default=10.0, help="if GT normalized, depth_m = depth*depth_scale_m")
    parser.parser.add_argument("--gt_depth_unit", type=str, default="auto", choices=["auto", "normalized", "meters"])
    parser.parser.add_argument("--depth_align", type=str, default="median", choices=["none", "median", "ls"])
    parser.parser.add_argument("--min_depth_m", type=float, default=0.05)
    parser.parser.add_argument("--max_depth_m", type=float, default=1e6)

    # normal/occ
    parser.parser.add_argument("--normal_valid_thr", type=float, default=0.2)
    parser.parser.add_argument("--occ_gt_semantics", type=str, default="auto", choices=["auto", "visible", "occluded"])
    parser.parser.add_argument("--occ_tau", type=float, default=0.05)
    parser.parser.add_argument("--occ_margin", type=float, default=0.01)
    parser.parser.add_argument("--occ_outside_as_occluded", action="store_true", help="treat points outside cam2 FOV as occluded")
    parser.parser.set_defaults(occ_outside_as_occluded=True)

    parser.parser.add_argument("--num_workers", type=int, default=4)
    parser.parser.add_argument("--print_every", type=int, default=20)

    args = parser.parse()

    print(f"[INFO] split={args.split}, ckpt={args.ckpt_path}")
    print(f"[INFO] Will save into: {Path(args.ckpt_path).parent}")

    model = build_model(args)
    test_with_metrics(args, model)

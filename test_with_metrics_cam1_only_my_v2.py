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
# from model.StereoMatching_copy import DepthEstimator
from model.StereoMatching_tiny import DepthEstimator
from model.e2e import E2E


# ----------------------------
# Helpers
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
    if x is None: return None
    if not torch.is_tensor(x): x = torch.as_tensor(x)
    if x.ndim == 2: return x[None, None]
    if x.ndim == 3: return x[:, None]
    if x.ndim == 4:
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

    def one_shift(shift: int) -> torch.Tensor:
        p1, p0 = pred[:, :, :, shift:], pred[:, :, :, :-shift]
        g1, g0 = gt[:, :, :, shift:], gt[:, :, :, :-shift]
        m = mask[:, :, :, shift:] * mask[:, :, :, :-shift]
        gx = (p1 - p0) - (g1 - g0)
        p1y, p0y = pred[:, :, shift:, :], pred[:, :, :-shift, :]
        g1y, g0y = gt[:, :, shift:, :], gt[:, :, :-shift, :]
        my = mask[:, :, shift:, :] * mask[:, :, :-shift, :]
        gy = (p1y - p0y) - (g1y - g0y)
        return masked_mean(gx.abs(), m).mean() + masked_mean(gy.abs(), my).mean()

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
        "mae": mae
    }


def to_depth_m_from_gt(depth_gt: torch.Tensor, depth_scale_m: float, gt_unit: str = "auto") -> torch.Tensor:
    d = depth_gt
    if gt_unit == "auto":
        mx = float(d.max().item())
        gt_unit = "normalized" if mx <= 1.5 else "meters"
    if gt_unit == "normalized": return d * depth_scale_m
    if gt_unit == "meters": return d
    raise ValueError(f"Unknown gt_unit: {gt_unit}")


def to_depth_m_from_inv_pred(inv_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 1.0 / inv_pred.clamp_min(eps)


def align_depth_scale(pred_m: torch.Tensor, gt_m: torch.Tensor, mask: torch.Tensor, mode: str = "median") -> torch.Tensor:
    if mode == "none": return pred_m
    valid = (mask > 0.5) & (pred_m > 1e-6) & (gt_m > 1e-6)
    if valid.sum().item() < 10: return pred_m
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
# Normal
# ----------------------------
def depth_to_normal_from_cam(depth_m: torch.Tensor, cam, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, H, W = depth_m.shape
    rays = cam.get_whole_pts().reshape(3, H, W).to(depth_m.device).float()
    P = depth_m[0, 0] * rays
    P = P[None, ...]
    v = (valid_mask > 0.5).float()
    v_l, v_r = F.pad(v[:, :, :, :-1], (1, 0, 0, 0)), F.pad(v[:, :, :, 1:], (0, 1, 0, 0))
    v_u, v_d = F.pad(v[:, :, :-1, :], (0, 0, 1, 0)), F.pad(v[:, :, 1:, :], (0, 0, 0, 1))
    n_valid = v * v_l * v_r * v_u * v_d
    P_l, P_r = F.pad(P[:, :, :, :-1], (1, 0, 0, 0)), F.pad(P[:, :, :, 1:], (0, 1, 0, 0))
    P_u, P_d = F.pad(P[:, :, :-1, :], (0, 0, 1, 0)), F.pad(P[:, :, 1:, :], (0, 0, 0, 1))
    Px = (P_r - P_l) * n_valid
    Py = (P_d - P_u) * n_valid
    n = torch.cross(Px, Py, dim=1)
    n_norm = torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(1e-8)
    return (n / n_norm) * n_valid.repeat(1, 3, 1, 1), n_valid


def normalize_gt_normal(n: torch.Tensor, valid_thr: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    n = to_nchw(n).float()
    if n.shape[1] != 3: n = n.repeat(1, 3, 1, 1)
    nmin, nmax = n.min().item(), n.max().item()
    if nmax > 1.5: n = n / 127.5 - 1.0
    elif nmin >= 0.0 and nmax <= 1.0: n = n * 2.0 - 1.0
    norm = torch.linalg.norm(n, dim=1, keepdim=True)
    valid = (norm > valid_thr).float()
    return (n / norm.clamp_min(1e-8)) * valid.repeat(1, 3, 1, 1), valid


def find_best_normal_transform(pred_n: torch.Tensor, gt_n: torch.Tensor, valid_mask: torch.Tensor):
    best = ((0, 1, 2), (1.0, 1.0, 1.0))
    best_dot = -1e9
    if (valid_mask > 0.5).sum().item() < 10: return best
    for perm in itertools.permutations([0, 1, 2], 3):
        p = pred_n[:, perm, :, :]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            s = torch.tensor(signs, device=pred_n.device, dtype=pred_n.dtype)[None, :, None, None]
            dot = (p * s * gt_n).sum(dim=1, keepdim=True).clamp(-1, 1)
            mdot = masked_mean(dot, valid_mask).mean().item()
            if mdot > best_dot:
                best_dot = mdot
                best = (perm, signs)
    return best


def apply_normal_transform(n: torch.Tensor, perm, signs):
    s = torch.tensor(signs, device=n.device, dtype=n.dtype)[None, :, None, None]
    return n[:, perm, :, :] * s


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
        "cos_loss": cos_loss
    }


# ----------------------------
# Occlusion
# ----------------------------
def normalize_gt_occ(o: torch.Tensor) -> torch.Tensor:
    o = to_nchw(o).float()
    if o.shape[1] != 1: o = o[:, :1]
    return (o > 127.0).float() if o.max().item() > 1.5 else (o > 0.5).float()


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
    loss = -(gt_prob * torch.log(pred) + (1 - gt_prob) * torch.log(1 - pred))
    return masked_mean(loss, mask).mean().item()


def relative_pose_cam1_to_cam2(cam1, cam2, device) -> Tuple[torch.Tensor, torch.Tensor]:
    T1 = cam1.get_extrinsic().to(
        device=device, dtype=torch.float32
    ) if not torch.is_tensor(cam1.get_extrinsic()) else cam1.get_extrinsic().to(
        device=device, dtype=torch.float32
    )
    T2 = cam2.get_extrinsic().to(
        device=device, dtype=torch.float32
    ) if not torch.is_tensor(cam2.get_extrinsic()) else cam2.get_extrinsic().to(
        device=device, dtype=torch.float32
    )
    T_21 = torch.linalg.inv(T2) @ T1
    return T_21[:3, :3].contiguous(), T_21[:3, 3].contiguous()


@torch.no_grad()
def occlusion_cam1_seen_from_cam2_no_cam2_depth(
    depth_cam1_m, cam1, cam2, R_21, t_21, valid_mask_cam1, outside_as_occluded=True, tau=0.05, margin=0.01
):
    B, _, H, W = depth_cam1_m.shape
    rays1 = cam1.get_whole_pts().reshape(3, H, W).to(depth_cam1_m.device).float()
    P1 = (depth_cam1_m[0, 0] * rays1).reshape(3, -1)
    P2 = (R_21 @ P1) + t_21.reshape(3, 1)
    rng2 = torch.linalg.norm(P2, dim=0).clamp_min(1e-8)
    pix = cam2.world2pixel(P2)
    u, v, vproj = pix[0], pix[1], pix[2]

    ui, vi = torch.round(u).long().clamp(0, W - 1), torch.round(v).long().clamp(0, H - 1)
    idx = (vi * W + ui).long()
    m = (valid_mask_cam1[0, 0].reshape(-1) > 0.5) & (vproj > 0.5)

    min_rng = torch.full((H * W, ), float("inf"), device=depth_cam1_m.device, dtype=torch.float32)
    if m.any(): min_rng.scatter_reduce_(0, idx[m], rng2[m], reduce="amin", include_self=True)

    visible = (rng2 - min_rng[idx].clamp_min(1e-8)) <= margin
    occ = (~visible)
    occ_map = torch.zeros((H * W, ), device=depth_cam1_m.device, dtype=torch.float32)
    if outside_as_occluded: occ_map[(valid_mask_cam1[0, 0].reshape(-1) > 0.5) & (vproj <= 0.5)] = 1.0
    occ_map[m] = occ[m].float()
    occ_map = occ_map.reshape(1, 1, H, W)

    occ_prob = torch.zeros_like(occ_map)
    if m.any():
        p = torch.sigmoid(((rng2 - min_rng[idx].clamp_min(1e-8)) - margin) / max(tau, 1e-6)).float()
        flat_p = torch.zeros_like(occ_map).flatten()
        flat_p[m] = p[m]
        if outside_as_occluded: flat_p[(valid_mask_cam1[0, 0].reshape(-1) > 0.5) & (vproj <= 0.5)] = 1.0
        occ_prob = flat_p.reshape(1, 1, H, W)
    return occ_prob, occ_map, valid_mask_cam1.clone()


# ----------------------------
# Checkpoint & Main
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
    model.load_state_dict(sd, strict=True)
    model.to(device)
    return model


@torch.no_grad()
def test_with_metrics(opt, model: nn.Module):
    device = torch.device(opt.device)
    dataset_test = CreateSyntheticDataset(opt.test_path, opt.split)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    ckpt_path = Path(opt.ckpt_path)
    ckpt_dir = ckpt_path.parent
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    model = load_checkpoint_robust(model, str(ckpt_path), device)
    model.eval()

    depth_root, normal_root, occ_root = ckpt_dir / "Depth", ckpt_dir / "Normal", ckpt_dir / "Occlusion"
    for r in [depth_root, normal_root, occ_root]:
        ensure_dir(r)

    fisheye_mask = torch.from_numpy(np.load(opt.fisheye_mask_path)
                                    ).float() if opt.fisheye_mask_path and os.path.exists(opt.fisheye_mask_path) else None

    cam1, cam2 = model.renderer.cam_calib[0], model.renderer.cam_calib[1]
    R_21, t_21 = relative_pose_cam1_to_cam2(cam1, cam2, device=device)
    print(f"[INFO] pose cam1->cam2 t={t_21.detach().cpu().numpy().tolist()}")

    normal_transform_set = False
    normal_perm, normal_signs = (0, 1, 2), (1.0, 1.0, 1.0)
    occ_gt_sem = opt.occ_gt_semantics

    depth_rows, normal_rows, occ_rows = [], [], []

    for cn in ["cam1", "cam2"]:
        for sub in ["pred", "gt", "error"]:
            ensure_dir(depth_root / cn / sub)
            ensure_dir(normal_root / cn / sub)
            ensure_dir(occ_root / cn / sub)

    for i, data in enumerate(dataloader_test):
        name = sanitize_name(data.get("name", f"{i:06d}")[0] if isinstance(data.get("name"), (list, tuple)) else data.get("name"))
        ref_im_list, depth_im_list = data["ref_im_list"], data["depth_im_list"]
        occ_im_list, normal_im_list = data.get("occ_im_list", []), data.get("normal_im_list", [])

        inv_depth_pred, _ = model(ref_im_list, depth_im_list, occ_im_list, normal_im_list)
        inv_preds = list(inv_depth_pred) if isinstance(inv_depth_pred, (list, tuple)) else [inv_depth_pred]

        depth_im_list = [depth_im_list] if not isinstance(depth_im_list, (list, tuple)) else depth_im_list
        occ_im_list = [occ_im_list] if not isinstance(occ_im_list, (list, tuple)) else occ_im_list
        normal_im_list = [normal_im_list] if not isinstance(normal_im_list, (list, tuple)) else normal_im_list

        H, W = to_nchw(depth_im_list[0]).shape[-2:]
        if fisheye_mask is None: mask_fish = torch.ones((1, 1, H, W), device=device)
        else:
            mask_fish = fisheye_mask.to(device)[None, None]
            if mask_fish.shape[-2:] != (H, W): mask_fish = F.interpolate(mask_fish, size=(H, W), mode="nearest")

        for ci in range(min(len(depth_im_list), len(inv_preds))):
            cn = "cam1" if ci == 0 else f"cam{ci+1}"
            depth_gt, inv_pred = to_nchw(depth_im_list[ci].to(device)).float(), to_nchw(inv_preds[ci].to(device)).float()
            depth_gt_m = to_depth_m_from_gt(depth_gt, opt.depth_scale_m, gt_unit=opt.gt_depth_unit)
            depth_pred_m = to_depth_m_from_inv_pred(inv_pred)

            valid = mask_fish * (depth_gt_m > opt.min_depth_m).float() * (depth_gt_m < opt.max_depth_m).float()
            depth_pred_m = align_depth_scale(depth_pred_m, depth_gt_m, valid, mode=opt.depth_align)
            valid = valid * (depth_pred_m > opt.min_depth_m).float() * (depth_pred_m < opt.max_depth_m).float()

            if (i < 3) or ((i + 1) % opt.print_every == 0):
                mae = masked_mean((depth_gt_m - depth_pred_m).abs(), valid).mean().item()
                print(f"[DEPTH] {name} {cn} | MAE={mae:.4f}")

            depth_rows.append({"name": name, "cam": cn, **depth_metrics(depth_pred_m, depth_gt_m, valid)})

            # Save Depth Maps
            gt_vals = depth_gt_m[valid > 0.5]
            dmn, dmx = (gt_vals.min().item(), gt_vals.max().item()) if gt_vals.numel() else (0.0, 1.0)
            dp, dg, vm = depth_pred_m[0, 0].detach().cpu().numpy(), depth_gt_m[0, 0].detach().cpu().numpy(), valid[0,
                                                                                                                   0].detach().cpu().numpy()
            save_colormap_png(depth_root / cn / "pred" / f"{name}.png", np.clip((dp - dmn) / (dmx - dmn + 1e-8), 0, 1) * vm, "Spectral_r")
            save_colormap_png(depth_root / cn / "gt" / f"{name}.png", np.clip((dg - dmn) / (dmx - dmn + 1e-8), 0, 1) * vm, "Spectral_r")
            save_colormap_png(
                depth_root / cn / "error" / f"{name}.png",
                np.clip(np.abs(dp - dg) / (dmx - dmn + 1e-8), 0, 1) * vm, "inferno"
            )

            # Normal
            if len(normal_im_list) > ci and normal_im_list[ci] is not None:
                cam = cam1 if ci == 0 else cam2
                n_pred, n_valid = depth_to_normal_from_cam(depth_pred_m, cam, valid)
                gt_n, gt_nv = normalize_gt_normal(normal_im_list[ci].to(device), valid_thr=opt.normal_valid_thr)
                gt_n = F.interpolate(gt_n, size=n_pred.shape[-2:], mode="bilinear", align_corners=False)
                gt_nv = F.interpolate(gt_nv, size=n_pred.shape[-2:], mode="nearest")

                if (not normal_transform_set) and ci == 0:
                    n_gt_d, n_v2 = depth_to_normal_from_cam(depth_gt_m, cam1, valid)
                    normal_perm, normal_signs = find_best_normal_transform(n_gt_d, gt_n, n_v2 * gt_nv)
                    normal_transform_set = True
                    print(f"[INFO] Normal transform: {normal_perm}, {normal_signs}")

                n_pred = apply_normal_transform(n_pred, normal_perm, normal_signs)
                nmet = normal_metrics(n_pred, gt_n, n_valid * gt_nv)
                normal_rows.append({"name": name, "cam": cn, **nmet})

                # Save Normal Maps
                dot = (n_pred * gt_n).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
                ang_vis = np.clip((torch.acos(dot) * (180.0 / math.pi) * n_valid * gt_nv)[0, 0].detach().cpu().numpy() / 60.0, 0, 1)
                save_colormap_png(normal_root / cn / "error" / f"{name}_angular.png", ang_vis, "inferno")
                save_uint8_png(
                    normal_root / cn / "pred" / f"{name}.png",
                    (np.clip(n_pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
                )
                save_uint8_png(
                    normal_root / cn / "gt" / f"{name}.png",
                    (np.clip(gt_n[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
                )

            # Occlusion (Matches visual error map using L1/MAE)
            if ci == 0 and len(occ_im_list) > 0 and occ_im_list[0] is not None:
                gt_occ = normalize_gt_occ(occ_im_list[0].to(device))
                if gt_occ.shape[-2:] != (H, W): gt_occ = F.interpolate(gt_occ, size=(H, W), mode="nearest")

                occ_prob, occ_bin, occ_mask = occlusion_cam1_seen_from_cam2_no_cam2_depth(
                    depth_pred_m, cam1, cam2, R_21, t_21, valid, opt.occ_outside_as_occluded, opt.occ_tau, opt.occ_margin
                )

                if occ_gt_sem == "auto" and i == 0:
                    gt_b = (gt_occ > 0.5) & (occ_mask > 0.5)
                    iou_occ = bin_stats((occ_bin > 0.5) & (occ_mask > 0.5), gt_b)["iou"]
                    iou_vis = bin_stats(((1.0 - occ_bin) > 0.5) & (occ_mask > 0.5), gt_b)["iou"]
                    occ_gt_sem = "occluded" if iou_occ >= iou_vis else "visible"
                    print(f"[INFO] Occlusion GT semantics: {occ_gt_sem}")

                pred_bin2 = (1.0 - occ_bin) if occ_gt_sem == "visible" else occ_bin
                pred_prob2 = (1.0 - occ_prob) if occ_gt_sem == "visible" else occ_prob
                pred_bin2 = pred_bin2 * occ_mask
                pred_prob2 = pred_prob2 * occ_mask

                st = bin_stats((pred_bin2 > 0.5) & (occ_mask > 0.5), (gt_occ > 0.5) & (occ_mask > 0.5))

                # --- NEW: Calculate MAE specifically for matching visual error map (abs diff) ---
                occ_mae = masked_mean((pred_bin2 - gt_occ).abs(), occ_mask).mean().item()
                # ---------------------------------------------------------------------------------

                loss_bce = bce_loss(pred_prob2, gt_occ, occ_mask)
                occ_rows.append({"name": name, "cam": "cam1", **st, "bce": loss_bce, "mae": occ_mae})  # Added MAE here

                gt_np = (gt_occ[0, 0].detach().cpu().numpy()).astype(np.float32) * (occ_mask[0, 0].detach().cpu().numpy())
                pr_np = (pred_bin2[0, 0].detach().cpu().numpy()).astype(np.float32) * (occ_mask[0, 0].detach().cpu().numpy())
                err_np = np.abs(pr_np - gt_np)

                save_uint8_png(occ_root / "cam1" / "gt" / f"{name}.png", (gt_np * 255).astype(np.uint8))
                save_uint8_png(occ_root / "cam1" / "pred" / f"{name}.png", (pr_np * 255).astype(np.uint8))
                save_uint8_png(occ_root / "cam1" / "error" / f"{name}.png", (err_np * 255).astype(np.uint8))

    # Save outputs
    for p, rows in [(ckpt_dir / "metrics_depth.csv", depth_rows), (ckpt_dir / "metrics_normal.csv", normal_rows),
                    (ckpt_dir / "metrics_occlusion.csv", occ_rows)]:
        if rows:
            with open(p, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=rows[0].keys()).writeheader()
                csv.DictWriter(f, fieldnames=rows[0].keys()).writerows(rows)

    per_image_map = {}
    for rows, prefix in [(depth_rows, "depth"), (normal_rows, "normal"), (occ_rows, "occ")]:
        for r in rows:
            if r.get("cam", "cam1") != "cam1": continue
            nm = r["name"]
            if nm not in per_image_map: per_image_map[nm] = {}
            for k, v in r.items():
                if isinstance(v, (int, float)) and not math.isnan(v): per_image_map[nm][f"{prefix}_{k}"] = v

    with open(ckpt_dir / "per_image_metrics.json", "w") as f:
        json.dump(per_image_map, f, indent=2)
    print("[DONE] Saved separate CSVs and 'per_image_metrics.json'")


def build_model(args):
    device = torch.device(args.device)
    metasurface = Metasurface(args, device)
    radian_90 = math.radians(90)
    cam1 = FisheyeCam(args, (0.05, 0.05, 0), (radian_90, 0, 0), "cam1", device, args.cam_config_path)
    cam2 = FisheyeCam(args, (-0.05, 0.05, 0), (radian_90, 0, 0), "cam2", device, args.cam_config_path)
    cam_calib = nn.ModuleList([cam1, cam2])
    return E2E(
        metasurface, ActiveStereoRenderer(args, metasurface, cam_calib, device),
        DepthEstimator(PanoramaCam(args, (0, 0, 0), (radian_90, 0, 0), "pano", device), cam_calib, device, args)
    )


if __name__ == "__main__":
    parser = Argument()
    parser.parser.add_argument("--ckpt_path", type=str, required=True)
    parser.parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.parser.add_argument("--fisheye_mask_path", type=str, default="./fisheye_mask.npy")
    parser.parser.add_argument("--depth_scale_m", type=float, default=10.0)
    parser.parser.add_argument("--gt_depth_unit", type=str, default="auto")
    parser.parser.add_argument("--depth_align", type=str, default="median")
    parser.parser.add_argument("--min_depth_m", type=float, default=0.05)
    parser.parser.add_argument("--max_depth_m", type=float, default=1e6)
    parser.parser.add_argument("--normal_valid_thr", type=float, default=0.2)
    parser.parser.add_argument("--occ_gt_semantics", type=str, default="auto")
    parser.parser.add_argument("--occ_tau", type=float, default=0.05)
    parser.parser.add_argument("--occ_margin", type=float, default=0.01)
    parser.parser.add_argument("--occ_outside_as_occluded", action="store_true")
    parser.parser.set_defaults(occ_outside_as_occluded=True)
    parser.parser.add_argument("--num_workers", type=int, default=4)
    parser.parser.add_argument("--print_every", type=int, default=20)
    args = parser.parse()
    test_with_metrics(args, build_model(args))

import os
import re
from typing import List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def _extract_timestamp_token(path: str) -> str:
    m = re.search(r"(\d{6}-\d{6})", path)
    if m:
        return m.group(1).replace("-", "")
    m = re.search(r"(\d{12})", path)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(path))[0]


def _scatter_size_from_data_diameter(ax, fig, data_diameter: float) -> float:
    """Convert marker diameter in data units -> scatter `s` (points^2)."""
    fig.canvas.draw()
    x0, _ = ax.transData.transform((0.0, 0.0))
    x1, _ = ax.transData.transform((data_diameter, 0.0))
    diameter_pixels = abs(x1 - x0)
    diameter_points = diameter_pixels * 72.0 / fig.dpi
    return float(diameter_points ** 2)


def _draw_orientation_circles_and_arrows(
    ax,
    phase_map: np.ndarray,
    stride: int = 4,
    circle_diameter: Optional[float] = None,
    arrow_scale: float = 0.90,
    draw_circles: bool = True,
    circle_edge: str = "0.55",
    circle_face_alpha: float = 0.0,
    circle_lw: float = 0.25,
    arrow_color: str = "0.05",
    arrow_width: float = 0.6,
    headwidth: float = 3.0,
    headlength: float = 4.0,
    headaxislength: float = 3.5,
):
    """
    UNIQUE mapping: angle_deg = deg(phase) % 360  (0~360)
    Clockwise from Up in image coordinates (origin='upper'):
      0° up, 90° right, 180° down, 270° left.
    """
    h, w = phase_map.shape
    rows, cols = np.indices((h, w))
    rows = rows[::stride, ::stride]
    cols = cols[::stride, ::stride]
    ph = phase_map[::stride, ::stride].astype(np.float32)

    X = cols.ravel().astype(np.float32)
    Y = rows.ravel().astype(np.float32)
    ph = ph.ravel()

    spacing = float(stride)
    if circle_diameter is None:
        circle_diameter = 0.92 * spacing
    radius = 0.5 * circle_diameter
    arrow_len = radius * float(arrow_scale)

    angle_deg = np.mod(np.degrees(ph), 360.0).astype(np.float32)
    ang = np.deg2rad(angle_deg)

    # image coords (y down): 0° is up => (0,-1); clockwise => (sin, -cos)
    U = (np.sin(ang) * arrow_len).astype(np.float32)
    V = (-np.cos(ang) * arrow_len).astype(np.float32)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)  # keep same as imshow(origin='upper')
    ax.axis("off")

    if draw_circles:
        s = _scatter_size_from_data_diameter(ax, ax.figure, circle_diameter)
        face = (1, 1, 1, circle_face_alpha)
        ax.scatter(
            X,
            Y,
            s=s,
            marker="o",
            facecolors=face,
            edgecolors=circle_edge,
            linewidths=circle_lw,
            zorder=2,
        )

    ax.quiver(
        X,
        Y,
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="tail",
        units="dots",
        width=arrow_width,
        headwidth=headwidth,
        headlength=headlength,
        headaxislength=headaxislength,
        color=arrow_color,
        linewidth=0.0,
        minlength=0,
        zorder=3,
    )


def _downsample_to_max_res(img: np.ndarray, max_res: int) -> np.ndarray:
    """Downsample a 2D image to (max_res, max_res) if needed (bilinear)."""
    if max_res is None or max_res <= 0:
        return img
    h, w = img.shape
    if h <= max_res and w <= max_res:
        return img
    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(img.astype(np.float32))[None, None]  # 1x1xHxW
        t = F.interpolate(t, size=(max_res, max_res), mode="bilinear", align_corners=False)
        return t[0, 0].cpu().numpy()
    except Exception:
        # fallback: nearest
        ys = np.linspace(0, h - 1, max_res).astype(np.int64)
        xs = np.linspace(0, w - 1, max_res).astype(np.int64)
        return img[np.ix_(ys, xs)]


def compute_fourier_pattern_from_phase(
    phase_map: np.ndarray,
    use_legacy_physics: bool = True,
    legacy_extra_square: Optional[bool] = None,
    max_res: int = 500,
) -> np.ndarray:
    """
    Reproduce renderer's `fourier_pattern` logic using ONLY the saved phase map.

    - Metasurface.propagate():
        field = exp(1j*phase)
        spectrum = fftshift(fft2(fftshift(field), norm=('ortho' if not legacy else None)))
        intensity = |spectrum|^2

    - renderer.render():
        if use_legacy_physics: pattern_360 = metasurface.propagate() ** 2  (extra square)
        else:                 pattern_360 = metasurface.propagate()
    """
    if legacy_extra_square is None:
        legacy_extra_square = bool(use_legacy_physics)

    field = np.exp(1j * phase_map.astype(np.float32)).astype(np.complex64)
    field = np.fft.fftshift(field)

    if use_legacy_physics:
        spec = np.fft.fft2(field)  # no norm
    else:
        spec = np.fft.fft2(field, norm="ortho")  # energy-preserving (new physics)
    spec = np.fft.fftshift(spec)

    intensity = (np.abs(spec) ** 2).astype(np.float32)
    if legacy_extra_square:
        intensity = (intensity ** 2).astype(np.float32)

    intensity = _downsample_to_max_res(intensity, max_res=max_res)
    return intensity


def visualize_phase_maps_split_save(
    file_list: List[str],
    name_list: List[str],
    save_root: str = "vis/phasemap",
    dpi_hsv: int = 400,
    dpi_ori: int = 400,
    dpi_fourier: int = 400,
    stride: int = 16,
    cell_px: int = 20,
    # --- Fourier config ---
    use_legacy_physics: bool = True,
    legacy_extra_square: Optional[bool] = None,
    fourier_max_res: int = 500,
    # --- phase scaling (e.g. DINER old npy may need *pi) ---
    phase_scale_list: Optional[List[float]] = None,
):
    """
    Save three figures:
      (A) 1xN phase (wrapped to [-pi,pi]) using twilight
      (B) 1xN orientation glyphs (circles+arrows, angle=deg(phase)%360)
      (C) 1xN Fourier pattern (renderer-style pattern_360), log scale

    phase_scale_list:
      - If you suspect some saved npy is normalized (e.g. DINER outputs in [-1,1]),
        pass phase_scale_list=[np.pi, 1.0, ...] to convert to radians.
      - If your npy is already radians, leave it None (defaults to 1.0 for all).
    """
    num_plots = len(file_list)
    if num_plots == 0 or num_plots > 3:
        raise ValueError(f"Expected 1 to 3 file paths, but got {num_plots}.")
    if len(name_list) != num_plots:
        raise ValueError(f"name_list length must equal file_list length. Got {len(name_list)} vs {num_plots}.")
    if phase_scale_list is not None and len(phase_scale_list) != num_plots:
        raise ValueError("phase_scale_list must be None or have same length as file_list.")

    os.makedirs(save_root, exist_ok=True)
    out_stem = "_".join(_extract_timestamp_token(p) for p in file_list)

    # --- Load phase maps ---
    phase_maps = []
    valid_names = []
    for idx, (p, n) in enumerate(zip(file_list, name_list)):
        if not os.path.exists(p):
            print(f"Warning: File not found: {p}, skip.")
            continue
        pm = np.load(p).astype(np.float32)
        scale = 1.0 if phase_scale_list is None else float(phase_scale_list[idx])
        pm = pm * scale
        phase_maps.append(pm)
        valid_names.append(n)

    if len(phase_maps) == 0:
        raise FileNotFoundError("No valid npy files found.")

    print(f"Loaded {len(phase_maps)} phase maps.")
    for nm, pm in zip(valid_names, phase_maps):
        print(f"[{nm}] max={pm.max():.6f}, min={pm.min():.6f}, mean={pm.mean():.6f}, std={pm.std():.6f}")

    # ========== (A) Phase figure ==========
    phase_path = os.path.join(save_root, f"{out_stem}_phase.png")
    fig1, axes1 = plt.subplots(1, len(phase_maps), figsize=(7 * len(phase_maps), 5), squeeze=False)
    for i, (pm, nm) in enumerate(zip(phase_maps, valid_names)):
        ax = axes1[0, i]
        pm_vis = np.angle(np.exp(1j * pm))  # wrap to (-pi, pi]
        # im = ax.imshow(pm_vis, cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="upper")
        im = ax.imshow(pm_vis, cmap="twilight", origin="upper")
        ax.set_title(f"{nm} (phase, wrapped)", fontsize=12)
        ax.axis("off")
        cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Phase (radians)", fontsize=10)
    plt.tight_layout()
    plt.savefig(phase_path, dpi=dpi_hsv, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved phase figure to: {phase_path}")

    # ========== (B) Orientation figure ==========
    h, w = phase_maps[0].shape
    gh, gw = h // stride, w // stride
    one_panel_w_in = (gw * cell_px) / dpi_ori
    one_panel_h_in = (gh * cell_px) / dpi_ori
    ori_fig_w = max(4.0, one_panel_w_in * len(phase_maps))
    ori_fig_h = max(4.0, one_panel_h_in)

    ori_path = os.path.join(save_root, f"{out_stem}_orientation_deg360_stride{stride}.png")
    fig2, axes2 = plt.subplots(1, len(phase_maps), figsize=(ori_fig_w, ori_fig_h), squeeze=False)
    for i, (pm, nm) in enumerate(zip(phase_maps, valid_names)):
        ax = axes2[0, i]
        _draw_orientation_circles_and_arrows(
            ax,
            pm,
            stride=stride,
            draw_circles=True,
            circle_face_alpha=0.15,
            arrow_width=0.6,
        )
        ax.set_title(f"{nm} deg(phase)%360", fontsize=12)

    plt.tight_layout()
    plt.savefig(ori_path, dpi=dpi_ori, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved orientation figure to: {ori_path}")

    # ========== (C) Fourier pattern figure ==========
    suffix = "legacy" if use_legacy_physics else "new"
    eff_pow4 = legacy_extra_square if legacy_extra_square is not None else use_legacy_physics
    if eff_pow4:
        suffix += "_pow4"

    fourier_path = os.path.join(save_root, f"{out_stem}_fourier_{suffix}.png")

    patterns = [
        compute_fourier_pattern_from_phase(
            pm,
            use_legacy_physics=use_legacy_physics,
            legacy_extra_square=legacy_extra_square,
            max_res=fourier_max_res,
        ) for pm in phase_maps
    ]

    eps = 1e-8
    logs = [np.log(p + eps) for p in patterns]
    all_vals = np.concatenate([x.ravel() for x in logs])
    vmin = float(np.percentile(all_vals, 5.0))
    vmax = float(np.percentile(all_vals, 99.5))

    fig3, axes3 = plt.subplots(1, len(patterns), figsize=(7 * len(patterns), 5), squeeze=False)
    for i, (lg, nm) in enumerate(zip(logs, valid_names)):
        ax = axes3[0, i]
        im = ax.imshow(lg, cmap="magma", origin="upper")
        # im = ax.imshow(lg, cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(f"{nm} (Fourier pattern)", fontsize=12)
        ax.axis("off")
        cbar = fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("intensity", fontsize=10)

    plt.tight_layout()
    plt.savefig(fourier_path, dpi=dpi_fourier, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved Fourier figure to: {fourier_path}")

    return phase_path, ori_path, fourier_path


if __name__ == "__main__":
    file_list = [
        "/data/wudelong/result/360-sl-metasurface/runs/251229-165705/phase_best_epoch_436.npy",
        "/data/wudelong/result/360-sl-metasurface/runs/251229-165719/phase_best_epoch_988.npy"
    ]
    name_list = ["NORM", "Pixelwise"]
    save_root = "./vis/251229/phasemap"

    # If your DINER npy was saved as normalized output in [-1, 1], try:
    # phase_scale_list = [np.pi, 1.0]
    phase_scale_list = None

    visualize_phase_maps_split_save(
        file_list,
        name_list,
        save_root,
        stride=32,
        dpi_ori=400,
        cell_px=40,
        use_legacy_physics=True,  # set True if training used legacy physics
        legacy_extra_square=None,  # None -> follow renderer behavior
        fourier_max_res=500,
        phase_scale_list=phase_scale_list,
    )

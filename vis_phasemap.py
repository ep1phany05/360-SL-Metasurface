import os
import re
from typing import List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

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
        face = (1, 1, 1, circle_face_alpha)  # 半透明白填充更容易“分格子”
        ax.scatter(X, Y, s=s, marker="o", facecolors=face, edgecolors=circle_edge, linewidths=circle_lw, zorder=2)

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


def visualize_phase_maps_split_save(
    file_list: List[str],
    name_list: List[str],
    save_root: str = "vis/phasemap",
    dpi_hsv: int = 300,
    dpi_ori: int = 300,
    stride: int = 4,
    cell_px: int = 12,  # 每个“采样点”希望占多少像素（越大越清楚，但图越大）
):
    """
    Save two separate figures:
      1) 1xN HSV maps
      2) 1xN orientation (circles+arrows, angle=deg(phase)%360)
    """
    num_plots = len(file_list)
    if num_plots == 0 or num_plots > 3:
        raise ValueError(f"Expected 1 to 3 file paths, but got {num_plots}.")
    if len(name_list) != num_plots:
        raise ValueError(f"name_list length must equal file_list length. Got {len(name_list)} vs {num_plots}.")

    os.makedirs(save_root, exist_ok=True)
    out_stem = "_".join(_extract_timestamp_token(p) for p in file_list)

    # Load phase maps once
    phase_maps = []
    valid_names = []
    valid_paths = []
    for p, n in zip(file_list, name_list):
        if not os.path.exists(p):
            print(f"Warning: File not found: {p}, skip.")
            continue

        pm = np.load(p).astype(np.float32)
        # if n.strip().lower() == "diner" or "diner" in n.strip().lower():
        #     pm = pm * torch.pi

        phase_maps.append(pm)
        valid_names.append(n)
        valid_paths.append(p)

    print(f"Loaded {len(phase_maps)} valid phase maps from {len(file_list)} paths.")
    print(phase_maps[0].max(), phase_maps[0].min(), phase_maps[1].max(), phase_maps[1].min())

    if len(phase_maps) == 0:
        raise FileNotFoundError("No valid npy files found.")

    # ========== (A) HSV figure ==========
    hsv_path = os.path.join(save_root, f"{out_stem}_phase.png")
    fig1, axes1 = plt.subplots(1, len(phase_maps), figsize=(7 * len(phase_maps), 5), squeeze=False)
    for i, (pm, nm) in enumerate(zip(phase_maps, valid_names)):
        ax = axes1[0, i]
        pm_vis = np.angle(np.exp(1j * pm))
        im = ax.imshow(pm_vis, cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="upper")
        ax.set_title(f"{nm}", fontsize=12)
        ax.axis("off")
        cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Phase (radians)", fontsize=10)
    plt.tight_layout()
    plt.savefig(hsv_path, dpi=dpi_hsv, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved HSV figure to: {hsv_path}")

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
            circle_face_alpha=0.15,  # 0.1~0.3
            arrow_width=0.6,  # 0.4~1.2
        )
        ax.set_title(f"{nm} (deg(phase)%360, stride={stride})", fontsize=12)

    plt.tight_layout()
    plt.savefig(ori_path, dpi=dpi_ori, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved Orientation figure to: {ori_path}")

    return hsv_path, ori_path


if __name__ == "__main__":
    file_list = ["runs/DINER/phase_best_epoch_417.npy", "runs/Origin/phase_best_epoch_818.npy"]
    name_list = ["DINER", "Origin"]
    visualize_phase_maps_split_save(file_list, name_list, stride=16, dpi_ori=300, cell_px=20)

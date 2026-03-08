import json
import math
import shutil
import os
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------------------------------------
# 配置区域
# ----------------------------------------------------
PATH_PIXEL = Path("/data/wudelong/result/360-sl-metasurface/runs/260125-172900/per_image_metrics.json")
PATH_NORM = Path("/data/wudelong/result/360-sl-metasurface/runs/260125-172954/per_image_metrics.json")
OUTPUT_DIR = Path("/data/wudelong/result/360-sl-metasurface/runs/best_visuals_tiny")  # 结果将保存在当前目录下的这个文件夹
# ----------------------------------------------------


def load_metrics(json_path: Path) -> Dict[str, Dict[str, float]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_files_for_entry(
    rank: int,
    name: str,
    category_subdir: Path,
    root_pixel: Path,
    root_norm: Path,
    vis_type: str,
    suffix: str = ".png"  # 图片后缀
):
    """
    Copies:
      1. GT (from NORM folder, as GT is same)
      2. Pixelwise Pred & Error
      3. NORM Pred & Error
    """
    # Mapping for folder structure based on vis_type
    # Structure in test script: {Root}/{Type}/cam1/{subtype}/{name}.png
    # vis_type: "Depth", "Normal", "Occlusion"

    # 1. Copy GT
    gt_src = root_norm / vis_type / "cam1" / "gt" / f"{name}{suffix}"
    gt_dst = category_subdir / f"rank{rank}_{name}_GT.png"
    if gt_src.exists(): shutil.copy(gt_src, gt_dst)

    # 2. Copy Pixelwise
    pix_pred = root_pixel / vis_type / "cam1" / "pred" / f"{name}{suffix}"
    pix_err = root_pixel / vis_type / "cam1" / "error" / (f"{name}_angular.png" if vis_type == "Normal" else f"{name}{suffix}")

    if pix_pred.exists(): shutil.copy(pix_pred, category_subdir / f"rank{rank}_{name}_Pixel_Pred.png")
    if pix_err.exists(): shutil.copy(pix_err, category_subdir / f"rank{rank}_{name}_Pixel_Err.png")

    # 3. Copy NORM
    norm_pred = root_norm / vis_type / "cam1" / "pred" / f"{name}{suffix}"
    norm_err = root_norm / vis_type / "cam1" / "error" / (f"{name}_angular.png" if vis_type == "Normal" else f"{name}{suffix}")

    if norm_pred.exists(): shutil.copy(norm_pred, category_subdir / f"rank{rank}_{name}_NORM_Pred.png")
    if norm_err.exists(): shutil.copy(norm_err, category_subdir / f"rank{rank}_{name}_NORM_Err.png")


def get_top_improvements(metrics_base, metrics_target, key, top_k=5):
    """Returns List[(name, val_base, val_target, diff)] sorted by diff desc"""
    improvements = []
    common = set(metrics_base.keys()) & set(metrics_target.keys())
    for name in common:
        if key not in metrics_base[name] or key not in metrics_target[name]: continue
        vb, vt = metrics_base[name][key], metrics_target[name][key]
        if math.isnan(vb) or math.isnan(vt): continue
        diff = vb - vt  # Positive if Base > Target (Lower is better)
        improvements.append((name, vb, vt, diff))
    improvements.sort(key=lambda x: x[3], reverse=True)
    return improvements[:top_k]


def main():
    print(f"Loading metrics...")
    data_pixel = load_metrics(PATH_PIXEL)
    data_norm = load_metrics(PATH_NORM)

    # Determine image roots from json paths
    root_pixel = PATH_PIXEL.parent
    root_norm = PATH_NORM.parent

    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    print("-" * 60)

    # ---------------------------------------------------------
    # 1. Depth (Metric: MAE)
    # ---------------------------------------------------------
    print(">> Processing Depth (Best NORM vs Pixel improvements based on MAE)...")
    depth_top = get_top_improvements(data_pixel, data_norm, "depth_mae", top_k=5)

    out_d = OUTPUT_DIR / "Depth"
    out_d.mkdir()

    for i, (name, vb, vn, diff) in enumerate(depth_top):
        print(f"  {i+1}. {name} | Pixel: {vb:.4f} -> NORM: {vn:.4f} (Diff: {diff:.4f})")
        copy_files_for_entry(i + 1, name, out_d, root_pixel, root_norm, "Depth")

    # ---------------------------------------------------------
    # 2. Normal (Metric: Ang Mean)
    # ---------------------------------------------------------
    print("\n>> Processing Normal (Best NORM vs Pixel improvements based on Angular Mean)...")
    # normal_ang_mean corresponds to visualization
    normal_top = get_top_improvements(data_pixel, data_norm, "normal_ang_mean", top_k=5)

    out_n = OUTPUT_DIR / "Normal"
    out_n.mkdir()

    for i, (name, vb, vn, diff) in enumerate(normal_top):
        print(f"  {i+1}. {name} | Pixel: {vb:.4f} -> NORM: {vn:.4f} (Diff: {diff:.4f})")
        copy_files_for_entry(i + 1, name, out_n, root_pixel, root_norm, "Normal")

    # ---------------------------------------------------------
    # 3. Occlusion (Metric: MAE) - Replaced BCE with MAE
    # ---------------------------------------------------------
    print("\n>> Processing Occlusion (Best NORM vs Pixel improvements based on MAE)...")
    # Using 'occ_mae' which we added to the script
    occ_top = get_top_improvements(data_pixel, data_norm, "occ_mae", top_k=5)

    out_o = OUTPUT_DIR / "Occlusion"
    out_o.mkdir()

    for i, (name, vb, vn, diff) in enumerate(occ_top):
        print(f"  {i+1}. {name} | Pixel: {vb:.4f} -> NORM: {vn:.4f} (Diff: {diff:.4f})")
        copy_files_for_entry(i + 1, name, out_o, root_pixel, root_norm, "Occlusion")

    # ---------------------------------------------------------
    # 4. Combined
    # ---------------------------------------------------------
    print("\n>> Processing Combined (Best in ALL metrics)...")
    combined_scores = []
    common = set(data_pixel.keys()) & set(data_norm.keys())

    for name in common:
        dp, dn = data_pixel[name], data_norm[name]
        # Check required keys (using occ_mae now)
        if not all(k in dp for k in ["depth_mae", "normal_ang_mean", "occ_mae"]): continue
        if not all(k in dn for k in ["depth_mae", "normal_ang_mean", "occ_mae"]): continue

        d_mae_p, d_mae_n = dp["depth_mae"], dn["depth_mae"]
        n_ang_p, n_ang_n = dp["normal_ang_mean"], dn["normal_ang_mean"]
        o_mae_p, o_mae_n = dp["occ_mae"], dn["occ_mae"]

        # Strictly better in all
        if (d_mae_n < d_mae_p) and (n_ang_n < n_ang_p) and (o_mae_n < o_mae_p):
            # Sum of relative improvements
            score = ((d_mae_p - d_mae_n) / (d_mae_p + 1e-9) + (n_ang_p - n_ang_n) / (n_ang_p + 1e-9) + (o_mae_p - o_mae_n) /
                     (o_mae_p + 1e-9))
            combined_scores.append({"name": name, "score": score})

    combined_scores.sort(key=lambda x: x["score"], reverse=True)
    top_comb = combined_scores[:5]

    out_c = OUTPUT_DIR / "Combined"
    out_c.mkdir()

    for i, item in enumerate(top_comb):
        name = item["name"]
        print(f"  {i+1}. {name} | Score: {item['score']:.4f}")
        # Copy ALL types for the combined winner
        # Create a subfolder per image to keep it clean, or just prefix
        # Let's prefix rank_type
        copy_files_for_entry(i + 1, name, out_c, root_pixel, root_norm, "Depth", suffix=".png")
        # For Normal/Occlusion, we reuse the function but rename files slightly if needed to avoid collision?
        # Actually copy_files_for_entry uses fixed names, so we need to be careful.
        # Modified strategy for combined: Copy into rank{i}_{name}/ folder

        target_sub = out_c / f"rank{i+1}_{name}"
        target_sub.mkdir()

        copy_files_for_entry(i + 1, name, target_sub, root_pixel, root_norm, "Depth")
        copy_files_for_entry(i + 1, name, target_sub, root_pixel, root_norm, "Normal")
        copy_files_for_entry(i + 1, name, target_sub, root_pixel, root_norm, "Occlusion")

    print(f"\n[DONE] All results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

import piq
import torch

# def psnr_metric(
#     ground_truth: torch.Tensor,
#     prediction: torch.Tensor,
#     max_pixel: float = 1.0,
#     average: str = "average"  # 新增参数，默认为 "per_sample"
# ) -> torch.Tensor:
#     """
#     计算峰值信噪比 (PSNR)。
#
#     参数：
#     - ground_truth (torch.Tensor): 真实图像张量，形状为 [Batch, H, W, Channels]。
#     - prediction (torch.Tensor): 预测图像张量，形状为 [Batch, H, W, Channels]。
#     - max_pixel (float): 图像的最大像素值。默认为1.0。
#     - average (str): 返回值的类型。
#         - "per_sample"：返回每个样本的 PSNR，形状为 [Batch]。
#         - "average"：返回所有样本的平均 PSNR，标量。
#
#     返回：
#     - torch.Tensor: 根据 `average` 参数返回 PSNR 值。
#     """
#     # 验证输入形状一致
#     if ground_truth.shape != prediction.shape:
#         raise ValueError("ground_truth 和 prediction 的形状必须相同")
#
#     # 计算每个样本的均方误差 (MSE)
#     mse = torch.mean((ground_truth - prediction) ** 2, dim=[1, 2, 3])
#
#     # 避免 MSE 为零
#     mse = torch.clamp(mse, min=1e-10)
#
#     # 计算每个样本的 PSNR
#     psnr = 20 * torch.log10(torch.tensor(max_pixel)) - 10 * torch.log10(mse)
#
#     if average == "per_sample":
#         return psnr  # 返回每个样本的 PSNR，形状为 [Batch]
#     elif average == "average":
#         return torch.mean(psnr)  # 返回所有样本的平均 PSNR，标量
#     else:
#         raise ValueError("参数 `average` 必须为 'per_sample' 或 'average'")


def psnr_metric(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    计算 PSNR (Peak Signal-to-Noise Ratio) 得分，返回对 batch 维度求平均后的标量。

    Args:
        img1 (torch.Tensor): 预测图像，形状 [B, H, W, C]。
        img2 (torch.Tensor): 目标（参考）图像，形状 [B, H, W, C]。

    Returns:
        torch.Tensor: PSNR 得分的平均值（dB）。
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    # ! 假设图像在 [0,1] 范围；如不是，请自行归一化或将 data_range=255.0 等
    data_range = 1.0

    # 形状转换 [B, C, H, W]
    img1 = img1.permute(0, 3, 1, 2)
    img2 = img2.permute(0, 3, 1, 2)

    # reduction="mean" 表示对 batch 维度结果求平均
    psnr_val = piq.psnr(x=img1, y=img2, data_range=data_range, reduction="mean")
    return psnr_val


def ssim_metric(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    计算 SSIM，返回对 batch 维度求平均后的标量。

    Args:
        img1 (torch.Tensor): 预测图像，形状 [B, H, W, C]。
        img2 (torch.Tensor): 目标（参考）图像，形状 [B, H, W, C]。

    Returns:
        torch.Tensor: SSIM 得分的平均值，范围通常在 [0,1]，越大表示相似度越高。
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    # ! 假设输入图像在 [0,1] 范围；若不是请根据实际情况修改 data_range
    data_range = 1.0  # 也可设置为 255.0

    # 调整形状到 [B, C, H, W]
    img1 = img1.permute(0, 3, 1, 2)
    img2 = img2.permute(0, 3, 1, 2)

    # reduction="mean" 表示对 batch 维度结果求平均
    ssim_val = piq.ssim(x=img1, y=img2, data_range=data_range, reduction="mean")
    return ssim_val


def lpips_metric(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    计算 LPIPS (Learned Perceptual Image Patch Similarity) 得分，返回对 batch 维度求平均后的标量。

    Args:
        img1 (torch.Tensor): 预测图像，形状 [B, H, W, C]。
        img2 (torch.Tensor): 目标（参考）图像，形状 [B, H, W, C]。

    Returns:
        torch.Tensor: LPIPS 得分的平均值，越低表示感知相似度越高。
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    # ! 同理，假设输入图像在 [0,1] 范围
    data_range = 1.0

    # 调整形状到 [B, C, H, W]
    img1 = img1.permute(0, 3, 1, 2)
    img2 = img2.permute(0, 3, 1, 2)

    lpips_fn = piq.LPIPS(reduction='mean')
    lpips_val = lpips_fn(img1, img2)
    return lpips_val

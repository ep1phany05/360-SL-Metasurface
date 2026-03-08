import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.ArgParser import Argument
from dataset.dataset import CreateSyntheticDataset
from model.Metasurface import Metasurface
from utils.Camera import *
from model.utils.setup_seed import setup_seed
from Image_formation.renderer import *
# from model.StereoMatching_copy import DepthEstimator
from model.StereoMatching_tiny import DepthEstimator
from termcolor import colored
from datetime import datetime
from pathlib import Path
import os
from model.e2e import *
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def mkdirs(pth):
    Path(pth).mkdir(parents=True, exist_ok=True)


def grad_loss(output, gt):

    def one_grad(shift):
        ox = output[:, shift:] - output[:, :-shift]
        oy = output[:, :, shift:] - output[:, :, :-shift]
        gx = gt[:, shift:] - gt[:, :-shift]
        gy = gt[:, :, shift:] - gt[:, :, :-shift]
        loss = (ox - gx).abs().mean() + (oy - gy).abs().mean()
        return loss

    loss = (one_grad(1) + one_grad(2) + one_grad(3)) / 3.
    return loss


def illum_tv(output):

    def one_grad(shift):
        ox = output[shift:] - output[:-shift]
        oy = output[:, shift:] - output[:, :-shift]

        loss = ox.abs().mean() + oy.abs().mean()
        return loss

    loss = (one_grad(1) + one_grad(2) + one_grad(3)) / 3.
    return loss


def plot_grid_wireframe(ax, grid, step=20):

    # 降采样
    grid_sub = grid[::step, ::step, :]
    h, w, _ = grid_sub.shape

    U = grid_sub[..., 0]
    V = grid_sub[..., 1]

    # 1. 绘制“横线”（连接图像中每一行的点）
    # 形状构造: (h, w, 2) -> (h, w-1, 2) 连接相邻点
    # 为了简单，使用 LineCollection 需要一系列线段 [(x0,y0), (x1,y1)]
    segs1 = []
    for i in range(h):
        # 每一行的点序列: (w, 2)
        row_pts = np.stack((U[i, :], V[i, :]), axis=1)
        # 创建线段: p0->p1, p1->p2...
        # 构造 (N-1, 2, 2) 的数组
        seg_row = np.stack((row_pts[:-1], row_pts[1:]), axis=1)
        segs1.append(seg_row)
    segs1 = np.concatenate(segs1, axis=0)  # (Total_Segs, 2, 2)

    # 2. 绘制“竖线”（连接图像中每一列的点）
    segs2 = []
    for j in range(w):
        col_pts = np.stack((U[:, j], V[:, j]), axis=1)
        seg_col = np.stack((col_pts[:-1], col_pts[1:]), axis=1)
        segs2.append(seg_col)
    segs2 = np.concatenate(segs2, axis=0)

    # 添加到绘图
    lc1 = LineCollection(segs1, colors='blue', linewidths=0.5, alpha=0.6, label='Image Rows')
    lc2 = LineCollection(segs2, colors='red', linewidths=0.5, alpha=0.6, label='Image Cols')

    ax.add_collection(lc1)
    ax.add_collection(lc2)

    # 自动调整范围 (虽然理论上是 -1~1，但为了保险起见)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)


def save_intermediate_results(save_dir, epoch, scene_idx, intermediates, meta_phase):
    """
    Args:
        save_dir: 保存路径
        epoch: 当前epoch
        scene_idx: 场景索引
        intermediates: renderer返回的字典
        meta_phase: Metasurface的相位图
    """
    save_path = Path(save_dir) / str(epoch) / f"scene_{scene_idx}"
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Metasurface Phase Map (相位图)
    if meta_phase is not None:
        phase_np = meta_phase.detach().cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(phase_np, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(label='Phase (rad)')
        plt.title(f"1. Metasurface Phase")
        plt.savefig(save_path / "1_meta_phase.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Fourier Transformation (Pattern 360)
    if 'fourier_pattern' in intermediates:
        fourier = intermediates['fourier_pattern']
        if torch.is_tensor(fourier):
            fourier = fourier.detach().cpu().numpy()

        # 维度修正: 确保是 2D (H, W)
        if fourier.ndim == 4: fourier = fourier[0, 0]
        elif fourier.ndim == 3: fourier = fourier[0]
        # else: ndim==2, keep as is

        plt.figure(figsize=(5, 5))
        plt.imshow(np.log(np.abs(fourier) + 1e-8), cmap='magma')
        plt.colorbar(label='Log Intensity')
        plt.title("2. Fourier Pattern (Far-field)")
        plt.savefig(save_path / "2_fourier_transform.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3 & 4. Coordinate Conversion & Replication
    for i, view_name in enumerate(['Front', 'Rear']):
        grid_key = f'view_{i}_grid'  # (B, H, W, 2)
        pat_key = f'view_{i}_pattern_pure'  # (B, 1, H, W) or (B, H, W)

        # --- A. 绘制重参数化后的图案 (Pattern) ---
        if pat_key in intermediates:
            pat = intermediates[pat_key]
            if torch.is_tensor(pat):
                pat_np = pat.detach().cpu().numpy()
            else:
                pat_np = pat

            # [关键修复] 自动维度压缩
            # 目标: 获取 (H, W) 的 2D 数组
            if pat_np.ndim == 4:  # (B, C, H, W)
                img_to_show = pat_np[0, 0]
            elif pat_np.ndim == 3:  # (B, H, W)
                img_to_show = pat_np[0]
            elif pat_np.ndim == 2:  # (H, W)
                img_to_show = pat_np
            else:  # (C, H, W) or others, try taking last two dims
                img_to_show = pat_np[-2:, :]  # Fallback

            if img_to_show.ndim < 2:
                print(f"[WARN] Skipping {view_name} pattern plot due to invalid shape: {pat_np.shape}")
            else:
                plt.figure(figsize=(6, 4))
                plt.imshow(np.log(img_to_show + 1e-8), cmap='magma')  # 结构光图案，使用 log 增强对比度
                plt.colorbar()
                plt.title(f"3. {view_name} Reparameterized Pattern")
                plt.savefig(save_path / f"3_{view_name}_pattern.png", dpi=300, bbox_inches='tight')
                plt.close()

        # --- B. 绘制坐标变换网格 (Grid Distortion - Wireframe) ---
        if grid_key in intermediates:
            grid = intermediates[grid_key]
            if torch.is_tensor(grid):
                grid = grid.detach().cpu().numpy()  # (B, H, W, 2)

            # 取第一个 batch: (H, W, 2)
            if grid.ndim == 4:
                grid_vis = grid[0]
            else:
                grid_vis = grid

            plt.figure(figsize=(6, 6))
            ax = plt.gca()

            # 调用上面的辅助函数绘制网格线
            plot_grid_wireframe(ax, grid_vis, step=32)

            # 画一个单位圆 (视场边界)
            circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', linewidth=1.5, label='FOV Limit (NA=1)')
            ax.add_artist(circle)

            plt.title(f"4. {view_name} Coord Conversion Grid\n(Blue=Rows, Red=Cols)")
            plt.xlabel("Fourier Space U")
            plt.ylabel("Fourier Space V")
            plt.legend(loc='upper right', fontsize='small')
            plt.gca().set_aspect('equal')
            plt.savefig(save_path / f"4_{view_name}_grid_distortion.png", dpi=300, bbox_inches='tight')
            plt.close()


def train(opt, model, dataset_path):
    dataset_train = CreateSyntheticDataset(opt.train_path, 'train')
    dataset_test = CreateSyntheticDataset(opt.valid_path, 'valid')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    dataloader_valid = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

    if isinstance(model, torch.nn.DataParallel):
        if opt.phase_layer_type == 'DinerPhaseLayer':
            optimizer_meta = torch.optim.Adam(model.module.metasurface.phase_layer.parameters(), lr=opt.lr)
        elif opt.phase_layer_type == 'Parameters':
            optimizer_meta = torch.optim.Adam([model.module.metasurface.phase_layer], lr=opt.lr)
        else:
            raise ValueError("Unsupported phase_layer_type")
    else:
        if opt.phase_layer_type == 'DinerPhaseLayer':
            optimizer_meta = torch.optim.Adam(model.metasurface.phase_layer.parameters(), lr=opt.lr)
        elif opt.phase_layer_type == 'Parameters':
            optimizer_meta = torch.optim.Adam([model.metasurface.phase_layer], lr=opt.lr)
        else:
            raise ValueError("Unsupported phase_layer_type")

    optimizer_net = torch.optim.Adam(list(model.parameters()), lr=opt.lr)

    scheduler_meta = torch.optim.lr_scheduler.StepLR(optimizer_meta, step_size=200, gamma=0.2)
    scheduler_net = torch.optim.lr_scheduler.StepLR(optimizer_net, step_size=350, gamma=0.4)

    l1_loss = torch.nn.L1Loss()
    l1_loss.requires_grad = True

    model = model.to(opt.device)
    writer = SummaryWriter(log_dir=opt.log)
    fisheye_mask = torch.from_numpy(np.load("./fisheye_mask.npy")).to(opt.device)

    min_valid_loss = float('inf')  # 初始化最小 validation loss 为正无穷
    min_valid_loss_epoch = -1  # 用于记录最小 validation loss 对应的 epoch
    min_valid_l1_loss = float('inf')  # 初始化最小验证 L1 损失为正无穷
    min_valid_l1_loss_epoch = -1  # 用于记录最小验证 L1 损失对应的 epoch

    best_model_path = None
    best_phase_path = None

    train_loss_history = []  # 训练损失列表
    valid_loss_history = []  # 验证损失列表
    train_l1_loss_history = []  # 训练 L1 损失列表
    valid_l1_loss_history = []  # 验证 L1 损失列表

    # loss_plot_dir = create_loss_plot_dir()
    for epoch in range(1000):
        # 每个 epoch 开始前清理 GPU 缓存
        torch.cuda.empty_cache()  # 关键！释放未使用的缓存

        losses = []
        l1_losses = []
        model.train()
        # minibatch
        for i, data in enumerate(dataloader_train):
            B = opt.batch_size

            ref_im_list = data['ref_im_list']
            depth_map_list = data['depth_im_list']
            occ_im_list = data['occ_im_list']
            normal_im_list = data['normal_im_list']

            gt = 1.0 / (depth_map_list[0].to(device).float() * 10)

            # 训练阶段不需要 return_intermediates，保持速度
            inv_depth_pred, _ = model(ref_im_list, depth_map_list, occ_im_list, normal_im_list)

            front_l1loss = l1_loss(gt[:, fisheye_mask], inv_depth_pred[0][:, fisheye_mask])
            front_tvloss = grad_loss(gt, inv_depth_pred[0])

            loss = front_l1loss + front_tvloss * 0.4  # + 0.01 * illum_loss
            losses.append(loss.item())
            l1_losses.append(front_l1loss.item())

            if optimizer_meta:
                optimizer_meta.zero_grad()
            if optimizer_net:
                optimizer_net.zero_grad()
            loss.backward()
            if optimizer_meta:
                optimizer_meta.step()
            if optimizer_net:
                optimizer_net.step()

        # 计算当前 epoch 的平均训练 loss
        avg_train_loss = sum(losses) / len(losses)
        avg_train_l1_loss = sum(l1_losses) / len(l1_losses)
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/L1_Loss', avg_train_l1_loss, epoch)
        train_loss_history.append(avg_train_loss)
        train_l1_loss_history.append(avg_train_l1_loss)

        current_meta_lr = optimizer_meta.param_groups[0]['lr']
        current_net_lr = optimizer_net.param_groups[0]['lr']
        writer.add_scalar('Lr/meta', current_meta_lr, epoch)
        writer.add_scalar('Lr/net', current_net_lr, epoch)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"[{epoch}/1000] epoch - Train loss : {avg_train_loss} - Train l1_loss : {avg_train_l1_loss} ")
        print(f"Current Meta Lr : {current_meta_lr} - Net Lr : {current_net_lr} ")

        # Test
        model.eval()
        losses = []
        valid_l1_losses = []  # 用来存储每个验证 batch 的 L1 损失
        mkdirs(opt.test_save_path / str(epoch))
        with torch.no_grad():
            for j, data in enumerate(dataloader_valid):

                ref_im_list = data['ref_im_list']
                depth_map_list = data['depth_im_list']
                occ_im_list = data['occ_im_list']
                normal_im_list = data['normal_im_list']

                gt = 1.0 / (depth_map_list[0].to(device).float() * 10)

                # 在验证阶段获取中间结果
                inv_depth_pred, synthetic_images, intermediates = model(
                    ref_im_list, depth_map_list, occ_im_list, normal_im_list, return_intermediates=True
                )

                front_l1loss = l1_loss(gt[:, fisheye_mask], inv_depth_pred[0][:, fisheye_mask])
                front_tvloss = grad_loss(gt, inv_depth_pred[0])

                # Save Results
                gt_cpu = gt.cpu().squeeze(0).numpy()
                inv_depth_pred_cpu = inv_depth_pred[0].cpu().squeeze(0).numpy()

                plt.imshow(gt_cpu, cmap='viridis')  # 使用 'viridis' 色图
                plt.colorbar()
                plt.title(f"GT Depth - Scene {j}")
                plt.savefig(os.path.join(opt.test_save_path / str(epoch), f"gt_depth_{j}.png"), dpi=300, bbox_inches='tight')
                plt.close()

                plt.imshow(inv_depth_pred_cpu, cmap='viridis')  # 使用 'viridis' 色图
                plt.colorbar()
                plt.title(f"Predicted Depth - Scene {j}")
                plt.savefig(os.path.join(opt.test_save_path / str(epoch), f"inv_depth_pred_{j}.png"), dpi=300, bbox_inches='tight')
                plt.close()

                if j < 2:  # 仅保存前2个场景的中间态
                    if isinstance(model, torch.nn.DataParallel):
                        meta_phase = model.module.get_meta_phase()
                    else:
                        meta_phase = model.get_meta_phase()
                    save_intermediate_results(opt.test_save_path, epoch, j, intermediates, meta_phase)

                loss = front_l1loss + front_tvloss * 0.4
                losses.append(loss.item())
                valid_l1_losses.append(front_l1loss.item())  # 添加每个验证 batch 的 L1 损失

            avg_valid_loss = sum(losses) / len(losses)
            avg_valid_l1_loss = sum(valid_l1_losses) / len(valid_l1_losses)
            writer.add_scalar('Valid/Loss', avg_valid_loss, epoch)
            writer.add_scalar('Valid/L1_Loss', avg_valid_l1_loss, epoch)
            valid_loss_history.append(avg_valid_loss)
            valid_l1_loss_history.append(avg_valid_l1_loss)  # 保存验证 L1 损失

            # 检查当前 epoch 是否是最小验证 loss
            if avg_valid_loss < min_valid_loss:
                min_valid_loss = avg_valid_loss
                min_valid_loss_epoch = epoch

            # 更新最小验证 L1 损失
            if avg_valid_l1_loss < min_valid_l1_loss:
                min_valid_l1_loss = avg_valid_l1_loss
                min_valid_l1_loss_epoch = epoch

                # 删除旧的最佳模型（若存在）
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                if best_phase_path is not None and os.path.exists(best_phase_path):
                    os.remove(best_phase_path)

                # 保存新的最佳模型
                best_model_path = os.path.join(opt.output_dir, f"best_epoch_{epoch:03d}.pth")
                best_phase_path = os.path.join(opt.output_dir, f"phase_best_epoch_{epoch:03d}.npy")

                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), best_model_path)
                    np.save(best_phase_path, model.module.get_meta_phase().detach().cpu().numpy())
                else:
                    torch.save(model.state_dict(), best_model_path)
                    np.save(best_phase_path, model.get_meta_phase().detach().cpu().numpy())

            print("[{0}/1000] epoch - validation loss : {1} - validation l1_loss : {2}".format(epoch, avg_valid_loss, avg_valid_l1_loss))

            # 保存训练和验证损失到文件
            np.save(os.path.join(opt.log, "train_loss_history.npy"), np.array(train_loss_history))
            np.save(os.path.join(opt.log, "valid_loss_history.npy"), np.array(valid_loss_history))
            np.save(os.path.join(opt.log, "train_l1_loss_history.npy"), np.array(train_l1_loss_history))
            np.save(os.path.join(opt.log, "valid_l1_loss_history.npy"), np.array(valid_l1_loss_history))

        print(colored(f"Minimum validation loss: {min_valid_loss} at epoch {min_valid_loss_epoch}", 'yellow', attrs=['bold']))
        print(colored(f"Minimum validation L1 loss: {min_valid_l1_loss} at epoch {min_valid_l1_loss_epoch}", 'yellow', attrs=['bold']))

        if epoch % 10 == 0:
            # torch.save(model.state_dict(), os.path.join(opt.log, "model_epoch_%d.pth"%(epoch)))
            save_path = os.path.join(opt.chk_path, f"model_epoch_{epoch}.pth")
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
                np.save(os.path.join(opt.chk_path, "phase_epoch_%d.npy" % (epoch)), model.module.get_meta_phase().detach().cpu().numpy())
            else:
                torch.save(model.state_dict(), save_path)
                np.save(os.path.join(opt.chk_path, "phase_epoch_%d.npy" % (epoch)), model.get_meta_phase().detach().cpu().numpy())

    writer.close()


if __name__ == "__main__":
    setup_seed(3407)

    # for fair comparison, seeds 0217, 0415, 1227, 3407  are tested.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 减少内存碎片

    parser = Argument()
    parser.parser.add_argument('--use_extrinsic', type=bool, default=False)

    args = parser.parse()
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    mkdirs(args.output_dir)
    mkdirs(args.output_dir / "log")
    mkdirs(args.output_dir / "inference")
    mkdirs(args.output_dir / "checkpoint")

    device = torch.device(args.device)

    metasurface = Metasurface(args, device)

    radian_90 = math.radians(90)

    cam1 = FisheyeCam(args, (0.05, 0.05, 0), (radian_90, 0, 0), 'cam1', device, args.cam_config_path)
    cam2 = FisheyeCam(args, (-0.05, 0.05, 0), (radian_90, 0, 0), 'cam2', device, args.cam_config_path)

    # Front-back / in training time, we just trained cam1-cam2 front system.
    cam_calib = nn.ModuleList([cam1, cam2])

    renderer = ActiveStereoRenderer(args, metasurface, cam_calib, device)
    pano_cam = PanoramaCam(args, (0, 0, 0), (radian_90, 0, 0), 'pano', device)
    estimator = DepthEstimator(pano_cam, cam_calib, device, args)

    e2e_model = E2E(metasurface, renderer, estimator)
    # ✅ 启用多卡训练
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        e2e_model = torch.nn.DataParallel(e2e_model)

    train(args, e2e_model, args.input_path)

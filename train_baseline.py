import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.ArgParser import Argument
from dataset.dataset import CreateSyntheticDataset
from model.Metasurface import Metasurface
from utils.Camera import *
from model.utils.setup_seed import setup_seed
from Image_formation.renderer import *
from model.StereoMatching_copy import DepthEstimator
from termcolor import colored
import scipy.io
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from torch.autograd import Variable

import GPUtil, os
from model.e2e import *


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


# def create_loss_plot_dir(base_dir='loss_plot'):
#     """创建带时间戳的保存文件夹"""
#     from datetime import datetime
#     import os
#
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     folder = os.path.join(base_dir, f'run_{timestamp}')
#     os.makedirs(folder, exist_ok=True)
#     return folder
#
# def save_loss_curve(train_loss, valid_loss, epoch, save_dir):
#     """保存当前 epoch 的训练/验证 loss 曲线"""
#     import matplotlib.pyplot as plt
#     import os
#
#     fig, ax = plt.subplots()
#     ax.plot(train_loss, label='Train Loss', color='blue')
#     ax.plot(valid_loss, label='Validation Loss', color='red')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Loss')
#     ax.set_title('Training and Validation Loss vs Epoch')
#     ax.legend()
#     fig_path = os.path.join(save_dir, f'loss_epoch_{epoch}.png')
#     fig.savefig(fig_path)
#     plt.close(fig)
#     return fig_path


def train(opt, model, dataset_path):
    dataset_train = CreateSyntheticDataset(opt.train_path, 'train')  # path
    dataset_test = CreateSyntheticDataset(opt.valid_path, 'valid')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    dataloader_valid = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

    # meta_phase = torch.autograd.Variable(model.get_meta_phase(), requires_grad=True)
    # ✅ 判断 get_meta_phase 要从 .module 获取
    # if isinstance(model, torch.nn.DataParallel):
    #     meta_phase = torch.autograd.Variable(model.module.get_meta_phase(), requires_grad=True)
    # else:
    #     meta_phase = torch.autograd.Variable(model.get_meta_phase(), requires_grad=True)

    # optimizer_meta = torch.optim.Adam([meta_phase], lr=opt.lr)
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

    # 使用 id() 来排除 phase_layer 的参数
    # optimizer_net = torch.optim.Adam(
    #     [p for p in model.parameters() if
    #      id(p) not in [id(param) for param in model.module.metasurface.phase_layer.parameters()]],
    #     lr=opt.lr
    # )

    scheduler_meta = torch.optim.lr_scheduler.StepLR(optimizer_meta, step_size=200, gamma=0.2)
    scheduler_net = torch.optim.lr_scheduler.StepLR(optimizer_net, step_size=350, gamma=0.4)

    l1_loss = torch.nn.L1Loss()
    l1_loss.requires_grad = True

    # device = torch.device(opt.device)
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

            # # update meta-surface phase
            # if optimizer_meta:
            #     # model.update_phase(meta_phase)
            #     if isinstance(model, torch.nn.DataParallel):
            #         model.module.update_phase(opt)
            #     else:
            #         model.update_phase(opt)

            gt = 1.0 / (depth_map_list[0].to(device).float() * 10)
            inv_depth_pred, synthetic_images = model(ref_im_list, depth_map_list, occ_im_list, normal_im_list)

            front_l1loss = l1_loss(gt[:, fisheye_mask], inv_depth_pred[0][:, fisheye_mask])
            front_tvloss = grad_loss(gt, inv_depth_pred[0])

            # pattern loss
            #pattern = model.get_pattern()
            #illum = torch.nn.functional.grid_sample(pattern.repeat(1, 1, 1, 1), grid, mode='bilinear', padding_mode='zeros').squeeze(0).squeeze(0)
            #illum_loss = 1 / illum_tv(illum / illum.max())

            loss = front_l1loss + front_tvloss * 0.4  # + 0.01 * illum_loss
            # print("{0}th iter : {1}".format(i, loss.item()))
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
        train_loss_history.append(avg_train_loss)
        train_l1_loss_history.append(avg_train_l1_loss)  # 保存每个 epoch 的 L1 损失

        current_meta_lr = optimizer_meta.param_groups[0]['lr']
        current_net_lr = optimizer_meta.param_groups[0]['lr']
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("[{0}/1000] epoch - Train loss : {1} - Train l1_loss : {2} ".format(epoch, avg_train_loss, avg_train_l1_loss))
        print("Current Meta Lr : {0} - Net Lr : {1} ".format(current_meta_lr, current_net_lr))
        # 项目原始代码中并未进行更新
        # After the training loop for each epoch
        # scheduler_meta.step()
        # scheduler_net.step()

        # Test
        model.eval()
        losses = []
        valid_l1_losses = []  # 用来存储每个验证 batch 的 L1 损失

        mkdirs(opt.test_save_path / str(epoch))

        with torch.no_grad():
            for j, data in enumerate(dataloader_valid):
                # B = opt.batch_size

                ref_im_list = data['ref_im_list']
                depth_map_list = data['depth_im_list']
                occ_im_list = data['occ_im_list']
                normal_im_list = data['normal_im_list']

                gt = 1.0 / (depth_map_list[0].to(device).float() * 10)
                inv_depth_pred, _ = model(ref_im_list, depth_map_list, occ_im_list, normal_im_list)

                front_l1loss = l1_loss(gt[:, fisheye_mask], inv_depth_pred[0][:, fisheye_mask])
                front_tvloss = grad_loss(gt, inv_depth_pred[0])

                #----------------- Save Results Begin ----------------#
                gt_cpu = gt.cpu().squeeze(0).numpy()
                inv_depth_pred_cpu = inv_depth_pred[0].cpu().squeeze(0).numpy()

                plt.imshow(gt_cpu, cmap='viridis')  # 使用 'viridis' 色图
                plt.colorbar()
                plt.title(f"GT Depth - Scene {j}")
                plt.savefig(os.path.join(opt.test_save_path / str(epoch), f"gt_depth_{j}.png"))
                plt.close()

                plt.imshow(inv_depth_pred_cpu, cmap='viridis')  # 使用 'viridis' 色图
                plt.colorbar()
                plt.title(f"Predicted Depth - Scene {j}")
                plt.savefig(os.path.join(opt.test_save_path / str(epoch), f"inv_depth_pred_{j}.png"))
                plt.close()
                # ----------------- Save Results End ----------------#

                loss = front_l1loss + front_tvloss * 0.4
                losses.append(loss.item())
                valid_l1_losses.append(front_l1loss.item())  # 添加每个验证 batch 的 L1 损失

            avg_valid_loss = sum(losses) / len(losses)
            avg_valid_l1_loss = sum(valid_l1_losses) / len(valid_l1_losses)
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

        print(colored(f"Minimum validation loss: {min_valid_loss} at epoch {min_valid_loss_epoch}", 'yellow', attrs=['bold']))
        print(colored(f"Minimum validation L1 loss: {min_valid_l1_loss} at epoch {min_valid_l1_loss_epoch}", 'yellow', attrs=['bold']))

        # ✅ 可视化函数 根据history保存训练曲线
        # png_path = save_loss_curve(train_loss_history, valid_loss_history, epoch, loss_plot_dir)
        # print(colored(f"Loss curve saved to: {png_path}", 'cyan'))

        if epoch % 10 == 0:
            # torch.save(model.state_dict(), os.path.join(opt.log, "model_epoch_%d.pth"%(epoch)))
            save_path = os.path.join(opt.chk_path, f"model_epoch_{epoch}.pth")
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
                np.save(os.path.join(opt.chk_path, "phase_epoch_%d.npy" % (epoch)), model.module.get_meta_phase().detach().cpu().numpy())
            else:
                torch.save(model.state_dict(), save_path)
                np.save(os.path.join(opt.chk_path, "phase_epoch_%d.npy" % (epoch)), model.get_meta_phase().detach().cpu().numpy())

    # 保存训练和验证损失到文件
    np.save(os.path.join(opt.log, "train_loss_history.npy"), np.array(train_loss_history))
    np.save(os.path.join(opt.log, "valid_loss_history.npy"), np.array(valid_loss_history))
    np.save(os.path.join(opt.log, "train_l1_loss_history.npy"), np.array(train_l1_loss_history))
    np.save(os.path.join(opt.log, "valid_l1_loss_history.npy"), np.array(valid_l1_loss_history))


if __name__ == "__main__":
    setup_seed(3407)

    # for fair comparison, seeds 0217, 0415, 1227, 3407  are tested.
    # 添加环境变量设置（关键！需在任何CUDA操作前执行）
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

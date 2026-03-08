import torch
import torch.nn as nn
from utils.ArgParser import Argument
from dataset.dataset import *
from model.Metasurface import Metasurface
from utils.Camera import *
from Image_formation.renderer import *
from model.StereoMatching import DepthEstimator
from model.e2e import *


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


def test(opt, model, dataset_path):
    # if opt.test_real_scene :
    #     dataset_test = RealDataset(opt.test_path, 'test')
    # else:
    #     dataset_test = CreateSyntheticDataset(opt.test_path, 'test')

    dataset_test = CreateSyntheticDataset(opt.test_path, 'valid')
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

    l1_loss = torch.nn.L1Loss()
    l1_loss.requires_grad = False

    l2_loss = torch.nn.MSELoss()
    l2_loss.requires_grad = False

    device = torch.device(opt.device)
    model = model.to(device)
    # model.load_state_dict(torch.load('runs/Origin/best_epoch_818.pth'))
    model.load_state_dict(torch.load('runs/DINER/best_epoch_417.pth'))
    # model.load_state_dict(torch.load('runs/251217-201154/best_epoch_389.pth'))
    # model.load_state_dict(torch.load('runs/251217-201145/best_epoch_645.pth'))

    fisheye_mask = torch.from_numpy(np.load("./fisheye_mask.npy")).to(device)
    model.eval()

    # if opt.test_real_scene : # captured by your real system
    #     with torch.no_grad():
    #         for i, data in enumerate(dataloader_test):
    #
    #             img_list = data['img']
    #
    #             for j, img in enumerate(img_list):
    #                 img_list[j] = img.to(device)
    #
    #             result = model.estimator(img_list)[0]
    #             name = data['name']
    #
    #             B = result.shape[0]
    #
    #             # TODO: saving process
    #             for b in range(B):
    #                 plt.imsave(os.path.join(opt.test_save_path, name[b] + '.png'), result[b].cpu().numpy(),  cmap='inferno')

    with torch.no_grad():
        losses = []
        # inv_losses = []
        valid_l1_losses = []
        valid_tv_losses = []
        valid_l2_losses = []

        for i, data in enumerate(dataloader_test):
            B = opt.batch_size

            ref_im_list = data['ref_im_list']
            depth_map_list = data['depth_im_list']
            occ_im_list = data['occ_im_list']
            normal_im_list = data['normal_im_list']

            gt = 1.0 / (depth_map_list[0].to(device).float() * 10)
            inv_depth_pred, _ = model(ref_im_list, depth_map_list, occ_im_list, normal_im_list)

            front_l1loss = l1_loss(gt[:, fisheye_mask], inv_depth_pred[0][:, fisheye_mask])
            front_l2loss = l2_loss(gt[:, fisheye_mask], inv_depth_pred[0][:, fisheye_mask])
            front_tvloss = grad_loss(gt, inv_depth_pred[0])

            loss = front_l1loss + front_tvloss * 0.4
            losses.append(loss.item())
            valid_l1_losses.append(front_l1loss.item())  # 添加每个验证 batch 的 L1 损失
            valid_l2_losses.append(front_l2loss.item())
            valid_tv_losses.append(front_tvloss.item())

        avg_valid_loss = sum(losses) / len(losses)
        avg_valid_l1_loss = sum(valid_l1_losses) / len(valid_l1_losses)
        avg_valid_l2_loss = sum(valid_l2_losses) / len(valid_l2_losses)
        avg_valid_tv_loss = sum(valid_tv_losses) / len(valid_tv_losses)

        print((
            f"avg_valid_loss    = {avg_valid_loss:.6f}\n"
            f"avg_valid_l1_loss = {avg_valid_l1_loss:.6f}\n"
            f"avg_valid_l2_loss = {avg_valid_l2_loss:.6f}\n"
            f"avg_valid_tv_loss = {avg_valid_tv_loss:.6f}"
        ))

        # saving predicted inverse depth
        #np.save(os.path.join(opt.test_save_path, 'ours_%d.npy'%(i)), inv_depth_pred[0].cpu().numpy())


if __name__ == "__main__":
    parser = Argument()
    # parser.parser.add_argument('--test_real_scene', type=bool, default=False)
    parser.parser.add_argument('--use_extrinsic', type=bool, default=False)
    args = parser.parse()

    device = torch.device(args.device)
    metasurface = Metasurface(args, device)

    # TODO: for Origin runs/Origin/phase_best_epoch_818.npy
    # optimized_phase = np.load('runs/Origin/phase_best_epoch_818.npy')
    # metasurface.update_phase(torch.from_numpy(optimized_phase).float().to(device))

    radian_90 = math.radians(90)
    cam1 = FisheyeCam(args, (0.05, 0.05, 0), (radian_90, 0, 0), 'cam1', device, args.cam_config_path)
    cam2 = FisheyeCam(args, (-0.05, 0.05, 0), (radian_90, 0, 0), 'cam2', device, args.cam_config_path)
    cam_calib = nn.ModuleList([cam1, cam2])

    # if args.use_extrinsic: # for real_scene
    #
    #     cam2_ext = torch.from_numpy(np.load(args.front_right_config)).to(device).float()
    #     cam4_ext = torch.from_numpy(np.load(args.back_right_config)).to(device).float()
    #
    #     cam1.set_extrinsic(torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(device))
    #     cam3.set_extrinsic(torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(device))
    #     cam2.set_extrinsic(cam2_ext)
    #     cam4.set_extrinsic(cam4_ext)
    #
    #     cam_calib = [cam1, cam2, cam3, cam4]

    renderer = ActiveStereoRenderer(args, metasurface, cam_calib, device)
    pano_cam = PanoramaCam(args, (0, 0, 0), (radian_90, 0, 0), 'pano', device)
    estimator = DepthEstimator(pano_cam, cam_calib, device, args)

    e2e_model = E2E(metasurface, renderer, estimator)

    # # ✅ 启用多卡训练
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     e2e_model = torch.nn.DataParallel(e2e_model)

    print(
        f"Testing with Physics Config: Legacy={getattr(args, 'use_legacy_physics', False)}, LaserPower={getattr(args, 'laser_power', 'Unknown')}"
    )
    test(args, e2e_model, args.input_path)

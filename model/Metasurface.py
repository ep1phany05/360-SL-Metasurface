import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .diner import DINER
from .inr_params import *


class DinerPhaseLayer(nn.Module):

    def __init__(self, opt, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.N_phase = opt.N_phase

        self.phase_pixels = int(self.N_phase * self.N_phase)
        self.diner = DINER(
            in_features=1,
            out_features=1,
            hidden_features=HIDDEN_FEATURES,
            hidden_layers=HIDDEN_LAYERS,
            hash_table_length=self.phase_pixels,
            first_omega_0=FIRST_OMEGA_0,
            hidden_omega_0=HIDDEN_OMEGA_0
        ).to(
            device, dtype=torch.float32
        )

        Logger.print_and_write("INFO", "DINER", f"Model structure: {HIDDEN_FEATURES} x {HIDDEN_LAYERS}")
        Logger.print_and_write("INFO", "DINER", f"First omega: {FIRST_OMEGA_0} Hidden omega: {HIDDEN_OMEGA_0}")

    def forward(self):
        return self.diner(None)["model_out"].view(self.N_phase, self.N_phase)


class Metasurface(nn.Module):

    def __init__(self, opt, device):
        super().__init__()
        self.opt = opt
        self.device = device
        self.N_theta = opt.N_theta
        self.N_alpha = opt.N_alpha
        self.N_phase = opt.N_phase

        self.phase_type = opt.phase_layer_type
        self.wl = opt.wave_length
        self.p = opt.pixel_pitch

        if self.phase_type == 'Parameters':
            phase_init = torch.rand(self.N_phase, self.N_phase) * 2 * torch.pi - torch.pi
            self.phase_layer = nn.Parameter(phase_init.to(device))
        elif self.phase_type == 'DinerPhaseLayer':
            self.phase_layer = DinerPhaseLayer(opt, device)
        else:
            raise ValueError(f"Unknown phase_layer_type: {self.phase_type}")

    def get_phase(self):
        # shape [N, N]
        if self.phase_type == 'Parameters':
            # parameters范围无约束，归一化到 0-1 对应 [-pi, pi]
            return torch.sigmoid(self.phase_layer) * 2 * torch.pi - torch.pi
        elif self.phase_type == 'DinerPhaseLayer':
            # diner 输出[-1, 1] 对应相位 [-pi, pi]
            return self.phase_layer() * torch.pi  # 注意单位归一，这里乘π
        else:
            raise RuntimeError("phase_layer_type not recognized")

    def propagate(self):
        phase = self.get_phase()
        field = torch.exp(1j * phase)
        if self.opt.use_legacy_physics:  # origin: 无归一化，数值随分辨率爆炸
            spectrum = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field)))
        else:  # new: 正交归一化，能量守恒
            spectrum = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field), norm='ortho'))
        return torch.abs(spectrum) ** 2

        # # self.phase = torch.rand(self.N_phase, self.N_phase).to(device) * torch.pi*2 - torch.pi # initialization
        # self.phase = torch.rand(self.N_phase, self.N_phase) * torch.pi * 2 - torch.pi
        #
        # # Choice 1. 将 phase 注册为可训练参数（使用 nn.Parameter）
        # if self.phase_layer_type == 'Parameters':
        #     phase_init = torch.rand(self.N_phase, self.N_phase) * torch.pi * 2 - torch.pi
        #     self.phase_layer = nn.Parameter(phase_init.to(device))
        #
        # elif self.phase_layer_type == 'DinerPhaseLayer':
        # # Choice 2. 将 phase 注册为Diner输出（使用 DinerPhaseLayer）
        #     self.phase_layer = DinerPhaseLayer(opt, device)

    # def propagate(self):
    #     return torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(torch.exp(1j * self.phase))))) ** 2

    def update_phase(self, new_phase):
        with torch.no_grad():
            if self.phase_type == 'Parameters':
                self.phase_layer.copy_(new_phase)
            elif self.phase_type == 'DinerPhaseLayer':
                self.phase_layer.copy_(new_phase) * torch.pi

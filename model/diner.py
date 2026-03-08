import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super(SineLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

    def forward_with_intermediate(self, x):
        intermediate = self.omega_0 * self.linear(x)
        return torch.sin(intermediate), intermediate


class DINER(nn.Module):

    def __init__(
        self,
        hash_mod: bool = True,  # Whether to use hash table modulation.
        hash_table_length: int = 512 * 512,  # Length of the hash table.
        in_features: int = 3,  # Number of input features.
        hidden_features: int = 64,  # Number of hidden units in each layer.
        hidden_layers: int = 2,  # Number of hidden layers.
        out_features: int = 3,  # Number of output features.
        outermost_linear: bool = True,  # Whether the final layer is linear or a sine layer.
        first_omega_0: float = 6.0,  # Frequency parameter for the first sine layer.
        hidden_omega_0: float = 6.0  # Frequency parameter for the hidden sine layers.
    ):
        super(DINER, self).__init__()
        self.hash_mod = hash_mod
        self.table = nn.Parameter(1e-4 * (torch.rand((hash_table_length, in_features)) * 2 - 1), requires_grad=True)

        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        layers += [SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0) for _ in range(hidden_layers)]

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(SineLayer(hidden_features, out_features, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        if self.hash_mod:
            output = self.net(self.table)
        else:
            output = self.net(coords)
        output = torch.clamp(output, min=-1.0, max=1.0)
        return {"model_out": output, "table": self.table}

    def load_pretrained(self, model_path, device=None):
        if device:
            self.to(device)
            checkpoint = torch.load(model_path, map_location=device)
        else:
            checkpoint = torch.load(model_path, map_location="cpu")
        self.load_state_dict(checkpoint["net"])

import math
import torch
import torch.nn as nn
import numpy as np

class FourierFeatureLayer(nn.Module):
    def __init__(self, in_features, mapping_size=512, scale=(10.0, 10.0, 50.0), inject_freq=True):
        super().__init__()
        self.mapping_size = mapping_size
        B = torch.randn(in_features, mapping_size)

        # Anisotropic Scaling
        sigma_z, sigma_x, sigma_t = scale
        B[0, :] *= sigma_z 
        B[1, :] *= sigma_x 
        B[2, :] *= sigma_t 

        # Frequency Injection
        if inject_freq:
            n_inject = int(0.25 * mapping_size) 
            B[0, :n_inject] = 0.0
            B[1, :n_inject] = 0.0
            B[2, :n_inject] = torch.empty(n_inject).uniform_(100.0, 500.0)

        self.register_buffer("B", B) 

    def forward(self, x):
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SpeedNet(nn.Module):
    def __init__(self, in_features=2, hidden_layers=[64, 64, 64, 64], fourier_scales=(550.0, 70.0)):
        super().__init__()
        self.mapping_size = 128
        B = torch.randn(in_features, self.mapping_size)
        B[0, :] *= fourier_scales[0] 
        B[1, :] *= fourier_scales[1] 
        self.register_buffer("B", B)

        input_dim = self.mapping_size * 2
        layers = []
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.SiLU())
            input_dim = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, 1)

        self.c_min = 750.0
        self.c_max = 1750.0
        self._init_bias(target_c=1450.0)

    def _init_bias(self, target_c):
        y_norm = (target_c - self.c_min) / (self.c_max - self.c_min)
        bias_val = np.log(y_norm / (1.0 - y_norm))
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, bias_val)

    def forward(self, z, x):
        inp = torch.cat([z, x], dim=-1)
        x_proj = (2.0 * np.pi * inp) @ self.B
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        psi = self.net(features)
        psi = self.head(psi)
        return self.c_min + (self.c_max - self.c_min) * torch.sigmoid(psi)

class WaveFieldNet(nn.Module):
    def __init__(self, num_angles=3, hidden_layers=[512, 512, 512, 512, 512, 512]):
        super().__init__()
        self.fourier = FourierFeatureLayer(
            in_features=3, mapping_size=1024, scale=(350.0, 200.0, 350.0), inject_freq=True
        )
        feat_dim = 1024 * 2
        embed_dim = 32
        self.angle_embed = nn.Embedding(num_embeddings=num_angles, embedding_dim=embed_dim)

        input_dim = feat_dim + embed_dim
        layers = []
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.Tanh())
            input_dim = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z, x, t, angle_idx):
        coords = torch.cat([z, x, t], dim=-1)
        coord_feats = self.fourier(coords)
        angle_feats = self.angle_embed(angle_idx.reshape(-1))
        combined = torch.cat([coord_feats, angle_feats], dim=-1)
        return self.head(self.net(combined))

class PINN_Architecture(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.speed_net = SpeedNet()
        self.wave_net = WaveFieldNet(num_angles=len(config.target_angles))

    def forward(self, z, x, t, angle_idx):
        c_pred = self.speed_net(z, x)
        u_pred = self.wave_net(z, x, t, angle_idx)
        return u_pred, c_pred

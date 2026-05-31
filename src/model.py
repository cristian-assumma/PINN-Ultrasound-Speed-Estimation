import math
import torch
import torch.nn as nn
import numpy as np

class FourierFeatureLayer(nn.Module):
    """
    Advanced Fourier Feature mapping with Anisotropic Scaling and targeted Frequency Injection.
    Designed to mitigate spectral bias by explicitly forcing a subset of neurons to resonate 
    within the high-frequency ultrasound band.
    """
    def __init__(self, in_features, mapping_size=512, scale=(10.0, 10.0, 50.0), inject_freq=True):
        super().__init__()
        self.mapping_size = mapping_size
        
        # Initialize the projection matrix B with a standard Gaussian distribution
        B = torch.randn(in_features, mapping_size)

        # 1. Anisotropic Scaling: Apply specific standard deviations for spatial (z, x) and temporal (t) axes
        sigma_z, sigma_x, sigma_t = scale
        B[0, :] *= sigma_z 
        B[1, :] *= sigma_x 
        B[2, :] *= sigma_t 

        # 2. Targeted Frequency Injection: Force a subset of features to act as pure high-frequency temporal waves
        if inject_freq:
            n_inject = int(0.25 * mapping_size) 
            # Zero out spatial frequencies for the injected subset
            B[0, :n_inject] = 0.0
            B[1, :n_inject] = 0.0
            # Inject high temporal frequencies (normalized range corresponding to ~7.5 MHz)
            B[2, :n_inject] = torch.empty(n_inject).uniform_(100.0, 500.0)

        # Register B as a buffer so it is saved with the model state but not updated by the optimizer
        self.register_buffer("B", B) 

    def forward(self, x):
        # Compute the Fourier basis projection
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SpeedNet(nn.Module):
    """
    A constrained Multi-Layer Perceptron (MLP) estimating the continuous, 
    spatially varying sound speed map c(x,z).
    """
    def __init__(self, in_features=2, hidden_layers=[64, 64, 64, 64], fourier_scales=(550.0, 70.0)):
        super().__init__()
        self.mapping_size = 128
        
        # Spatial-only Fourier embedding for the kinematic parameter map
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

        # Physiological bounds for the sound speed mapping (m/s)
        self.c_min = 750.0
        self.c_max = 1750.0
        
        # Initialize bias to start optimization near the expected soft tissue baseline
        self._init_bias(target_c=1450.0)

    def _init_bias(self, target_c):
        """
        Analytically initializes the output layer bias to ensure the pre-activation 
        sigmoid maps exactly to the target initial sound speed.
        """
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
        
        # Hard constraint mapping to physical sound speed bounds
        return self.c_min + (self.c_max - self.c_min) * torch.sigmoid(psi)

class WaveFieldNet(nn.Module):
    """
    Reconstructs the spatiotemporal scattered pressure field utilizing injected Fourier features 
    and angular embeddings to resolve multi-view insonations.
    """
    def __init__(self, num_angles=3, hidden_layers=[512, 512, 512, 512, 512, 512]):
        super().__init__()
        
        # Spatiotemporal embedding with active frequency injection
        self.fourier = FourierFeatureLayer(
            in_features=3, mapping_size=1024, scale=(350.0, 200.0, 350.0), inject_freq=True
        )
        
        feat_dim = 1024 * 2
        embed_dim = 32
        
        # Latent embedding to condition the network on the specific steering angle
        self.angle_embed = nn.Embedding(num_embeddings=num_angles, embedding_dim=embed_dim)

        input_dim = feat_dim + embed_dim
        layers = []
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.Tanh())
            input_dim = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, 1)
        
        # Crucial for Tanh activations to prevent vanishing gradients
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Xavier Normal initialization tailored for Tanh non-linearities."""
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
    """
    Dual-network architecture coupling the kinematic parameter estimator (SpeedNet) 
    and the physical wavefield solver (WaveFieldNet).
    """
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.speed_net = SpeedNet()
        self.wave_net = WaveFieldNet(num_angles=len(config.target_angles))

    def forward(self, z, x, t, angle_idx):
        c_pred = self.speed_net(z, x)
        u_pred = self.wave_net(z, x, t, angle_idx)
        return u_pred, c_pred

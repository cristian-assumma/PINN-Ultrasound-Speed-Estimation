import torch
import numpy as np
from src.config import GlobalConfig

class PDE_Sampler:
    """
    Campionatore spazio-temporale per PINN. 
    Usa una strategia ibrida (Guided + Uniform) per concentrare
    il calcolo dei residui fisici lungo il fronte d'onda in movimento.
    """
    def __init__(self, config: GlobalConfig):
        self.cfg = config
        self.device = config.device
        
        # Estrazione degli angoli target
        self.angle_names = config.target_angles
        self.angles_deg = [float(a.split('_')[1]) for a in self.angle_names]
        self.angles_tensor = torch.tensor(self.angles_deg, device=self.device, dtype=torch.float32)
        self.num_angles = len(self.angles_deg)

        self.c0 = config.physics.c0
        # Calcolo dinamico di T_acq in base alla griglia
        Nt = getattr(config.physics, 'Nt', 2030)
        self.T_acq = Nt * config.physics.dt_grid

        # Sigma spaziale: copre lo spessore dell'impulso
        duration_phys = 18 * config.physics.dt_grid 
        sigma_t = duration_phys / 4.0
        self.sigma_spatial = self.c0 * sigma_t * 3.0 

        self.z_max = config.physics.z_max
        self.x_min = config.physics.x_min
        self.x_max = config.physics.x_max

    def _normalize(self, z_phys, x_phys, t_phys):
        z_norm = 2.0 * (z_phys / self.z_max) - 1.0
        x_norm = 2.0 * (x_phys - self.x_min) / (self.x_max - self.x_min) - 1.0
        t_norm = 2.0 * (t_phys / self.T_acq) - 1.0
        return z_norm, x_norm, t_norm

    def sample_guided(self, N_guided):
        if N_guided == 0: return None

        angle_indices = torch.randint(0, self.num_angles, (N_guided,), device=self.device)
        theta_vals = self.angles_tensor[angle_indices] 

        t_phys = torch.rand(N_guided, device=self.device) * self.T_acq
        x_phys = torch.rand(N_guided, device=self.device) * (self.x_max - self.x_min) + self.x_min

        theta_rad = torch.deg2rad(theta_vals)
        sin_t = torch.sin(theta_rad)
        cos_t = torch.cos(theta_rad)

        z_wavefront = (self.c0 * t_phys - x_phys * sin_t) / (cos_t + 1e-6)
        jitter = torch.randn(N_guided, device=self.device) * self.sigma_spatial
        z_phys = z_wavefront + jitter

        mask_valid = (z_phys >= 0) & (z_phys <= self.z_max)
        n_invalid = (~mask_valid).sum()
        if n_invalid > 0:
            z_phys[~mask_valid] = torch.rand(n_invalid, device=self.device) * self.z_max

        return self._normalize(z_phys, x_phys, t_phys) + (angle_indices,)

    def sample_uniform(self, N_uniform):
        if N_uniform == 0: return None
        z_norm = torch.rand(N_uniform, device=self.device) * 2.0 - 1.0
        x_norm = torch.rand(N_uniform, device=self.device) * 2.0 - 1.0
        t_norm = torch.rand(N_uniform, device=self.device) * 2.0 - 1.0
        angle_indices = torch.randint(0, self.num_angles, (N_uniform,), device=self.device)
        return z_norm, x_norm, t_norm, angle_indices

    def get_batch(self, total_points=8000, guided_ratio=0.8):
        N_guided = int(guided_ratio * total_points)
        N_uniform = total_points - N_guided

        zg, xg, tg, ag = self.sample_guided(N_guided)
        zu, xu, tu, au = self.sample_uniform(N_uniform)

        z = torch.cat([zg, zu]).unsqueeze(1).requires_grad_(True)
        x = torch.cat([xg, xu]).unsqueeze(1).requires_grad_(True)
        t = torch.cat([tg, tu]).unsqueeze(1).requires_grad_(True)
        a = torch.cat([ag, au]).unsqueeze(1) 

        return z, x, t, a

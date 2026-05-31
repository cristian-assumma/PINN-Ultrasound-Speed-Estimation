import torch
import numpy as np
from src.config import GlobalConfig

class PDE_Sampler:
    """
    Spatiotemporal collocation sampler for the Physics-Informed Neural Network (PINN).
    Implements a hybrid sampling strategy (Guided + Uniform) to concentrate
    the evaluation of physical PDE residuals along the propagating wavefront.
    """
    def __init__(self, config: GlobalConfig):
        self.cfg = config
        self.device = config.device
        
        # Extract target steering angles
        self.angle_names = config.target_angles
        self.angles_deg = [float(a.split('_')[1]) for a in self.angle_names]
        self.angles_tensor = torch.tensor(self.angles_deg, device=self.device, dtype=torch.float32)
        self.num_angles = len(self.angles_deg)

        self.c0 = config.physics.c0
        
        # Dynamically compute total acquisition time based on the grid parameters
        Nt = getattr(config.physics, 'Nt', 2030)
        self.T_acq = Nt * config.physics.dt_grid

        # Spatial sigma: dictates the Gaussian spread to cover the ultrasound pulse thickness
        duration_phys = 18 * config.physics.dt_grid 
        sigma_t = duration_phys / 4.0
        self.sigma_spatial = self.c0 * sigma_t * 3.0 

        self.z_max = config.physics.z_max
        self.x_min = config.physics.x_min
        self.x_max = config.physics.x_max

    def _normalize(self, z_phys, x_phys, t_phys):
        """Normalizes physical coordinates to the [-1, 1] range required by the neural network."""
        z_norm = 2.0 * (z_phys / self.z_max) - 1.0
        x_norm = 2.0 * (x_phys - self.x_min) / (self.x_max - self.x_min) - 1.0
        t_norm = 2.0 * (t_phys / self.T_acq) - 1.0
        return z_norm, x_norm, t_norm

    def sample_guided(self, N_guided):
        """
        Generates collocation points dynamically focused on the analytical moving wavefront.
        """
        if N_guided == 0: return None

        # Randomly assign steering angles to the guided points
        angle_indices = torch.randint(0, self.num_angles, (N_guided,), device=self.device)
        theta_vals = self.angles_tensor[angle_indices] 

        # Uniformly sample time (t) and lateral position (x)
        t_phys = torch.rand(N_guided, device=self.device) * self.T_acq
        x_phys = torch.rand(N_guided, device=self.device) * (self.x_max - self.x_min) + self.x_min

        theta_rad = torch.deg2rad(theta_vals)
        sin_t = torch.sin(theta_rad)
        cos_t = torch.cos(theta_rad)

        # Analytically solve for depth (z) representing the exact wavefront position
        z_wavefront = (self.c0 * t_phys - x_phys * sin_t) / (cos_t + 1e-6)
        
        # Add Gaussian jitter to simulate the physical spatial width of the pulse
        jitter = torch.randn(N_guided, device=self.device) * self.sigma_spatial
        z_phys = z_wavefront + jitter

        # Filter points to ensure they strictly fall within the physical Region of Interest (ROI)
        mask_valid = (z_phys >= 0) & (z_phys <= self.z_max)
        n_invalid = (~mask_valid).sum()
        
        # Reassign out-of-bounds points uniformly across the depth domain
        if n_invalid > 0:
            z_phys[~mask_valid] = torch.rand(n_invalid, device=self.device) * self.z_max

        return self._normalize(z_phys, x_phys, t_phys) + (angle_indices,)

    def sample_uniform(self, N_uniform):
        """Generates purely random collocation points across the entire spatiotemporal domain."""
        if N_uniform == 0: return None
        z_norm = torch.rand(N_uniform, device=self.device) * 2.0 - 1.0
        x_norm = torch.rand(N_uniform, device=self.device) * 2.0 - 1.0
        t_norm = torch.rand(N_uniform, device=self.device) * 2.0 - 1.0
        angle_indices = torch.randint(0, self.num_angles, (N_uniform,), device=self.device)
        return z_norm, x_norm, t_norm, angle_indices

    def get_batch(self, total_points=8000, guided_ratio=0.8):
        """
        Assembles the final hybrid batch (e.g., 80% guided, 20% uniform) 
        and enables gradient tracking for Autograd.
        """
        N_guided = int(guided_ratio * total_points)
        N_uniform = total_points - N_guided

        zg, xg, tg, ag = self.sample_guided(N_guided)
        zu, xu, tu, au = self.sample_uniform(N_uniform)

        # Concatenate and enable gradient computation for spatial and temporal inputs
        z = torch.cat([zg, zu]).unsqueeze(1).requires_grad_(True)
        x = torch.cat([xg, xu]).unsqueeze(1).requires_grad_(True)
        t = torch.cat([tg, tu]).unsqueeze(1).requires_grad_(True)
        
        # Angles are discrete indices used for embeddings; no gradients required
        a = torch.cat([ag, au]).unsqueeze(1) 

        return z, x, t, a

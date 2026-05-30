import torch
import torch.nn as nn
import numpy as np

class AnalyticalIncidentWave(nn.Module):
    """
    Simula l'onda incidente basandosi su una Gabor Wavelet fittata.
    """
    def __init__(self, config, gabor_params):
        super().__init__()
        self.cfg = config
        
        # Parametri Gabor passati esplicitamente
        self.A = gabor_params['A'] / config.norm.U_ref
        self.sigma = gabor_params['sigma']
        self.omega = gabor_params['omega']
        self.phi = gabor_params['phi']
        self.t0 = gabor_params['t0']

        # Pre-calcoliamo la durata totale dell'acquisizione
        # (Se Nt non è in config.physics, lo dedurremo altrove, ma qui assumiamo il fallback)
        Nt = getattr(config.physics, 'Nt', 2030)
        self.T_acq = Nt * config.physics.dt_grid
        self.c0 = config.physics.c0

    def forward(self, x, z, t, theta_deg, angle_idx, t_delays_s_tensor):
        """
        Calcola l'ampiezza dell'onda incidente e la sua derivata seconda temporale (u_tt).
        """
        # ---- Denormalizzazione spazio ----
        z_phys = (z + 1.0) * 0.5 * self.cfg.physics.z_max
        x_width = self.cfg.physics.x_width
        x_min = self.cfg.physics.x_min
        x_phys = (x + 1.0) * 0.5 * x_width + x_min

        # ---- Denormalizzazione tempo ----
        t_phys = (t + 1.0) * 0.5 * self.T_acq

        # ---- Steering ----
        theta_rad = torch.deg2rad(theta_deg)
        dist_proj = x_phys * torch.sin(theta_rad) + z_phys * torch.cos(theta_rad)

        current_delay = t_delays_s_tensor[angle_idx.squeeze().long()].view(-1, 1)

        tau = dist_proj / self.c0 + current_delay
        t_loc = t_phys - tau - self.t0

        # ---- Gabor differenziabile (Autograd-friendly) ----
        g = torch.exp(-(t_loc**2) / (2 * self.sigma**2))
        s = torch.sin(self.omega * t_loc + self.phi)
        u_inc = self.A * g * s

        # Derivate analitiche seconde (u_tt)
        g_dt = -(t_loc / self.sigma**2) * g
        g_dt2 = ((t_loc**2 / self.sigma**4) - (1 / self.sigma**2)) * g

        s_dt = self.omega * torch.cos(self.omega * t_loc + self.phi)
        s_dt2 = -self.omega**2 * s

        u_inc_tt = self.A * (g_dt2 * s + 2 * g_dt * s_dt + g * s_dt2)

        return u_inc, u_inc_tt

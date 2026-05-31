import torch
import torch.nn as nn
import torch.autograd as autograd
from src.config import GlobalConfig

class PINN_Loss(nn.Module):
    def __init__(self, config: GlobalConfig, physics_engine):
        super().__init__()
        self.cfg = config
        self.device = config.device
        self.physics_engine = physics_engine

        # Angles configuration
        self.angles_deg = [float(a.split('_')[1]) for a in config.target_angles]
        self.angles_tensor = torch.tensor(self.angles_deg, device=self.device, dtype=torch.float32)

        # Derivative scaling factors (Chain Rule for [-1, 1] normalized coordinates)
        self.T_acq = physics_engine.T_acq
        self.lambda_t = 2.0 / self.T_acq
        self.lambda_z = 2.0 / config.physics.z_max
        self.lambda_x = 2.0 / config.physics.x_width

        # PDE Normalizer and Loss Weights
        self.S_scale = 1.0e9
        self.data_loss_base_weight = 1.0
        self.data_loss_amp_factor = 50.0

    def diff(self, u, x, order=1):
        """Helper function to compute gradients using PyTorch Autograd."""
        grads = autograd.grad(u, x,
                              grad_outputs=torch.ones_like(u),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
        if order == 1:
            return grads
        grads_2 = autograd.grad(grads, x,
                                grad_outputs=torch.ones_like(grads),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True)[0]
        return grads_2

    def forward(self, model, batch_pde, batch_data, t_delays_s_tensor):
        # --- A. DATA LOSS (Sensor Supervision) ---
        z_d, x_d, t_d, ang_d = batch_data['inputs']
        u_meas = batch_data['targets'] 
        mask = batch_data['mask']      

        u_pred_data, _ = model(z_d, x_d, t_d, ang_d)
        diff = u_pred_data - u_meas
        diff_masked = diff * mask

        # Adaptive Weighting (assigns higher importance to strong echoes)
        weights = self.data_loss_base_weight + self.data_loss_amp_factor * torch.abs(u_meas)
        loss_data = torch.mean(weights * mask * (diff**2))

        epsilon = 1e-8
        rel_l2 = torch.norm(diff_masked) / (torch.norm(u_meas * mask) + epsilon)

        # --- B. PHYSICS LOSS (Acoustic Wave Equation Constraint) ---
        z_p, x_p, t_p, ang_p = batch_pde
        u_sct, c_pred = model(z_p, x_p, t_p, ang_p)

        # Normalized Derivatives
        u_t_hat = self.diff(u_sct, t_p, order=1)
        u_tt_hat = self.diff(u_t_hat, t_p, order=1)
        u_zz_hat = self.diff(self.diff(u_sct, z_p, order=1), z_p, order=1)
        u_xx_hat = self.diff(self.diff(u_sct, x_p, order=1), x_p, order=1)

        # Gradient denormalization to physical units
        u_tt_phys = u_tt_hat * (self.lambda_t ** 2)
        u_zz_phys = u_zz_hat * (self.lambda_z ** 2)
        u_xx_phys = u_xx_hat * (self.lambda_x ** 2)
        laplacian_phys = u_zz_phys + u_xx_phys

        # Analytical Source Term
        theta_batch = self.angles_tensor[ang_p].view(-1, 1)
        _, u_inc_tt_phys = self.physics_engine(x_p, z_p, t_p, theta_batch, ang_p, t_delays_s_tensor)

        # PDE Assembly (Inverse Scattering Formulation)
        inv_c2 = 1.0 / (c_pred ** 2)
        inv_c02 = 1.0 / (self.cfg.physics.c0 ** 2)
        contrast = inv_c2 - inv_c02

        pde_res_phys = laplacian_phys - (inv_c2 * u_tt_phys) - (contrast * u_inc_tt_phys)
        pde_res_norm = pde_res_phys / self.S_scale
        loss_pde = torch.mean(pde_res_norm ** 2)

        # --- C. IC LOSS (Initial Conditions) ---
        mask_ic = (t_p < -0.98).view(-1)
        if mask_ic.sum() > 0:
            u_ic_val = u_sct[mask_ic]
            u_ic_dt_norm = u_t_hat[mask_ic]
            loss_ic = torch.mean(u_ic_val**2) + torch.mean(u_ic_dt_norm**2)
        else:
            loss_ic = torch.tensor(0.0, device=self.device)

        return {
            'loss_data': loss_data,
            'loss_pde': loss_pde,
            'loss_ic': loss_ic,
            'Rel_L2': rel_l2,
            'debug_c_min': c_pred.min().detach(),
            'debug_c_max': c_pred.max().detach(),
            'debug_c_mean': c_pred.mean().detach()
        }

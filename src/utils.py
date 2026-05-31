import os
import torch
import matplotlib.pyplot as plt

def get_next_run_name(base_dir):
    """Automatically manages and generates sequential Run_XX directories for logging."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    existing = [d for d in os.listdir(base_dir) if d.startswith("Run_") and os.path.isdir(os.path.join(base_dir, d))]
    nums = []
    for d in existing:
        try:
            nums.append(int(d.split("_")[1]))
        except ValueError:
            pass
    next_num = max(nums) + 1 if nums else 1
    return f"Run_{next_num:02d}"

def get_data_batch(multi_angle_data, target_angles, config, t_acq, batch_size=2000):
    """Extracts a random spatiotemporal batch from the ground truth RF data, including time-gating masks."""
    device = config.device
    angles = list(multi_angle_data.keys())
    n_per_angle = batch_size // len(angles)

    z_list, x_list, t_list, a_list, u_list, m_list = [], [], [], [], [], []

    z_max, x_min, x_width = config.physics.z_max, config.physics.x_min, config.physics.x_width

    for ang_name in angles:
        data = multi_angle_data[ang_name]
        u_tensor = data['u_sct']
        mask_tensor = data['mask']
        x_axis_phys = data['x_axis']
        t_axis_phys = data['t_axis']

        N_sens = len(x_axis_phys)
        N_time = len(t_axis_phys)

        idx_sens = torch.randint(0, N_sens, (n_per_angle,), device=device)
        idx_time = torch.randint(0, N_time, (n_per_angle,), device=device)

        x_vals_phys = torch.tensor(x_axis_phys, device=device)[idx_sens]
        t_vals_phys = torch.tensor(t_axis_phys, device=device)[idx_time]
        z_vals_phys = torch.zeros_like(x_vals_phys)

        z_norm = 2.0 * (z_vals_phys / z_max) - 1.0
        x_norm = 2.0 * (x_vals_phys - x_min) / x_width - 1.0
        t_norm = 2.0 * (t_vals_phys / t_acq) - 1.0

        real_idx = target_angles.index(ang_name)
        a_vals = torch.full_like(t_norm, real_idx, dtype=torch.long)

        z_list.append(z_norm)
        x_list.append(x_norm)
        t_list.append(t_norm)
        a_list.append(a_vals)
        u_list.append(u_tensor[idx_sens, idx_time])
        m_list.append(mask_tensor[idx_sens, idx_time])

    inputs = (torch.cat(z_list).view(-1,1), torch.cat(x_list).view(-1,1),
              torch.cat(t_list).view(-1,1), torch.cat(a_list).view(-1,1))
    
    return {
        'inputs': inputs, 
        'targets': torch.cat(u_list).view(-1,1), 
        'mask': torch.cat(m_list).view(-1,1)
    }

def debug_gradients(model):
    """Tracks the L2 norm of the gradients to monitor learning dynamics."""
    speed_sum, wave_sum = 0.0, 0.0
    speed_count, wave_count = 0, 0

    for p in model.speed_net.parameters():
        if p.grad is not None:
            speed_sum += p.grad.norm(2).item()
            speed_count += 1
    for p in model.wave_net.parameters():
        if p.grad is not None:
            wave_sum += p.grad.norm(2).item()
            wave_count += 1

    speed_mean = speed_sum / speed_count if speed_count > 0 else 0.0
    wave_mean = wave_sum / wave_count if wave_count > 0 else 0.0
    return speed_mean, wave_mean

def save_diagnostic_plot(model, epoch, run_dir, config, physics_engine, data_dict_multi, history):
    """Generates and saves a comprehensive diagnostic visual panel to disk (headless mode, no plt.show)."""
    model.eval()
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    target_angles = config.target_angles

    # (A) Speed Map
    nz, nx = 200, 128
    z_lin = torch.linspace(-1, 1, nz, device=config.device)
    x_lin = torch.linspace(-1, 1, nx, device=config.device)
    Z, X = torch.meshgrid(z_lin, x_lin, indexing='ij')

    with torch.no_grad():
        c_pred = model.speed_net(Z.reshape(-1, 1), X.reshape(-1, 1)).reshape(nz, nx).cpu().numpy()

    im1 = axs[0, 0].imshow(c_pred, extent=[config.physics.x_min*1000, config.physics.x_max*1000, config.physics.z_max*1000, 0], cmap='jet', vmin=750, vmax=1750, aspect='auto')
    plt.colorbar(im1, ax=axs[0, 0], label='c [m/s]')
    axs[0, 0].set_title(f"Speed Map (Ep {epoch})")

    # (B) Wavefield Snapshot
    t_mid_phys = physics_engine.T_acq / 2.0
    t_mid_norm = 2.0 * (t_mid_phys / physics_engine.T_acq) - 1.0
    with torch.no_grad():
        u_snap = model.wave_net(Z.reshape(-1, 1), X.reshape(-1, 1), torch.full_like(Z, t_mid_norm).reshape(-1, 1), torch.zeros_like(Z, dtype=torch.long).reshape(-1)).reshape(nz, nx).cpu().numpy()
    
    axs[0, 1].imshow(u_snap, aspect='auto', cmap='seismic', vmin=-0.5, vmax=0.5)
    axs[0, 1].set_title(f"Wavefield Angle {target_angles[0]} @ {t_mid_phys*1e6:.1f} us")
    axs[0, 1].axis('off')

    # (C) Loss History
    if len(history['loss_pde']) > 0:
        axs[0, 2].semilogy(history['loss_pde'], label='PDE')
        axs[0, 2].semilogy(history['loss_data'], label='Data')
        axs[0, 2].semilogy(history['loss_ic'], label='IC')
        axs[0, 2].legend()
        axs[0, 2].grid(True, which='both', alpha=0.3)

    # (D) A-scan Compare (Masked)
    for i, ang_name in enumerate(target_angles):
        data = data_dict_multi.get(ang_name)
        if data:
            rf_true = data['u_sct'].cpu().numpy()
            mask_true = data['mask'].cpu().numpy()
            center_idx = rf_true.shape[0] // 2
            
            t_axis_phys = data['t_axis']
            t_tensor_norm = 2.0 * (torch.tensor(t_axis_phys, device=config.device, dtype=torch.float32) / physics_engine.T_acq) - 1.0
            x_center_norm = 2.0 * (0.0 - config.physics.x_min) / config.physics.x_width - 1.0
            
            with torch.no_grad():
                trace_pred_center = model.wave_net(
                    torch.full((len(t_tensor_norm), 1), -1.0, device=config.device),
                    torch.full((len(t_tensor_norm), 1), x_center_norm, device=config.device),
                    t_tensor_norm.view(-1, 1),
                    torch.full((len(t_tensor_norm), 1), i, dtype=torch.long, device=config.device)
                ).cpu().numpy().flatten()

            mask_trace = mask_true[center_idx, :]
            axs[1, i].plot(t_axis_phys*1e6, rf_true[center_idx, :] * mask_trace, 'k', alpha=0.7, label='GT (Masked)')
            axs[1, i].plot(t_axis_phys*1e6, trace_pred_center * mask_trace, 'r--', label='Pred (Masked)')
            axs[1, i].set_title(f"A-Scan {ang_name}")
            axs[1, i].set_ylim(-1.0, 1.0)
            axs[1, i].legend(loc='upper right', fontsize='small')
            axs[1, i].grid(True, alpha=0.3)
        else:
            axs[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plots", f"diag_ep{epoch:04d}.png"), dpi=100)
    plt.close(fig)
    model.train()

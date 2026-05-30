import os
import time
import csv
import torch
import torch.optim as optim

from src.config import GlobalConfig, set_seed
from src.data_loader import UltrasoundDataLoader
from src.physics import AnalyticalIncidentWave
from src.sampler import PDE_Sampler
from src.loss import PINN_Loss
from src.model import PINN_Architecture
from src.utils import get_next_run_name, get_data_batch, debug_gradients, save_diagnostic_plot

GABOR_PARAMS = {
    'A': -0.64956,   
    'sigma': -0.085,
    'omega': 4.7e7, 
    'phi': 2.638,      
    't0': 0.288   
}

def train():
    CONFIG = GlobalConfig(dataset_type="anechoic_cyst", case_name="case_0012")
    set_seed(CONFIG.seed)

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
    run_name = get_next_run_name(base_dir)
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    print(f"=== START TRAINING: {run_name} (Freeze Strategy) ===")

    loader = UltrasoundDataLoader(CONFIG)
    data_dict_multi, gt_tensor, t_delays_s_tensor = loader.load_dataset()

    physics_engine = AnalyticalIncidentWave(CONFIG, GABOR_PARAMS).to(CONFIG.device)
    sampler = PDE_Sampler(CONFIG)
    loss_fn = PINN_Loss(CONFIG, physics_engine)
    pinn_model = PINN_Architecture(CONFIG).to(CONFIG.device)

    optimizer = optim.Adam(pinn_model.parameters(), lr=CONFIG.learning_rate)
    
    history = {'loss_pde': [], 'loss_data': [], 'loss_ic': [], 'Rel_L2': []}
    
    with open(os.path.join(run_dir, "log.csv"), "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Epoch", "Phase", "Loss_Total", "Loss_PDE", "Loss_Data", "Loss_IC",
                         "Rel_L2", "C_min", "C_max", "C_Mean", "Src_Max",
                         "Grad_Ratio", "Speed_Grad_L2_Mean", "Wave_Grad_L2_Mean"])

        best_loss = float('inf')
        speednet_frozen = False
        start_time = time.time()

        for epoch in range(1, CONFIG.train_epochs + 1):
            
            # Curriculum Logic
            if epoch <= 500:
                phase = "Kickstart (Frozen)"
                w_pde, w_data, w_ic = 10.0, 1.0, 0.001
                if not speednet_frozen:
                    for param in pinn_model.speed_net.parameters(): param.requires_grad = False
                    speednet_frozen = True
            elif epoch <= 2000:
                phase = "Injection (Unfrozen)"
                prog = (epoch - 500) / 1500
                w_pde = 1.0
                w_data = 50.0 + 2000.0 * prog
                w_ic = 0.0010 * (1 - prog) + 0.0002 * prog
                if speednet_frozen:
                    for param in pinn_model.speed_net.parameters(): param.requires_grad = True
                    speednet_frozen = False
            else:
                phase = "Refinement"
                w_pde, w_data, w_ic = 1.0, 2000.0, 0.0002

            batch_pde = sampler.get_batch(total_points=CONFIG.batch_size_collocation)
            batch_data = get_data_batch(data_dict_multi, CONFIG.target_angles, CONFIG, physics_engine.T_acq, batch_size=CONFIG.batch_size_data)

            optimizer.zero_grad()
            losses = loss_fn(pinn_model, batch_pde, batch_data, t_delays_s_tensor)
            
            loss_total = w_pde * losses['loss_pde'] + w_data * losses['loss_data'] + w_ic * losses['loss_ic']
            loss_total.backward()

            speed_grad_mean, wave_grad_mean = debug_gradients(pinn_model)
            grad_ratio = speed_grad_mean / (wave_grad_mean + 1e-8)

            optimizer.step()

            # Logging
            history['loss_pde'].append(losses['loss_pde'].item())
            history['loss_data'].append(losses['loss_data'].item())
            history['loss_ic'].append(losses['loss_ic'].item())
            history['Rel_L2'].append(losses['Rel_L2'].item())

            writer.writerow([epoch, phase, loss_total.item(), losses['loss_pde'].item(), losses['loss_data'].item(), losses['loss_ic'].item(),
                             losses['Rel_L2'].item(), losses['debug_c_min'].item(), losses['debug_c_max'].item(), losses['debug_c_mean'].item(), 0.0,
                             grad_ratio, speed_grad_mean, wave_grad_mean])

            if epoch % 50 == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(f"Ep {epoch:04d} [{phase}] | Tot Loss: {loss_total.item():.2e} | Rel_L2: {losses['Rel_L2'].item():.2e} | C_mean: {losses['debug_c_mean'].item():.0f} | Elapsed: {elapsed:.2f}s")
                start_time = time.time()

            if epoch % 100 == 0:
                save_diagnostic_plot(pinn_model, epoch, run_dir, CONFIG, physics_engine, data_dict_multi, history)

            if loss_total.item() < best_loss and losses['loss_pde'].item() < 1e-2 and not speednet_frozen:
                best_loss = loss_total.item()
                torch.save(pinn_model.state_dict(), os.path.join(run_dir, "checkpoints", "best_model.pth"))

            if epoch % 500 == 0:
                torch.save(pinn_model.state_dict(), os.path.join(run_dir, "checkpoints", f"ckpt_ep{epoch:04d}.pth"))

    print(f"=== TRAINING COMPLETE. Results saved in {run_dir} ===")

if __name__ == "__main__":
    train()

import os
import h5py
import numpy as np
import torch
from src.config import GlobalConfig

class UltrasoundDataLoader:
    def __init__(self, config: GlobalConfig):
        self.cfg = config
        self.device = config.device

    def _read_mat_v73(self, filepath):
        """Helper robusto per leggere file Matlab v7.3 HDF5."""
        data = {}
        try:
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    item = f[key]
                    if isinstance(item, h5py.Dataset):
                        val = np.array(item)
                        val = np.squeeze(val)
                        if val.ndim == 2: val = val.T
                        data[key] = val
                    elif isinstance(item, h5py.Group):
                        sub_data = {}
                        for sub_key in item.keys():
                            val = np.array(item[sub_key])
                            val = np.squeeze(val)
                            if val.ndim == 2: val = val.T
                            sub_data[sub_key] = val
                        data[key] = sub_data
            return data
        except Exception as e:
            print(f"[ERROR] Impossibile leggere {filepath}: {e}")
            return None

    def load_dataset(self):
        """
        Carica i dati COMPLETI e crea le MASCHERE temporali basate sugli angoli.
        """
        case_dir = self.cfg.dataset_path
        temp_data = {}

        for angle_name in self.cfg.target_angles:
            angle_dir = os.path.join(case_dir, angle_name)
            rf_path = os.path.join(angle_dir, "sensor_data_p.mat")

            data_dict = self._read_mat_v73(rf_path)
            if data_dict is None or 'sensor_data_p' not in data_dict:
                raise ValueError(f"Dati mancanti in {angle_name}. Assicurati che il file esista in {rf_path}")

            rf_signal = data_dict['sensor_data_p'] 
            
            # Masking Strategy
            mask_idx = 18 if "angle_+00" in angle_name else 218
            
            Nt = rf_signal.shape[1]
            mask_array = np.ones((1, Nt), dtype=np.float32)
            mask_array[:, :mask_idx] = 0.0
            mask_full = np.tile(mask_array, (rf_signal.shape[0], 1))

            temp_data[angle_name] = {
                'signal': rf_signal,
                'mask': mask_full,
                'theta_deg': float(angle_name.replace('angle_', '').replace('+', ''))
            }

        # Calcolo U_ref sui dati mascherati
        global_max_val = 0.0
        for angle in temp_data:
            valid_signal = temp_data[angle]['signal'] * temp_data[angle]['mask']
            curr_max = np.max(np.abs(valid_signal))
            if curr_max > global_max_val:
                global_max_val = curr_max

        if self.cfg.norm.U_ref == 1.0:
            self.cfg.norm.U_ref = float(global_max_val)

        # Caricamento Ground Truth e parametri finali
        workspace_path = os.path.join(case_dir, "medium_and_sim_params.mat")
        global_params = self._read_mat_v73(workspace_path)
        full_speed_map = global_params['medium']['sound_speed']

        Ny_grid = full_speed_map.shape[1]
        n_sensors = self.cfg.physics.num_elements
        start_col = (Ny_grid - n_sensors) // 2
        
        speed_map_cropped = full_speed_map[:, start_col:start_col + n_sensors]
        c_true_tensor = torch.tensor(speed_map_cropped, dtype=torch.float32, device=self.device)

        x_axis_phys = np.linspace(self.cfg.physics.x_min, self.cfg.physics.x_max, 
                                  self.cfg.physics.num_elements, dtype=np.float32)
        
        Nt_common = temp_data['angle_+00']['signal'].shape[1]
        t_axis_phys = np.arange(Nt_common, dtype=np.float32) * self.cfg.physics.dt_grid

        multi_angle_data = {}
        for angle in self.cfg.target_angles:
            dat = temp_data[angle]
            u_norm = dat['signal'] / self.cfg.norm.U_ref

            multi_angle_data[angle] = {
                'u_sct': torch.tensor(u_norm, dtype=torch.float32, device=self.device),
                'mask': torch.tensor(dat['mask'], dtype=torch.float32, device=self.device),
                'incident_params': {'steering_angle': dat['theta_deg']},
                'c_true': c_true_tensor,
                'x_axis': x_axis_phys,
                't_axis': t_axis_phys
            }

        # Nessun shift temporale applicato ai dati
        t_delays_tensor = torch.zeros(len(self.cfg.target_angles), dtype=torch.float32, device=self.device)

        return multi_angle_data, c_true_tensor, t_delays_tensor

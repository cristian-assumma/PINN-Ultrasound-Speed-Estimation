import os
import torch
import random
import numpy as np
from dataclasses import dataclass

def set_seed(seed=42):
    """Garantisce riproducibilità assoluta."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class PhysicsConfig:
    c0: float = 1540.0
    rho0: float = 1000.0
    f_c: float = 7.5e6
    dx_grid: float = 4.928e-05
    dt_grid: float = 3.2e-08
    pitch: float = 3.0e-04
    num_elements: int = 128

    @property
    def z_max(self) -> float:
        return 1015 * self.dx_grid 

    @property
    def x_width(self) -> float:
        return (self.num_elements - 1) * self.pitch 

    @property
    def x_min(self) -> float:
        return -self.x_width / 2.0

    @property
    def x_max(self) -> float:
        return self.x_width / 2.0

@dataclass
class NormalizationConfig:
    L_ref: float = 0.05 
    U_ref: float = 1.0

    @property
    def T_ref(self) -> float:
        return self.L_ref / 1540.0

class GlobalConfig:
    def __init__(self, dataset_type="anechoic_cyst", case_name="case_0012"):
        self.seed = 1234
        self.device = get_device()
        
        # Percorsi relativi alla root del repository
        # Assumiamo che i dati siano in: pinn-ultrasound-speed-estimation/data/dataset/...
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_path = os.path.join(project_root, "data", "dataset", dataset_type, case_name)
        
        self.target_angles = ["angle_-15", "angle_+00", "angle_+15"]

        self.physics = PhysicsConfig()
        self.norm = NormalizationConfig(L_ref=self.physics.z_max)

        # Training Hyperparameters
        self.train_epochs = 2500
        self.learning_rate = 1e-4
        self.batch_size_collocation = 8000
        self.batch_size_data = 2000

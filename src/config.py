import os
import torch
import random
import numpy as np
from dataclasses import dataclass

def set_seed(seed=42):
    """Ensures absolute reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """Detects and returns the available hardware device (GPU or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class PhysicsConfig:
    """Immutable constants defining the physical problem and acquisition grid."""
    c0: float = 1540.0         # Reference sound speed (m/s)
    rho0: float = 1000.0       # Reference density (kg/m^3)
    f_c: float = 7.5e6         # Transducer central frequency (Hz)
    dx_grid: float = 4.928e-05 # k-Wave spatial step (m)
    dt_grid: float = 3.2e-08   # k-Wave time step (s)
    pitch: float = 3.0e-04     # Transducer element pitch (m)
    num_elements: int = 128    # Number of transducer elements

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
    """Constants for non-dimensionalizing the PDE to [-1, 1]."""
    L_ref: float = 0.05 
    U_ref: float = 1.0

    @property
    def T_ref(self) -> float:
        return self.L_ref / 1540.0

class GlobalConfig:
    """Master configuration container for hyperparameters and paths."""
    def __init__(self, dataset_type="anechoic_cyst", case_name="case_0012"):
        self.seed = 1234
        self.device = get_device()
        
        # Relative paths from the repository root
        # Assuming data is located in: pinn-ultrasound-speed-estimation/data/dataset/...
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

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseConfig:
    output_dir: str = "logs/"
    source_dir: Optional[str] = None


@dataclass
class QDConfig(BaseConfig):
    init_steps: int = 20
    total_steps: int = 1000
    history_length: int = 1
    save_history: bool = False
    save_snapshot_interval: int = 250
    log_snapshot_interval: int = 50
    log_snapshot_dir: str = ""
    seed: Optional[int] = 42
    save_np_rng_state: bool = False
    load_np_rng_state: bool = False
    crossover: bool = True
    crossover_parents: int = 2
    pool_size: int = 10 # for GA
    centroids_folder: str = "2d_centroids"  # 3d_centroids for 3d, 2d_centroids for 2d (or any custom folder containing centroids npy files)
    thres: float = 1.0
    transfer: bool = False
    transfer_model: Optional[str] = None



@dataclass
class MAPElitesConfig(QDConfig):
    qd_name: str = "mapelites"
    map_grid_size: tuple[int, ...] = field(default_factory=lambda: (14,))


@dataclass
class CVTMAPElitesConfig(QDConfig):
    qd_name: str = "cvtmapelites"
    n_niches: int = 196
    cvt_samples: int = 10000


@dataclass
class EnvConfig(BaseConfig):
    batch_size: int = 1  # Batch size of MAP-Elites
    env_name: str = MISSING
    debug: bool = False
    seed: Optional[int] = 42
    dataset_model: str = "entrep" # mscoco_b32, mscoco_l14, entrep, mimic, entrep_class, mimic_class
    num_queries: int = 100
    variation_mode: str = "full"    # "full", "char-only", "token-only"


@dataclass
class AttackConfig(EnvConfig):
    env_name: str = "attack"
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [0, 1],
            [0, 1],
        ]
    )
    dim: int = 2


defaults_elm = [
    {"qd": "cvtmapelites"},
    {"env": "attack"},
    "_self_",
]


@dataclass
class ELMConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "${run_name}" 
            },
        }
    )
    defaults: list[Any] = field(default_factory=lambda: defaults_elm)
    qd: Any = MISSING
    env: Any = MISSING
    run_name: Optional[str] = None
    source_dir: Optional[str] = None




def register_configstore() -> ConfigStore:
    """Register configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(group="env", name="attack", node=AttackConfig)
    cs.store(group="qd", name="mapelites", node=MAPElitesConfig)
    cs.store(group="qd", name="cvtmapelites", node=CVTMAPElitesConfig)
    cs.store(group="qd", name="ga", node=MAPElitesConfig)
    cs.store(group="qd", name="cvtga", node=MAPElitesConfig)
    cs.store(name="elm", node=ELMConfig)
    return cs


CONFIGSTORE = register_configstore()
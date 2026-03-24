import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from openelm.elm import ELM
from openelm.utils.utils import set_seed

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(
    config_name="elm",
    version_base="1.2",
)
def main(config):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("----------------------------------------")
    config = OmegaConf.to_object(config)

    set_seed(config.env.seed)

    if config.env.dataset_model == "entrep":
        from openelm.dataset_model.entrep import init_queries
    elif config.env.dataset_model == "entrep_class":
        from openelm.dataset_model.entrep_class import init_queries
    elif config.env.dataset_model == "mimic":
        from openelm.dataset_model.mimic import init_queries
    elif config.env.dataset_model == "mimic_class":
        from openelm.dataset_model.mimic_class import init_queries
    elif config.env.dataset_model == "mscoco_b32":
        from openelm.dataset_model.coco_b32 import init_queries
    elif config.env.dataset_model == "mscoco_l14":
        from openelm.dataset_model.coco_l14 import init_queries
    else:
        raise ValueError(f"No dataset named {config.env.dataset_model}")

    # Choose the same set of queries for each run for fair comparison
    rng = np.random.default_rng(42)
    data_index = rng.integers(low=0, high=len(init_queries) - 1, size=config.env.num_queries)
    print(f"Data indices: {data_index}")
    sampled_queries = init_queries[data_index]

    for idx, query in tqdm(enumerate(sampled_queries)):
        print(query)
        elm = ELM(config, query=query, idx=idx)
        best_indv = elm.run(init_steps=config.qd.init_steps, total_steps=config.qd.total_steps, idx=idx)

    print("Best Individual: ", best_indv)

if __name__ == "__main__":
    main()
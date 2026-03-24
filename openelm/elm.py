import numpy as np
from typing import Any, Optional

from openelm.configs import ELMConfig
from openelm.environments.vlmattack_retrieval_2d import AttackRetrievalEvolution


def load_algorithm(algorithm_name: str) -> Any:
    if algorithm_name == "mapelites":
        from openelm.algorithms.map_elites import MAPElites
        return MAPElites
    
    elif algorithm_name == "cvtmapelites":
        from openelm.algorithms.map_elites import CVTMAPElites
        return CVTMAPElites
    
    elif algorithm_name == "ga":
        from openelm.algorithms.genetic import GA
        return GA

    elif algorithm_name == "cvtga":
        from openelm.algorithms.genetic import CVTGA
        return CVTGA


class ELM:
    def __init__(self, config: ELMConfig, query, idx) -> None:
        """
        The main class of ELM.

        This class will load a diff model, an environment, and a QD algorithm
        from the passed config.

        Args:
            config: The config containing the diff model, environment, and QD algorithm.
            env (Optional): An optional environment to pass in. Defaults to None.
        """
        self.config: ELMConfig = config
        self.config.qd.output_dir = self.config.run_name
        self.config.env.output_dir = self.config.run_name
        self.env_name: str = self.config.env.env_name
        self.qd_name: str = self.config.qd.qd_name

        self.config.env.behavior_space = [[0, 1]] * self.config.env.dim

        if self.config.env.dim == 2:
            from openelm.environments.vlmattack_retrieval_2d import AttackRetrievalEvolution
            self.environment = AttackRetrievalEvolution(config=self.config.env, query=query, idx=idx)
        elif self.config.env.dim == 3:
            from openelm.environments.vlmattack_retrieval_3d import AttackRetrievalEvolution
            self.environment = AttackRetrievalEvolution(config=self.config.env, query=query, idx=idx)
        else:
            raise ValueError(f"No environment implemented for {self.config.env.dim} dimensions")

    def run(
        self, init_steps: Optional[int] = None, total_steps: Optional[int] = None, idx = -1
    ) -> str:
        """
        Run the ELM algorithm to evolve the population in the environment.

        Args:
            init_steps: The number of steps to run the initialisation phase.
            total_steps: The number of steps to run the QD algorithm in total,
            including init_steps.

        Returns:
            str: A string representing the maximum fitness genotype. The
            `qd_algorithm` class attribute will be updated.
        """
        
        self.config.qd.source_dir = self.config.source_dir
        self.qd_algorithm = load_algorithm(self.qd_name)(
            env=self.environment,
            config=self.config.qd,
            env_config=self.config.env,
        )
        if init_steps is None:
            init_steps = self.config.qd.init_steps
        if total_steps is None:
            total_steps = self.config.qd.total_steps

        if self.config.source_dir is not None:
            return self.qd_algorithm.transfer(total_steps=total_steps, idx=idx)
        else:
            return self.qd_algorithm.search(init_steps=init_steps, total_steps=total_steps, idx=idx)
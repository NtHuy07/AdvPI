import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from tqdm import trange
import pandas as pd

from openelm.configs import CVTMAPElitesConfig, MAPElitesConfig, QDConfig, EnvConfig
from openelm.environments import BaseEnvironment, Genotype
from openelm.algorithms.map_elites import Map


Phenotype = Optional[np.ndarray]
MapIndex = Optional[tuple]
Individual = Tuple[np.ndarray, float]


class Pool:
    """The pool stores a set of solutions or individuals."""

    def __init__(self, pool_size: int):
        """Initializes an empty pool.

        Args:
            pool_size (int): The number of solutions to store in the pool.
            history_length (int): The number of historical solutions
                to maintain in the pool.
        """
        self.pool_size = pool_size
        self.pool = []

    def add(self, solution, fitness):
        """Adds a solution to the pool.

        If the pool is full, the oldest solution is removed. The solution
        is also added to the history.

        Args:
            solution: The solution to add to the pool.
        """
        # if there are not any individual yet, add it to the pool
        if len(self.pool) < self.pool_size:
            self.pool.append((solution, fitness))
            self.pool.sort(key=lambda x: x[1], reverse=True)
            return

        # if new fitness is better than the worst, add it to the pool
        if fitness > self.pool[-1][1]:
            if len(self.pool) >= self.pool_size:
                self.pool.pop(len(self.pool) - 1)
            self.pool.append((solution, fitness))
            # sort the pool by fitness
            self.pool.sort(key=lambda x: x[1], reverse=True)



class GABase:
    """
    Base class for a genetic algorithm
    """

    def __init__(
        self,
        env,
        config: QDConfig,
        env_config,
        init_pool: Optional[Pool] = None,
    ):
        """
        The base class for a genetic algorithm, implementing common functions and search.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (QDConfig): The configuration for the algorithm.
            init_pool (Pool, optional): A pool to use for the algorithm. If not passed,
            a new pool will be created. Defaults to None.
        """
        self.env: BaseEnvironment = env
        self.config: QDConfig = config
        self.env_config = env_config
        self.save_history = self.config.save_history
        self.save_snapshot_interval = self.config.save_snapshot_interval
        self.log_snapshot_interval = self.config.log_snapshot_interval
        self.history_length = self.config.history_length
        self.start_step = 0
        self.save_np_rng_state = self.config.save_np_rng_state
        self.load_np_rng_state = self.config.load_np_rng_state
        self.log_df = pd.DataFrame(columns=["Step", "Max fitness", "Min fitness", "Mean fitness", "QD Score", "Coverage"])
        self.rng = np.random.default_rng(self.config.seed)
        self.rng_generators = None
        self.start_step = 0

        # self.history will be set/reset each time when calling `.search(...)`
        self.history: dict = defaultdict(list)
        self.fitness_history: dict = defaultdict(list)

        self._init_pool(init_pool, self.config.log_snapshot_dir)
        try:
            with open(self.config.centroids_folder + "/centroids.pkl", 'rb') as f:
                centroid = pickle.load(f)
            print("Load old centroids file")
        except FileNotFoundError:
            centroid = None

        self._init_discretization(centroid)
        self._init_maps(log_snapshot_dir=self.config.log_snapshot_dir)
        print(f"MAP of size: {self.fitnesses.dims} = {self.fitnesses.map_size}")

    def _init_discretization(self):
        """Initializes the discretization of the behavior space."""
        raise NotImplementedError

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        raise NotImplementedError

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        raise NotImplementedError


    def _init_maps(
        self, init_map: Optional[Map] = None, log_snapshot_dir: Optional[str] = None
    ):
        # perfomance of niches
        if init_map is None:
            self.map_dims = self._get_map_dimensions()
            self.fitnesses: Map = Map(
                dims=self.map_dims,
                fill_value=-np.inf,
                dtype=float,
                history_length=self.history_length,
            )
        else:
            self.map_dims = init_map.dims
            self.fitnesses = init_map

        # niches' sources
        self.genomes: Map = Map(
            dims=self.map_dims,
            fill_value=0.0,
            dtype=object,
            history_length=self.history_length,
        )
        
        # index over explored niches to select from
        self.nonzero: Map = Map(dims=self.map_dims, fill_value=False, dtype=bool)

        log_path = Path(log_snapshot_dir + f"overall")
        if log_snapshot_dir and os.path.isdir(log_path):
            stem_dir = log_path.stem

            assert (
                "step_" in stem_dir
            ), f"loading directory ({stem_dir}) doesn't contain 'step_' in name"
            self.start_step = (
                int(stem_dir.replace("step_", "")) + 1
            )  # add 1 to correct the iteration steps to run

            with open(log_path / "config.json") as f:
                old_config = json.load(f)

            snapshot_path = log_path / "maps.pkl"
            assert os.path.isfile(
                snapshot_path
            ), f'{log_path} does not contain map snapshot "maps.pkl"'
            # first, load arrays and set them in Maps
            # Load maps from pickle file
            with open(snapshot_path, "rb") as f:
                maps = pickle.load(f)
            assert (
                self.genomes.array.shape == maps["genomes"].shape
            ), f"expected shape of map doesn't match init config settings, got {self.genomes.array.shape} and {maps['genomes'].shape}"

            self.genomes.array = maps["genomes"]
            self.fitnesses.array = maps["fitnesses"]
            self.nonzero.array = maps["nonzero"]
            # check if one of the solutions in the snapshot contains the expected genotype type for the run
            assert not np.all(
                self.nonzero.array is False
            ), "snapshot to load contains empty map"

            assert (
                self.env.config.env_name == old_config["env_name"]
            ), f'unmatching environments, got {self.env.config.env_name} and {old_config["env_name"]}'

            # compute top indices
            if hasattr(self.fitnesses, "top"):
                top_array = np.array(self.fitnesses.top)
                for cell_idx in np.ndindex(
                    self.fitnesses.array.shape[1:]
                ):  # all indices of cells in map
                    nonzero = np.nonzero(
                        self.fitnesses.array[(slice(None),) + cell_idx] != -np.inf
                    )  # check full history depth at cell
                    if len(nonzero[0]) > 0:
                        top_array[cell_idx] = nonzero[0][-1]
                # correct stats
                self.genomes.top = top_array.copy()
                self.fitnesses.top = top_array.copy()
            self.genomes.empty = False
            self.fitnesses.empty = False

            history_path = log_path / "history.pkl"
            if self.save_history and os.path.isfile(history_path):
                with open(history_path, "rb") as f:
                    self.history = pickle.load(f)
            with open((log_path / "fitness_history.pkl"), "rb") as f:
                self.fitness_history = pickle.load(f)

            if self.load_np_rng_state:
                with open((log_path / "np_rng_state.pkl"), "rb") as f:
                    self.rng_generators = pickle.load(f)
                    self.rng = self.rng_generators["qd_rng"]
                    self.env.set_rng_state(self.rng_generators["env_rng"])

            print("Loading finished")

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        raise NotImplementedError

    def _init_pool(
        self, init_map: Optional[Pool] = None, log_snapshot_dir: Optional[str] = None
    ):
        if init_map is None and log_snapshot_dir is None:
            self.pool = Pool(self.config.pool_size)
        elif init_map is not None and log_snapshot_dir is None:
            self.pool = init_map
        elif init_map is None and log_snapshot_dir is not None:
            self.pool = Pool(self.config.pool_size)
            log_path = Path(log_snapshot_dir)
            if log_snapshot_dir and os.path.isdir(log_path):
                stem_dir = log_path.stem

                assert (
                    "step_" in stem_dir
                ), f"loading directory ({stem_dir}) doesn't contain 'step_' in name"
                self.start_step = (
                    int(stem_dir.replace("step_", "")) + 1
                )  # add 1 to correct the iteration steps to run

                snapshot_path = log_path / "pool.pkl"
                assert os.path.isfile(
                    snapshot_path
                ), f'{log_path} does not contain map snapshot "pool.pkl"'
                # first, load arrays and set them in Maps
                # Load maps from pickle file
                with open(snapshot_path, "rb") as f:
                    self.pool = pickle.load(f)

        print("Loading finished")

    def random_selection(self) -> MapIndex:
        """Randomly select a niche (cell) in the map that has been explored."""
        return random.choice(self.pool.pool)
    
    def tournament_selection(self, k=5):
        pop_with_fit = self.pool.pool

        # shuffle population
        random.shuffle(pop_with_fit)

        selected = []
        # partition into groups of size k
        for i in range(0, len(pop_with_fit), k):
            group = pop_with_fit[i:i+k]
            winner = max(group, key=lambda x: x[1])[0]
            selected.append(winner)

        return selected

    def search(self, init_steps: int, total_steps: int, idx: int) -> str:
        """
        Run the genetic algorithm.

        Args:
            initsteps (int): Number of initial random solutions to generate.
            totalsteps (int): Total number of steps to run the algorithm for,
                including initial steps.
            atol (float, optional): Tolerance for how close the best performing
                solution has to be to the maximum possible fitness before the
                search stops early. Defaults to 1.

        Returns:
            str: A string representation of the best perfoming solution. The
                best performing solution object can be accessed via the
                `current_max_genome` class attribute.
        """

        if self.niches_filled() == 0:
            max_fitness = -np.inf
            max_genome = None
        else:  # take max fitness in case of filled loaded snapshot
            max_fitness = self.max_fitness()
            max_index = np.where(self.fitnesses.latest == max_fitness)
            max_genome = self.genomes[max_index]
        if self.save_history:
            self.history = defaultdict(list)


        start_step = int(self.start_step)
        total_steps = int(total_steps)
        tbar = trange(start_step, total_steps, initial=start_step, total=total_steps)
        for n_steps in tbar:
            if n_steps < init_steps:
                # Initialise by generating initsteps random solutions
                new_individuals: list[Genotype] = self.env.random()
            else:
                new_individuals = []
                while len(new_individuals) == 0:
                    try:
                        batch: list[Genotype] = []
                        if self.config.crossover:
                            crossover_parents = []
                            while len(crossover_parents) < self.config.crossover_parents:
                                # item = self.random_selection()
                                item = self.tournament_selection()
                                crossover_parents.append(item[0])
                            batch.append(crossover_parents)
                        else:
                            for _ in range(self.env.batch_size):
                                # item = self.random_selection()
                                item = self.tournament_selection()
                                batch.append(item)
                            # Mutate
                            batch = [indv[0] for indv in batch]
                        # Mutate the elite.
                        new_individuals = self.env.mutate(batch)

                    except Exception as e:
                        print(f"Error: {e}. Retrying...")                


            max_genome, max_fitness = self.update_map(
                new_individuals, 
                max_genome, 
                max_fitness, 
            )

            niches_filled = self.niches_filled()
            max_fitness = self.max_fitness()
            qd_score = self.qd_score()
            coverage = niches_filled / self.fitnesses.map_size
            
            tbar.set_description(f"{max_fitness=:.4f}, {coverage=:.4f}, {qd_score=:.4f}")

            if (
                self.save_snapshot_interval is not None
                and n_steps != 0
                and (n_steps + 1) % self.save_snapshot_interval == 0
            ):
                self.save_results(step=n_steps, idx=idx)

            if (
                self.log_snapshot_interval is not None
                and n_steps != 0
                and (n_steps + 1) % self.log_snapshot_interval == 0
            ):
                self.log_results(step=n_steps, idx=idx)

        self.current_max_genome = max_genome
        self.save_results(step=n_steps, idx=idx)
        return str(max_genome)
    
    def update_map(self, new_individuals, max_genome, max_fitness):
        """
        Update the map if new individuals achieve better fitness scores.

        Args:
            new_individuals (list[Genotype]) : List of new solutions
            max_fitness : current maximum fitness

        Returns:
            max_genome : updated maximum genome
            max_fitness : updated maximum fitness

        """
        # `new_individuals` is a list of generation/mutation. We put them
        # into the behavior space one-by-one.
        for individual in new_individuals:
            fitness = self.env.fitness(individual)
            max_genome, max_fitness = "", 0
            if np.isinf(fitness):
                continue

            phenotype = individual.to_phenotype()
            map_ix = self.to_mapindex(phenotype)

            # if np.any(phenotype > 1) or np.any(phenotype < 0):
            #     continue
                
            if np.all(phenotype <= self.config.thres):
                self.pool.add(individual, fitness)

            # if the return is None, the individual is invalid and is thrown
            # into the recycle bin.
            if map_ix is None:
                self.recycled[self.recycled_count % len(self.recycled)] = individual
                self.recycled_count += 1
                continue

            if self.save_history:
                self.history[map_ix].append(individual)

            self.nonzero[map_ix] = True

            # If new fitness greater than old fitness in niche, replace.
            if fitness > self.fitnesses[map_ix]:
                self.fitnesses[map_ix] = fitness
                self.genomes[map_ix] = individual

            # update if new fitness is the highest so far.
            if fitness > max_fitness:
                max_fitness = fitness
                max_genome = individual

        return max_genome, max_fitness

    def niches_filled(self):
        """Get the number of niches that have been explored in the map."""
        return self.fitnesses.niches_filled

    def max_fitness(self):
        """Get the maximum fitness value in the map."""
        return self.fitnesses.max_finite

    def mean_fitness(self):
        """Get the mean fitness value in the map."""
        return self.fitnesses.mean

    def min_fitness(self):
        """Get the minimum fitness value in the map."""
        return self.fitnesses.min_finite

    def qd_score(self):
        """
        Get the quality-diversity score of the map.

        The quality-diversity score is the sum of the performance of all solutions
        in the map.
        """
        return self.fitnesses.qd_score


    def save_results(self, step: int, idx):
        step = step + 1
        # create folder for dumping results and metadata
        output_folder = Path(self.config.output_dir + "/" + str(idx) + "/overall") / f"step_{step}"
        os.makedirs(output_folder, exist_ok=True)
        maps = {
            "fitnesses": self.fitnesses.array,
            "genomes": self.genomes.array,
            "nonzero": self.nonzero.array,
        }

        # Save maps as pickle file
        try:
            with open((output_folder / "maps.pkl"), "wb") as f:
                pickle.dump(maps, f)
        except Exception:
            pass
        if self.save_history:
            with open((output_folder / "history.pkl"), "wb") as f:
                pickle.dump(self.history, f)

        with open((output_folder / "fitness_history.pkl"), "wb") as f:
            pickle.dump(self.fitness_history, f)

        # save numpy rng state to load if resuming from deterministic snapshot
        if self.save_np_rng_state:
            rng_generators = {
                "env_rng": self.env.get_rng_state(),
                "qd_rng": self.rng,
            }
            with open((output_folder / "np_rng_state.pkl"), "wb") as f:
                pickle.dump(rng_generators, f)

        # save env_name to check later, for verifying correctness of environment to run with snapshot load
        tmp_config = dict()
        tmp_config["env_name"] = self.env.config.env_name

        with open((output_folder / "config.json"), "w") as f:
            json.dump(tmp_config, f)
        f.close()

    def log_results(self, step: int, idx):
        step = step + 1
        # create folder for dumping results and metadata
        global_output_folder = Path(self.config.output_dir + str(idx))
        os.makedirs(global_output_folder, exist_ok=True)

        total_info = {
            "Step": step,
            "Max fitness": self.max_fitness(), 
            "Min fitness": self.min_fitness(),
            "Mean fitness": self.mean_fitness(),
            "QD Score": self.qd_score(),
            "Coverage": self.niches_filled()/self.fitnesses.map_size,
        }

        if "class" in self.env_config.dataset_model:
            for name in ["Precision@1", "Precision@5", "Precision@10", "Precision@50", "Precision@100", "Precision@200"]:
                metric_arr = np.array([1 - indiv.metrics[name] if indiv != 0.0 else 0 for indiv in self.genomes.array.flatten()])   # Inverse Quality Metric
                qd_metric = metric_arr.sum()
                max_metric = metric_arr.max()
                total_info["QD I" + name] = qd_metric
                total_info["Max I" + name] = max_metric
        
        else:
            for name in ["Hit@1", "Hit@5", "Hit@10", "Hit@50", "Hit@100", "Hit@200"]:
                metric_arr = np.array([1 - indiv.metrics[name] if indiv != 0.0 else 0 for indiv in self.genomes.array.flatten()])   # Inverse Quality Metric
                qd_metric = metric_arr.sum()
                max_metric = metric_arr.max()
                total_info["QD I" + name] = qd_metric
                total_info["Max I" + name] = max_metric

        self.log_df = pd.concat([self.log_df, pd.DataFrame(total_info, index=[0])], axis=0, ignore_index=True)
        self.log_df.to_csv((global_output_folder / "log.csv"))


class GA(GABase):
    """
    Class implementing MAP-Elites, a quality-diversity algorithm.

    MAP-Elites creates a map of high perfoming solutions at each point in a
    discretized behavior space. First, the algorithm generates some initial random
    solutions, and evaluates them in the environment. Then, it  repeatedly mutates
    the solutions in the map, and places the mutated solutions in the map if they
    outperform the solutions already in their niche.
    """

    def __init__(
        self,
        env,
        config: MAPElitesConfig,
        *args,
        **kwargs,
    ):
        """
        Class implementing MAP-Elites, a quality-diversity algorithm.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (MAPElitesConfig): The configuration for the algorithm.
        """
        self.map_grid_size = config.map_grid_size
        super().__init__(env=env, config=config, *args, **kwargs)

    def _init_discretization(self):
        """Set up the discrete behaviour space for the algorithm."""
        # TODO: make this work for any number of dimensions
        self.bins = np.linspace(*self.env.behavior_space, self.map_grid_size[0] + 1)[1:-1].T  # type: ignore

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        return self.map_grid_size * self.env.behavior_ndim

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, self.bins))
        )


class CVTGA(GABase):
    """
    Class implementing CVT-MAP-Elites, a variant of MAP-Elites.

    This replaces the grid of niches in MAP-Elites with niches generated using a
    Centroidal Voronoi Tessellation. Unlike in MAP-Elites, we have a fixed number
    of total niches rather than a fixed number of subdivisions per dimension.
    """

    def __init__(
        self,
        env,
        config: CVTMAPElitesConfig,
        *args,
        **kwargs,
    ):
        """
        Class implementing CVT-MAP-Elites, a variant of MAP-Elites.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (CVTMAPElitesConfig): The configuration for the algorithm.
        """
        self.cvt_samples: int = config.cvt_samples
        self.n_niches: int = config.n_niches
        super().__init__(env=env, config=config, *args, **kwargs)

    def _init_discretization(self, centroid):
        """Discretize behaviour space using CVT."""
        # lower and upper bounds for each dimension
        low = self.env.behavior_space[0]
        high = self.env.behavior_space[1]

        if centroid is not None:
            self.centroids = centroid
        else:
            print("Create new centroid")

            points = np.zeros((self.cvt_samples, self.env.behavior_ndim))
            for i in range(self.env.behavior_ndim):
                points[:, i] = self.rng.uniform(low[i], high[i], size=self.cvt_samples)

            k_means = KMeans(init="k-means++", n_init="auto", n_clusters=self.n_niches)
            k_means.fit(points)
            self.centroids = k_means.cluster_centers_

            os.makedirs(self.config.centroids_folder, exist_ok=True)
            with open(self.config.centroids_folder + "/centroids.pkl", "wb") as f:
                pickle.dump(self.centroids, f)

        # self.plot_centroids(points, k_means)

    def _get_map_dimensions(self):
        """Returns the dimensions of the map."""
        return (self.n_niches,)

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Maps a phenotype (position in behaviour space) to the index of the closest centroid."""
        return (
            None
            if b is None
            else (np.argmin(np.linalg.norm(b - self.centroids, axis=1)),)
        )

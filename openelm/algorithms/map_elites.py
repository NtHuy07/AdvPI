import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm, trange

from openelm.configs import CVTMAPElitesConfig, MAPElitesConfig, QDConfig, EnvConfig
from openelm.environments import BaseEnvironment, Genotype


Phenotype = Optional[np.ndarray]
MapIndex = Optional[tuple]


class Map:
    """
    Class to represent a map of any dimensionality, backed by a numpy array.

    This class is necessary to handle the circular buffer for the history dimension.
    """

    def __init__(
        self,
        dims: tuple,
        fill_value: float,
        dtype: type = np.float32,
        history_length: int = 1,
    ):
        """
        Class to represent a map of any dimensionality, backed by a numpy array.

        This class is a wrapper around a numpy array that handles the circular
        buffer for the history dimension. We use it for the array of fitnesses,
        genomes, and to track whether a niche in the map has been explored or not.

        Args:
            dims (tuple): Tuple of ints representing the dimensions of the map.
            fill_value (float): Fill value to initialize the array.
            dtype (type, optional): Type to pass to the numpy array to initialize
                it. For example, we initialize the map of genomes with type `object`.
                Defaults to np.float32.
            history_length (int, optional): Length of history to store for each
                niche (cell) in the map. This acts as a circular buffer, so after
                storing `history_length` items, the buffer starts overwriting the
                oldest items. Defaults to 1.
        """
        self.history_length: int = history_length
        self.dims: tuple = dims
        if self.history_length == 1:
            self.array: np.ndarray = np.full(dims, fill_value, dtype=dtype)
        else:
            # Set starting top of buffer to 0 (% operator)
            self.top = np.full(dims, self.history_length - 1, dtype=int)
            self.array = np.full(
                (history_length,) + tuple(dims), fill_value, dtype=dtype
            )
        self.empty = True

    def __getitem__(self, map_ix):
        """If history length > 1, the history dim is an n-dim circular buffer."""
        if self.history_length == 1:
            return self.array[map_ix]
        else:
            return self.array[(self.top[map_ix], *map_ix)]

    def __setitem__(self, map_ix, value):
        self.empty = False
        if self.history_length == 1:
            self.array[map_ix] = value
        else:
            top_val = self.top[map_ix]
            top_val = (top_val + 1) % self.history_length
            self.top[map_ix] = top_val
            self.array[(self.top[map_ix], *map_ix)] = value

    def assign_fitness_in_depth(self, map_ix, value: float) -> int:
        indices_at_bin = (slice(None),) + map_ix
        # expecting a non-empty index, only calling this method when we know
        # current fitness can be placed somewhere
        insert_idx = np.where(self.array[indices_at_bin] < value)[0][-1]
        new_bin_fitnesses = np.concatenate(
            (
                self.array[indices_at_bin][1 : insert_idx + 1],
                np.array([value]),
                self.array[indices_at_bin][insert_idx + 1 :],
            )
        )
        self.array[indices_at_bin] = new_bin_fitnesses
        return insert_idx

    def insert_individual_at_depth(self, map_ix, depth, individual):
        indices_at_bin = (slice(None),) + map_ix
        new_bin_individuals = np.concatenate(
            (
                self.array[indices_at_bin][1 : depth + 1],
                np.array([individual]),
                self.array[indices_at_bin][depth + 1 :],
            )
        )
        self.array[indices_at_bin] = new_bin_individuals

    @property
    def latest(self) -> np.ndarray:
        """Returns the latest values in the history buffer."""
        if self.history_length == 1:
            return self.array
        else:
            # should be equivalent to np.choose(self.top, self.array), but without limit of 32 choices
            return np.take_along_axis(
                arr=self.array, indices=self.top[np.newaxis, ...], axis=0
            ).squeeze(axis=0)

    @property
    def shape(self) -> tuple:
        """Wrapper around the shape of the numpy array."""
        return self.array.shape

    @property
    def map_size(self) -> int:
        """Returns the product of the dimension sizes, not including history."""
        if self.history_length == 1:
            return self.array.size
        else:
            return self.array[0].size

    @property
    def qd_score(self) -> float:
        """Returns the quality-diversity score of the map."""
        return self.latest[np.isfinite(self.latest)].sum()

    @property
    def max(self) -> float:
        """Returns the maximum value in the map, not including history."""
        return self.latest.max()

    @property
    def min(self) -> float:
        """Returns the minimum value in the map, not including history."""
        return self.latest.min()

    @property
    def max_finite(self) -> float:
        """Returns the maximum finite value in the map, not including history."""
        if not np.isfinite(self.latest).any():
            return np.nan
        else:
            return self.latest[np.isfinite(self.latest)].max()

    @property
    def min_finite(self) -> float:
        """Returns the minimum finite value in the map, not including history."""
        if not np.isfinite(self.latest).any():
            return np.nan
        else:
            return self.latest[np.isfinite(self.latest)].min()

    @property
    def mean(self) -> float:
        """Returns the mean finite value in the map."""
        return self.latest[np.isfinite(self.latest)].mean()

    @property
    def niches_filled(self) -> int:
        """Returns the number of niches in the map that have been explored."""
        return np.count_nonzero(np.isfinite(self.array))


class MAPElitesBase:
    """
    Base class for MAP-Elites, a quality-diversity algorithm.

    MAP-Elites creates a map of high perfoming solutions at each point in a
    discretized behavior space. First, the algorithm generates some initial random
    solutions, and evaluates them in the environment. Then, it  repeatedly mutates
    the solutions in the map, and places the mutated solutions in the map if they
    outperform the solutions already in their niche.
    """

    def __init__(
        self,
        env,
        config: QDConfig,
        env_config: EnvConfig,
        init_map: Optional[Map] = None,
    ):
        """
        The base class for MAP-Elites and variants, implementing common functions and search.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (QDConfig): The configuration for the algorithm.
            init_map (Map, optional): A map to use for the algorithm. If not passed,
            a new map will be created. Defaults to None.
        """
        self.env: BaseEnvironment = env
        self.config: QDConfig = config
        self.env_config: EnvConfig = env_config
        self.log_df = pd.DataFrame(columns=["Step", "Max fitness", "Min fitness", "Mean fitness", "QD Score", "Coverage"])
        self.history_length = self.config.history_length
        self.save_history = self.config.save_history
        self.save_snapshot_interval = self.config.save_snapshot_interval
        self.log_snapshot_interval = self.config.log_snapshot_interval
        self.start_step = 0
        self.save_np_rng_state = self.config.save_np_rng_state
        self.load_np_rng_state = self.config.load_np_rng_state
        self.rng = np.random.default_rng(self.config.seed)
        self.rng_generators = None

        # self.history will be set/reset each time when calling `.search(...)`
        self.history: dict = defaultdict(list)
        self.fitness_history: dict = defaultdict(list)

        # bad mutations that ended up with invalid output.
        self.recycled = [None] * 1000
        self.recycled_count = 0

        try:
            with open(self.config.centroids_folder + "/centroids.pkl", 'rb') as f:
                centroid = pickle.load(f)
            print("Load old centroid file")
        except FileNotFoundError:
            centroid = None

        self._init_discretization(centroid)
        self._init_maps(init_map, self.config.log_snapshot_dir)
        print(f"MAP of size: {self.fitnesses.dims} = {self.fitnesses.map_size}")

    def _init_discretization(self, centroid):
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

        log_path = Path(log_snapshot_dir + "overall")
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

    def random_selection(self) -> MapIndex:
        """Randomly select a niche (cell) in the map that has been explored."""
        ix = self.rng.choice(np.flatnonzero(self.nonzero.array))
        return np.unravel_index(ix, self.nonzero.dims)

    def search(self, init_steps: int, total_steps: int, idx: int):
        """
        Run the MAP-Elites search algorithm.

        Args:
            initsteps (int): Number of initial random solutions to generate.
            totalsteps (int): Total number of steps to run the algorithm for,
                including initial steps.

        Returns:
            str: A string representation of the best perfoming solution. The
                best performing solution object can be accessed via the
                `current_max_genome` class attribute.
        """
        start_step = int(self.start_step)
        total_steps = int(total_steps)
        if self.niches_filled() == 0:
            max_fitness = -np.inf
            max_genome = None
        else:  # take max fitness in case of filled loaded snapshot
            max_fitness = self.max_fitness()
            max_index = np.where(self.fitnesses.latest == max_fitness)
            max_genome = self.genomes[max_index]
        if self.save_history:
            self.history = defaultdict(list)

        tbar = trange(start_step, total_steps, initial=start_step, total=total_steps)

        for n_steps in tbar:

            if n_steps < init_steps:
                new_individuals = self.env.random()

            else:
                try:
                    batch: list[Genotype] = []
                    if self.config.crossover:
                        crossover_parents = []
                        previous_ix = None
                        retry = 0
                        while len(crossover_parents) < self.config.crossover_parents:
                            map_ix = self.random_selection()
                            if map_ix != previous_ix:
                                crossover_parents.append(self.genomes[map_ix])
                                previous_ix = map_ix
                            retry += 1
                            if retry > 10:  # If cannot select non-duplicated individuals
                                for _ in range(self.config.crossover_parents):
                                    map_ix = self.random_selection()
                                    crossover_parents.append(self.genomes[map_ix])
                                break
                        batch.append(crossover_parents)
                    else:
                        for _ in range(self.env.batch_size):
                            map_ix = self.random_selection()
                            batch.append([self.genomes[map_ix], self.genomes[map_ix]])  # No crossover -> duplicate individuals
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

    def transfer(self, total_steps: int, idx: int):
        """
        Run the MAP-Elites search algorithm.

        Args:
            initsteps (int): Number of initial random solutions to generate.
            totalsteps (int): Total number of steps to run the algorithm for,
                including initial steps.

        Returns:
            str: A string representation of the best perfoming solution. The
                best performing solution object can be accessed via the
                `current_max_genome` class attribute.
        """
        total_steps = int(total_steps)

        if self.niches_filled() == 0:
            max_fitness = -np.inf
            max_genome = None
        else:  # take max fitness in case of filled loaded snapshot
            max_fitness = self.max_fitness()
            max_index = np.where(self.fitnesses.latest == max_fitness)
            max_genome = self.genomes[max_index]
        if self.save_history:
            self.history = defaultdict(list)

        with open(f"{self.config.source_dir}/{idx}/overall/step_{str(total_steps)}/maps.pkl", "rb") as f:
            prev = pickle.load(f)
            arch = prev["genomes"]
            nonzero = prev["nonzero"]

        tbar = tqdm(zip(arch, nonzero))

        for new_ind, new_nonzero in tbar:
            if not new_nonzero:
                continue

            new_individuals = [new_ind]


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

        self.save_results(step=total_steps, idx=idx)
        self.log_results(step=total_steps, idx=idx)

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


class MAPElites(MAPElitesBase):
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

    def _init_discretization(self, centroid):
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

class CVTMAPElites(MAPElitesBase):
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

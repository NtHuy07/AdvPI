import os
import math
import numpy as np
from copy import deepcopy

from typing import Optional, List

from nltk.tokenize import RegexpTokenizer

from openelm.configs import AttackConfig
from openelm.environments.base import BaseEnvironment
from openelm.environments.descriptors import FastPerplexity, TokenSim, lpips_distance
from openelm.environments.attack_utils import AttackUtils

ppl_calculator = FastPerplexity()

class AttackRetrievalGenotype():

    def __init__(
        self, 
        idx: int, 
        attack_prompt: str, 
        parent_idx: List[str] = -1, 
        fitness: float = -1.0
    ):
        self.parent_idx = parent_idx
        self.idx = idx
        self.attack_prompt = attack_prompt
        self.fitness = fitness
        self.rank = fitness
        self.descs = []
        self.metrics = {}

    def __str__(self):
        return str(self.attack_prompt)

    def evaluate(self, orig_tokens, class_id, get_rank_fn, get_toksim_fn):

        real_atk_prompt = " ".join(self.attack_prompt).replace("_", " ")
        self.rank, self.metrics = get_rank_fn([(real_atk_prompt, class_id)])
        self.fitness = math.log(self.rank) # log for less agressive fitness difference

        vis_sim = lpips_distance(orig_tokens, self.attack_prompt)
        sem_shift, _ = get_toksim_fn.avg_token_embedding_shift(self.attack_prompt)
        ppl = ppl_calculator(real_atk_prompt)

        vis_sim_norm = np.clip((vis_sim - 0.05) / (0.25 - 0.05), 0, 1)
        sem_shift_norm = np.clip((sem_shift - 0.05) / (0.25 - 0.05), 0, 1)
        ppl_norm = np.clip((ppl - 100) / (1500 - 100), 0, 1)

        self.descs = [vis_sim_norm, sem_shift_norm, ppl_norm]

        return self.fitness


    def to_phenotype(self) -> Optional[np.ndarray]:
        return np.array(self.descs)
    

class AttackRetrievalEvolution(BaseEnvironment[AttackRetrievalGenotype]):

    def __init__(
        self,
        config: AttackConfig,
        query,
        idx,
    ):
        self.config: AttackConfig = config
        self.batch_size = self.config.batch_size
        self.genotype_space = np.array(self.config.behavior_space).T
        self.genotype_ndim = self.genotype_space.shape[1]

        
        self.orig_prompt = query[0]

        if config.dataset_model == "entrep":
            from openelm.dataset_model.entrep import get_rank
        elif config.dataset_model == "entrep_class":
            from openelm.dataset_model.entrep_class import get_rank
        elif config.dataset_model == "mimic":
            from openelm.dataset_model.mimic import get_rank
        elif config.dataset_model == "mimic_class":
            from openelm.dataset_model.mimic_class import get_rank
        elif config.dataset_model == "mscoco_b32":
            from openelm.dataset_model.coco_b32 import get_rank
        elif config.dataset_model == "mscoco_l14":
            from openelm.dataset_model.coco_l14 import get_rank
        else:
            raise ValueError(f"No dataset named {config.dataset_model}")
        
        self.class_id = query[1]
        self.get_rank_fn = get_rank

        self.attack_lib = AttackUtils(orig_prompt=self.orig_prompt)

        tokenizer = RegexpTokenizer(r'\w+')
        self.orig_tokens = tokenizer.tokenize(self.orig_prompt.lower())

        self.get_toksim_fn = TokenSim(orig_tokens=self.orig_tokens)

        self.rng = np.random.default_rng(self.config.seed)

        self.idx = idx
        self.count = 0
        
    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        self.rng = rng_state
        
    def construct_prompt(self, prompt_genomes: Optional[AttackRetrievalGenotype] = None):

        if prompt_genomes is None:
            parent_idx = -1
            cand1 = self.orig_tokens
            cand2 = self.orig_tokens
        else:
            parent_idx = [prompt_genomes[0].idx, prompt_genomes[1].idx]
            cand1 = deepcopy(prompt_genomes[0].attack_prompt)    
            cand2 = deepcopy(prompt_genomes[1].attack_prompt)    

        #exchange two parts randomly
        mask = np.random.rand(len(cand1)) < 0.5
        tokens = np.where(mask, cand1, cand2).tolist()

        legal_token = []
        for k, tok in enumerate(tokens):
            legal_token.append(k)
        
        word_idx = legal_token[np.random.choice(len(legal_token), size=1)[0]]
        word = tokens[word_idx]

        final_bug = self.attack_lib.selectBug(word, word_idx, self.config.variation_mode)

        new_tokens = self.attack_lib.replaceWithBug(tokens, word_idx, final_bug)


        return {
            "prompt": new_tokens,
            "parent_id": parent_idx,
        }

    def random(self, batch_size=1) -> list[AttackRetrievalGenotype]:
        # Mutate seed, and pick random target genre and poem.
        results = [self.construct_prompt() for _ in range(batch_size)]
        self.count += 1
        return [AttackRetrievalGenotype(idx=self.count, attack_prompt=c['prompt']) for c in results]

    def mutate(self, genomes: list[AttackRetrievalGenotype], ga=False) -> list[AttackRetrievalGenotype]:
        results = [self.construct_prompt(genome) for genome in genomes]
        self.count += 1
        return [AttackRetrievalGenotype(idx=self.count, attack_prompt=c['prompt'], parent_idx=c["parent_id"]) for c in results]

    def fitness(self, x: AttackRetrievalGenotype) -> float:
        
        fitness = x.evaluate(self.orig_tokens, self.class_id, self.get_rank_fn, self.get_toksim_fn)

        gene_info_str = f"""
[[ID: {x.idx}]]

-- Gene --
{x.attack_prompt} 

-- Fitness: {fitness} --
-- Behavior: {x.to_phenotype()} --
-- Parent ID: {x.parent_idx} --
-- Metrics: {x.metrics} --

        """
        if self.config.debug:
            print(gene_info_str)
        os.makedirs(self.config.output_dir + f"/{self.idx}", exist_ok=True)
        with open(self.config.output_dir + f"/{self.idx}/all_genomes_log.txt", "+a") as log_file:
            log_file.write(gene_info_str)
                
        return fitness
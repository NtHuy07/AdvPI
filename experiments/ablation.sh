#!/bin/bash
for i in {1..10}; do
  for ablation in char-only token-only; do
    uv run main.py run_name=logs/ablation/$ablation/entrep_class/run_$i/ dataset_model=entrep_class env.seed=$i env.variation_mode=$ablation
  done
  
  uv run main.py run_name=logs/ablation/no-crossover/entrep_class/run_$i/ dataset_model=entrep_class env.seed=$i qd.crossover=False
  
done
#!/bin/bash
for i in {1..10}; do
  uv run main.py run_name=logs/qd/entrep/run_$i/ dataset_model=entrep env.seed=$i
  uv run main.py run_name=logs/qd/entrep_class/run_$i/ dataset_model=entrep_class env.seed=$i
  uv run main.py run_name=logs/qd/mimic/run_$i/ dataset_model=mimic env.seed=$i
  uv run main.py run_name=logs/qd/mimic_class/run_$i/ dataset_model=mimic_class env.seed=$i
  uv run main.py run_name=logs/qd/mscoco_b32/run_$i/ dataset_model=mscoco_b32 env.seed=$i
  uv run main.py run_name=logs/qd/mscoco_l14/run_$i/ dataset_model=mscoco_l14 env.seed=$i
done
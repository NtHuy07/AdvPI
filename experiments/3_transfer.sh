#!/bin/bash
for i in {1..10}; do
  uv run main.py run_name=logs/transfer/mscoco_b32/run_$i/ source_dir=logs/qd/mscoco_l14/run_$i/ dataset_model=mscoco_b32 env.seed=$i
  uv run main.py run_name=logs/transfer/mscoco_l14/run_$i/ source_dir=logs/qd/mscoco_b32/run_$i/ dataset_model=mscoco_l14 env.seed=$i
done
#!/bin/bash
for i in {1..10}; do
  uv run main.py run_name=logs/qd/entrep_class/run_$i/ dataset_model=entrep_class env.seed=$i
  for thres in 1.0 0.75 0.5 0.25; do
    uv run main.py run_name=logs/ga-$thres/entrep_class/run_$i/ qd.qd_name=cvtga dataset_model=entrep_class env.seed=$i qd.thres=$thres
  done
done
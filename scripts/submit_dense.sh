#!/bin/bash

density=1.0

for size in "20m"; do
for epoch in 1; do
for steps in 10000; do
for lr in 1.95e-3; do
for bs in 128; do

sbatch ./scripts/train_llm_dense.sh $epoch $steps $size $lr $bs

done
done
done
done
done

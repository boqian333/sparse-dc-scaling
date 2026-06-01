#!/bin/bash


for size in "20m"; do
for epoch in 1; do
for steps in 10000; do
for lr in 3.12e-2; do
for density in 0.0625; do
for bs in 128; do

sbatch ./scripts/train_llm_dst.sh $epoch $steps $size $lr $bs $density

done
done
done
done
done
done

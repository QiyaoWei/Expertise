#!/bin/bash
StringVal="prog "
for i in $StringVal; do
    for j in {1,}; do
        for k in {0.01,}; do
            for m in {0.01,}; do
                for n in {4,0.1,0.5,1,1.5,2,2.5,3,3.5}; do
                    python run_experiments.py --experiment_name=propensity_sensitivity --dataset_list=tcga_100 --propensity_type $i --seed_list 2 \
                    --prop_scale $n --scale_factor $j --loss_mult $k --penalty_disc $m
                done
            done
        done
    done
done

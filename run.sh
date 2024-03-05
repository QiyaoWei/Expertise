#!/bin/bash
<<<<<<< HEAD
prop="pred prog"
=======
prop="prog pred"
>>>>>>> 72bf8dc (First commit)
sim="linear nonlinear"
data="tcga_100 news_100"
for i in $prop; do
    for j in $sim; do
        for k in $data; do
            for b in {0.1,0.5,1,1.5,2,2.5,3,3.5,4}; do
                for d in {0,1,2,3,4,5,6,7,8,9,10}; do
                    python run_experiments.py --seeds 11 --propensity_type $i --synthetic_simulator_type $j --dataset=$k --prop_scale $b --shift $d
                done
            done
        done
    done
done
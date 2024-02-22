import argparse
import sys
from typing import Any

from prog_experiments import PrognosticPropensitySensitivity
from pred_experiments import PredictivePropensitySensitivity


def init_arg() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="propensity_sensitivity", type=str)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--seeds", default=11, type=int)
    
    parser.add_argument("--propensity_type", default="pred", type=str)
    parser.add_argument("--synthetic_simulator_type", default="linear", type=str)
    parser.add_argument("--dataset", default="tcga_100", type=str)
    parser.add_argument("--num_important_features", default=40, type=int)
    
    # This is the beta parameter
    parser.add_argument("--prop_scale", default=4, type=float)
    # This is the d parameter
    parser.add_argument("--shift", default=0, type=int)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = init_arg()
    if args.propensity_type == "pred":
        for seed in range(1, args.seeds):

            exp = PredictivePropensitySensitivity(
                seed=seed,
                synthetic_simulator_type=args.synthetic_simulator_type,
                propensity_type=args.propensity_type,
                prop_scale=args.prop_scale,
            )
            exp.run(
                dataset=args.dataset,
                train_ratio=args.train_ratio,
                num_important_features=args.num_important_features,
                shift=args.shift
            )
    else:
        assert args.propensity_type == "prog"
        for seed in range(1, args.seeds):
            
            exp = PrognosticPropensitySensitivity(
                seed=seed,
                synthetic_simulator_type=args.synthetic_simulator_type,
                propensity_type=args.propensity_type,
                prop_scale=args.prop_scale
            )
            exp.run(
                dataset=args.dataset,
                train_ratio=args.train_ratio,
                num_important_features=args.num_important_features,
                shift=args.shift
            )
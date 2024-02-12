import argparse
import sys
from typing import Any

import src.iterpretability.logger as log
from src.iterpretability.experiments import (
    PredictiveSensitivity,
    PropensitySensitivity,
    NonLinearitySensitivity,
)


def init_arg() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="propensity_sensitivity", type=str)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--synthetic_simulator_type", default="linear", type=str)

    parser.add_argument(
        "--dataset_list",
        nargs="+",
        type=str,
        default=["twins", "acic", "tcga_100", "news_100"],
    )

    parser.add_argument(
        "--num_important_features_list",
        nargs="+",
        type=int,
        default=[40, 8, 10, 20],
    )

    parser.add_argument(
        "--binary_outcome_list",
        nargs="+",
        type=bool,
        default=[False, False, False, False],
    )

    parser.add_argument(
        "--propensity_type",
        default="pred",
        type=str,
    )

    # Arguments for Propensity Sensitivity Experiment
    parser.add_argument("--predictive_scale", default=1.0, type=float)
    parser.add_argument(
        "--seed_list",
        default=30,
        type=int,
    )
    parser.add_argument(
        "--explainer_list",
        nargs="+",
        type=str,
        default=[
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
        ],
    )

    parser.add_argument("--run_name", type=str, default="results")
    parser.add_argument("--explainer_limit", type=int, default=1000)
    parser.add_argument("--prop_scale", type=float, default=1)
    parser.add_argument("--num_layers", type=int, default=0)
    parser.add_argument("--loss_mult", type=float, default=1)
    parser.add_argument("--penalty_disc", type=float, default=1)
    parser.add_argument("--scale_factor", type=float, default=1)
    parser.add_argument("--fully_synthetic", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    log.add(sink=sys.stderr, level="INFO")
    args = init_arg()
    for seed in range(1, args.seed_list):
        log.info(
            f"Experiment {args.experiment_name} with simulator {args.synthetic_simulator_type}, explainer limit {args.explainer_limit} and seed {seed}."
        )
        if args.experiment_name == "predictive_sensitivity":
            exp = PredictiveSensitivity(
                seed=seed,
                explainer_limit=args.explainer_limit,
                synthetic_simulator_type=args.synthetic_simulator_type,
            )
            for experiment_id in range(len(args.dataset_list)):
                log.info(
                    f"Running experiment for {args.dataset_list[experiment_id]}, {args.num_important_features_list[experiment_id]} with binary outcome {args.binary_outcome_list[experiment_id]}"
                )

                exp.run(
                    dataset=args.dataset_list[experiment_id],
                    train_ratio=args.train_ratio,
                    num_important_features=args.num_important_features_list[
                        experiment_id
                    ],
                    binary_outcome=args.binary_outcome_list[experiment_id],
                    explainer_list=args.explainer_list,
                )

        elif args.experiment_name == "nonlinearity_sensitivity":
            exp = NonLinearitySensitivity(
                seed=seed, explainer_limit=args.explainer_limit
            )
            for experiment_id in range(len(args.dataset_list)):
                log.info(
                    f"Running experiment for {args.dataset_list[experiment_id]}, "
                    f"{args.num_important_features_list[experiment_id]} important features "
                    f"with binary outcome {args.binary_outcome_list[experiment_id]}"
                )

                exp.run(
                    dataset=args.dataset_list[experiment_id],
                    train_ratio=args.train_ratio,
                    num_important_features=args.num_important_features_list[
                        experiment_id
                    ],
                    binary_outcome=args.binary_outcome_list[experiment_id],
                    explainer_list=args.explainer_list,
                )

        elif args.experiment_name == "propensity_sensitivity":
            print(args.fully_synthetic)
            exp = PropensitySensitivity(
                seed=seed,
                explainer_limit=args.explainer_limit,
                synthetic_simulator_type=args.synthetic_simulator_type,
                propensity_type=args.propensity_type,
                prop_scale=args.prop_scale,
                num_layers=args.num_layers,
                loss_mult=args.loss_mult,
                penalty_disc=args.penalty_disc,
                scale_factor=args.scale_factor
            )
            for experiment_id in range(len(args.dataset_list)):
                log.info(
                    f"Running experiment for {args.dataset_list[experiment_id]}, "
                    f"{args.num_important_features_list[experiment_id]}, "
                    f"propensity type {args.propensity_type}, with "
                    f"binary outcome {args.binary_outcome_list[experiment_id]}"
                )

                exp.run(
                    dataset=args.dataset_list[experiment_id],
                    train_ratio=args.train_ratio,
                    num_important_features=args.num_important_features_list[
                        experiment_id
                    ],
                    binary_outcome=args.binary_outcome_list[experiment_id],
                    explainer_list=args.explainer_list,
                    predictive_scale=args.predictive_scale,
                    fully_synthetic=args.fully_synthetic
                )

        else:
            raise ValueError("The experiment name is invalid.")

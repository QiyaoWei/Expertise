from pathlib import Path

import catenets.models as cate_models
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import src.iterpretability.logger as log
from src.iterpretability.explain import Explainer
from src.iterpretability.datasets.data_loader import load
from src.iterpretability.synthetic_simulate import (
    SyntheticSimulatorLinear,
    SyntheticSimulatorModulatedNonLinear,
)
from src.iterpretability.utils import (
    attribution_accuracy,
    compute_pehe,
)


class PredictiveSensitivity:
    """
    Sensitivity analysis for predictive scale.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 1024,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 0.5, 1, 2],
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = True,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
        ],
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
            )
        else:
            raise Exception("Unknown simulator type.")

        explainability_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                "TLearner": cate_models.torch.TLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "SLearner": cate_models.torch.SLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=1024,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "TARNet": cate_models.torch.TARNet(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_r=1,
                    n_layers_out=1,
                    n_units_out=100,
                    n_units_r=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "DRLearner": cate_models.torch.DRLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=1024,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "XLearner": cate_models.torch.XLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=1024,
                    batch_norm=False,
                    nonlin="relu",
                ),
            }

            learner_explainers = {}
            learner_explanations = {}

            for name in learners:
                log.info(f"Fitting {name}.")
                learners[name].fit(X=X_train, y=Y_train, w=W_train)
                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=explainer_list,
                )
                log.info(f"Explaining {name}.")
                learner_explanations[name] = learner_explainers[name].explain(
                    X_test[: self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for explainer_name in explainer_list:
                for learner_name in learners:
                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features = attribution_accuracy(
                        prog_features, attribution_est
                    )
                    cate_pred = learners[learner_name].predict(X=X_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_predictive_features,
                            acc_scores_prog_features,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "Pred features ACC",
                "Prog features ACC",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = self.save_path / "results/predictive_sensitivity"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_"
            f"{self.synthetic_simulator_type}_random_{random_feature_selection}_"
            f"binary_{binary_outcome}-seed{self.seed}.csv"
        )


class NonLinearitySensitivity:
    """
    Sensitivity analysis for nonlinearity in prognostic and predictive functions.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 1024,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        nonlinearity_scales: list = [0.0, 0.2, 0.5, 0.7, 1.0],
        predictive_scale: float = 1,
        synthetic_simulator_type: str = "random",
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.nonlinearity_scales = nonlinearity_scales
        self.predictive_scale = predictive_scale
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_100",
        num_important_features: int = 15,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
        ],
        train_ratio: float = 0.8,
        binary_outcome: bool = False,
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features}."
        )
        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)
        explainability_data = []

        for nonlinearity_scale in self.nonlinearity_scales:
            log.info(f"Now working with a nonlinearity scale {nonlinearity_scale}...")
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                selection_type=self.synthetic_simulator_type,
            )
            (
                X_train,
                W_train,
                Y_train,
                po0_train,
                po1_train,
                propensity_train,
            ) = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )
            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=self.predictive_scale,
                binary_outcome=binary_outcome,
            )

            log.info("Fitting and explaining learners...")
            learners = {
                "TLearner": cate_models.torch.TLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "SLearner": cate_models.torch.SLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=1024,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "TARNet": cate_models.torch.TARNet(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_r=1,
                    n_layers_out=1,
                    n_units_out=100,
                    n_units_r=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "DRLearner": cate_models.torch.DRLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=1024,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "XLearner": cate_models.torch.XLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=1024,
                    batch_norm=False,
                    nonlin="relu",
                ),
            }

            learner_explainers = {}
            learner_explanations = {}

            for name in learners:
                log.info(f"Fitting {name}.")
                learners[name].fit(X=X_train, y=Y_train, w=W_train)
                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=explainer_list,
                )
                log.info(f"Explaining {name}.")
                learner_explanations[name] = learner_explainers[name].explain(
                    X_test[: self.explainer_limit]
                )

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for explainer_name in explainer_list:
                for learner_name in learners:
                    attribution_est = np.abs(
                        learner_explanations[learner_name][explainer_name]
                    )
                    acc_scores_all_features = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    acc_scores_prog_features = attribution_accuracy(
                        prog_features, attribution_est
                    )

                    cate_pred = learners[learner_name].predict(X=X_test)

                    pehe_test = compute_pehe(cate_true=cate_test, cate_pred=cate_pred)

                    explainability_data.append(
                        [
                            nonlinearity_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_predictive_features,
                            acc_scores_prog_features,
                            pehe_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test)),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Nonlinearity Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "Pred features ACC",
                "Prog features ACC",
                "PEHE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],
        )

        results_path = (
            self.save_path
            / f"results/nonlinearity_sensitivity/{self.synthetic_simulator_type}"
        )
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path
            / f"{dataset}_{num_important_features}_binary_{binary_outcome}-seed{self.seed}.csv"
        )


class PropensitySensitivity:
    """
    Sensitivity analysis for confounding.
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 1024,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 1000,
        save_path: Path = Path.cwd(),
        num_interactions: int = 1,
        synthetic_simulator_type: str = "linear",
        propensity_type: str = "pred",
        propensity_scales: list = [0, 0.5, 1, 2, 5, 10],
        prop_scale: float = 1,
        num_layers: int = 0,
        loss_mult: int = 1,
        penalty_disc: float = 1,
        scale_factor: float = 1
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type
        self.propensity_type = propensity_type
        self.propensity_scales = propensity_scales
        self.prop_scale = prop_scale
        self.num_layers = num_layers
        self.loss_mult = loss_mult
        self.penalty_disc = penalty_disc
        self.scale_factor = scale_factor

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = False,
        predictive_scale: float = 1,
        nonlinearity_scale: float = 0.5,
        explainer_list: list = [
            "feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
        ],
        fully_synthetic: bool = True
    ) -> None:
        log.info(
            f"Using dataset {dataset} with num_important features = {num_important_features} and predictive scale {predictive_scale}."
        )

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if fully_synthetic:
            X_raw_train = np.random.normal(size=X_raw_train.shape)
            X_raw_test = np.random.normal(size=X_raw_test.shape)

        sim = SyntheticSimulatorLinear(
            X_raw_train,
            num_important_features=num_important_features,
            random_feature_selection=random_feature_selection,
            seed=self.seed,
            shift=0
        )
        X_train, _, _, po0_train, po1_train, propensity_train = sim.simulate_dataset(
            X_raw_train,
            predictive_scale=predictive_scale,
            binary_outcome=binary_outcome,
            treatment_assign=self.propensity_type,
            prop_scale=self.prop_scale,
            scale_factor=self.scale_factor
        )
        X_test, _, _, po0_test, po1_test, propensity_test = sim.simulate_dataset(
            X_raw_test,
            predictive_scale=predictive_scale,
            binary_outcome=binary_outcome,
            treatment_assign=self.propensity_type,
            prop_scale=self.prop_scale,
            scale_factor=self.scale_factor
        )
        # cate_test = sim.te(X_test)
        _, gt1_bins, gt0_bins = np.histogram2d(po1_test, po0_test)

        for shift in range(int(num_important_features / 4) + 1):

            if self.synthetic_simulator_type == "linear":
                sim = SyntheticSimulatorLinear(
                    X_raw_train,
                    num_important_features=num_important_features,
                    random_feature_selection=random_feature_selection,
                    seed=self.seed,
                    shift=shift
                )
            elif self.synthetic_simulator_type == "nonlinear":
                sim = SyntheticSimulatorModulatedNonLinear(
                    X_raw_train,
                    num_important_features=num_important_features,
                    non_linearity_scale=nonlinearity_scale,
                    seed=self.seed,
                    selection_type="random",
                    shift=shift
                )
            else:
                raise Exception("Unknown simulator type.")

            # sim.change_mask(shift)
            _, W_train, _, _, _, _ = sim.simulate_dataset(
                X_raw_train,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=self.prop_scale,
                scale_factor=self.scale_factor
            )

            _, W_test, _, _, _, _ = sim.simulate_dataset(
                X_raw_test,
                predictive_scale=predictive_scale,
                binary_outcome=binary_outcome,
                treatment_assign=self.propensity_type,
                prop_scale=self.prop_scale,
                scale_factor=self.scale_factor
            )

            Y_train = W_train * po1_train + (1 - W_train) * po0_train

            celearner = cate_models.torch.CENet(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_r=1,
                    n_layers_out=0,
                    n_units_out=100,
                    n_units_r=200,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                    prop_loss_multiplier=self.loss_mult,
                    n_layers_out_prop=self.num_layers
                )
            cfrlearner = cate_models.torch.TARNet(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_r=1,
                    n_layers_out=0,
                    n_units_out=100,
                    n_units_r=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                    penalty_disc=self.penalty_disc,
                )
            tarlearner = cate_models.torch.TARNet(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_r=1,
                    n_layers_out=0,
                    n_units_out=100,
                    n_units_r=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                )
            ipwlearner = cate_models.torch.TARNet(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_r=1,
                    n_layers_out=0,
                    n_units_out=100,
                    n_units_r=200,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                    prop_loss_multiplier=self.loss_mult,
                    ipw=True
                )
            
            celearner.fit(X=X_train, y=Y_train, w=W_train)
            cfrlearner.fit(X=X_train, y=Y_train, w=W_train)
            tarlearner.fit(X=X_train, y=Y_train, w=W_train)
            ipwlearner.fit(X=X_train, y=Y_train, w=W_train, p=propensity_train)

            actions, _ = np.histogram(W_test.numpy(), bins='auto')
            actions = actions / W_test.shape[0]
            actentropy = -np.sum(actions * np.ma.log(actions))

            num_ones = np.sum(W_test.numpy()) / W_test.shape[0]

            cond = -np.mean(propensity_test * np.log(propensity_test) + (1 - propensity_test) * np.log(1 - propensity_test))

            gt_uncond_hist, _, _ = np.histogram2d(po1_test, po0_test, bins=[gt1_bins, gt0_bins])
            gt_uncond_hist = gt_uncond_hist / po1_test.shape[0] #/ po0_test.shape[0]
            gt_uncond_hist = gt_uncond_hist[gt_uncond_hist != 0]
            gt_uncond_entropy = -np.sum(gt_uncond_hist * np.log(gt_uncond_hist))

            gt_cond_one1 = po1_test * W_test.numpy()
            gt_cond_one1 = gt_cond_one1[gt_cond_one1 != 0]
            gt_cond_one0 = po0_test * W_test.numpy()
            gt_cond_one0 = gt_cond_one0[gt_cond_one0 != 0]
            gt_one_hist, _, _ = np.histogram2d(gt_cond_one1, gt_cond_one0, bins=[gt1_bins, gt0_bins])
            gt_one_hist = gt_one_hist / gt_cond_one1.shape[0] #/ gt_cond_one0.shape[0]
            gt_one_hist = gt_one_hist[gt_one_hist != 0]
            gt_one_entropy = -np.sum(gt_one_hist * np.log(gt_one_hist))

            gt_cond_zero1 = po1_test * (1 - W_test).numpy()
            gt_cond_zero1 = gt_cond_zero1[gt_cond_zero1 != 0]
            gt_cond_zero0 = po0_test * (1 - W_test).numpy()
            gt_cond_zero0 = gt_cond_zero0[gt_cond_zero0 != 0]
            gt_zero_hist, _, _ = np.histogram2d(gt_cond_zero1, gt_cond_zero0, bins=[gt1_bins, gt0_bins])
            gt_zero_hist = gt_zero_hist / gt_cond_zero1.shape[0] #/ gt_cond_zero0.shape[0]
            gt_zero_hist = gt_zero_hist[gt_zero_hist != 0]
            gt_zero_entropy = -np.sum(gt_zero_hist * np.log(gt_zero_hist))

            gt_expertise = gt_uncond_entropy - num_ones * gt_one_entropy - (1 - num_ones) * gt_zero_entropy

            _, ce_y0, ce_y1 = celearner.predict(X_test, return_po=True)
            ce_uncond_hist, _, _ = np.histogram2d(ce_y1.detach().cpu().numpy(), ce_y0.detach().cpu().numpy(), bins=[gt1_bins, gt0_bins])
            ce_uncond_hist = ce_uncond_hist / ce_y1.shape[0] #/ ce_y0.shape[0]
            ce_uncond_hist = ce_uncond_hist[ce_uncond_hist != 0]
            ce_uncond_entropy = -np.sum(ce_uncond_hist * np.log(ce_uncond_hist))

            ce_cond_one1 = ce_y1.detach().cpu().numpy() * W_test.numpy()
            ce_cond_one1 = ce_cond_one1[ce_cond_one1 != 0]
            ce_cond_one0 = ce_y0.detach().cpu().numpy() * W_test.numpy()
            ce_cond_one0 = ce_cond_one0[ce_cond_one0 != 0]
            ce_one_hist, _, _ = np.histogram2d(ce_cond_one1, ce_cond_one0, bins=[gt1_bins, gt0_bins])
            ce_one_hist = ce_one_hist / ce_cond_one1.shape[0] #/ ce_cond_one0.shape[0]
            ce_one_hist = ce_one_hist[ce_one_hist != 0]
            ce_one_entropy = -np.sum(ce_one_hist * np.log(ce_one_hist))

            ce_cond_zero1 = ce_y1.detach().cpu().numpy() * (1 - W_test.numpy())
            ce_cond_zero1 = ce_cond_zero1[ce_cond_zero1 != 0]
            ce_cond_zero0 = ce_y0.detach().cpu().numpy() * (1 - W_test.numpy())
            ce_cond_zero0 = ce_cond_zero0[ce_cond_zero0 != 0]
            ce_zero_hist, _, _ = np.histogram2d(ce_cond_zero1, ce_cond_zero0, bins=[gt1_bins, gt0_bins])
            ce_zero_hist = ce_zero_hist / ce_cond_zero1.shape[0] #/ ce_cond_zero0.shape[0]
            ce_zero_hist = ce_zero_hist[ce_zero_hist != 0]
            ce_zero_entropy = -np.sum(ce_zero_hist * np.log(ce_zero_hist))

            ce_expertise = ce_uncond_entropy - num_ones * ce_one_entropy - (1 - num_ones) * ce_zero_entropy

            _, tar_y0, tar_y1 = tarlearner.predict(X_test, return_po=True)
            tar_uncond_hist, _, _ = np.histogram2d(tar_y1.detach().cpu().numpy(), tar_y0.detach().cpu().numpy(), bins=[gt1_bins, gt0_bins])
            tar_uncond_hist = tar_uncond_hist / tar_y1.shape[0] #/ tar_y0.shape[0]
            tar_uncond_hist = tar_uncond_hist[tar_uncond_hist != 0]
            tar_uncond_entropy = -np.sum(tar_uncond_hist * np.log(tar_uncond_hist))

            tar_cond_one1 = tar_y1.detach().cpu().numpy() * W_test.numpy()
            tar_cond_one1 = tar_cond_one1[tar_cond_one1 != 0]
            tar_cond_one0 = tar_y0.detach().cpu().numpy() * W_test.numpy()
            tar_cond_one0 = tar_cond_one0[tar_cond_one0 != 0]
            tar_one_hist, _, _ = np.histogram2d(tar_cond_one1, tar_cond_one0, bins=[gt1_bins, gt0_bins])
            tar_one_hist = tar_one_hist / tar_cond_one1.shape[0] #/ tar_cond_one0.shape[0]
            tar_one_hist = tar_one_hist[tar_one_hist != 0]
            tar_one_entropy = -np.sum(tar_one_hist * np.log(tar_one_hist))

            tar_cond_zero1 = tar_y1.detach().cpu().numpy() * (1 - W_test.numpy())
            tar_cond_zero1 = tar_cond_zero1[tar_cond_zero1 != 0]
            tar_cond_zero0 = tar_y0.detach().cpu().numpy() * (1 - W_test.numpy())
            tar_cond_zero0 = tar_cond_zero0[tar_cond_zero0 != 0]
            tar_zero_hist, _, _ = np.histogram2d(tar_cond_zero1, tar_cond_zero0, bins=[gt1_bins, gt0_bins])
            tar_zero_hist = tar_zero_hist / tar_cond_zero1.shape[0] #/ tar_cond_zero0.shape[0]
            tar_zero_hist = tar_zero_hist[tar_zero_hist != 0]
            tar_zero_entropy = -np.sum(tar_zero_hist * np.log(tar_zero_hist))

            tar_expertise = tar_uncond_entropy - num_ones * tar_one_entropy - (1 - num_ones) * tar_zero_entropy

            _, cfr_y0, cfr_y1 = cfrlearner.predict(X_test, return_po=True)
            cfr_uncond_hist, _, _ = np.histogram2d(cfr_y1.detach().cpu().numpy(), cfr_y0.detach().cpu().numpy(), bins=[gt1_bins, gt0_bins])
            cfr_uncond_hist = cfr_uncond_hist / cfr_y1.shape[0] #/ cfr_y0.shape[0]
            cfr_uncond_hist = cfr_uncond_hist[cfr_uncond_hist != 0]
            cfr_uncond_entropy = -np.sum(cfr_uncond_hist * np.log(cfr_uncond_hist))

            cfr_cond_one1 = cfr_y1.detach().cpu().numpy() * W_test.numpy()
            cfr_cond_one1 = cfr_cond_one1[cfr_cond_one1 != 0]
            cfr_cond_one0 = cfr_y0.detach().cpu().numpy() * W_test.numpy()
            cfr_cond_one0 = cfr_cond_one0[cfr_cond_one0 != 0]
            cfr_one_hist, _, _ = np.histogram2d(cfr_cond_one1, cfr_cond_one0, bins=[gt1_bins, gt0_bins])
            cfr_one_hist = cfr_one_hist / cfr_cond_one1.shape[0] #/ cfr_cond_one0.shape[0]
            cfr_one_hist = cfr_one_hist[cfr_one_hist != 0]
            cfr_one_entropy = -np.sum(cfr_one_hist * np.log(cfr_one_hist))

            cfr_cond_zero1 = cfr_y1.detach().cpu().numpy() * (1 - W_test.numpy())
            cfr_cond_zero1 = cfr_cond_zero1[cfr_cond_zero1 != 0]
            cfr_cond_zero0 = cfr_y0.detach().cpu().numpy() * (1 - W_test.numpy())
            cfr_cond_zero0 = cfr_cond_zero0[cfr_cond_zero0 != 0]
            cfr_zero_hist, _, _ = np.histogram2d(cfr_cond_zero1, cfr_cond_zero0, bins=[gt1_bins, gt0_bins])
            cfr_zero_hist = cfr_zero_hist / cfr_cond_zero1.shape[0] #/ cfr_cond_zero0.shape[0]
            cfr_zero_hist = cfr_zero_hist[cfr_zero_hist != 0]
            cfr_zero_entropy = -np.sum(cfr_zero_hist * np.log(cfr_zero_hist))

            cfr_expertise = cfr_uncond_entropy - num_ones * cfr_one_entropy - (1 - num_ones) * cfr_zero_entropy

            _, ipw_y0, ipw_y1 = ipwlearner.predict(X_test, return_po=True)
            ipw_uncond_hist, _, _ = np.histogram2d(ipw_y1.detach().cpu().numpy(), ipw_y0.detach().cpu().numpy(), bins=[gt1_bins, gt0_bins])
            ipw_uncond_hist = ipw_uncond_hist / ipw_y1.shape[0] #/ ipw_y0.shape[0]
            ipw_uncond_hist = ipw_uncond_hist[ipw_uncond_hist != 0]
            ipw_uncond_entropy = -np.sum(ipw_uncond_hist * np.log(ipw_uncond_hist))

            ipw_cond_one1 = ipw_y1.detach().cpu().numpy() * W_test.numpy()
            ipw_cond_one1 = ipw_cond_one1[ipw_cond_one1 != 0]
            ipw_cond_one0 = ipw_y0.detach().cpu().numpy() * W_test.numpy()
            ipw_cond_one0 = ipw_cond_one0[ipw_cond_one0 != 0]
            ipw_one_hist, _, _ = np.histogram2d(ipw_cond_one1, ipw_cond_one0, bins=[gt1_bins, gt0_bins])
            ipw_one_hist = ipw_one_hist / ipw_cond_one1.shape[0] #/ ipw_cond_one0.shape[0]
            ipw_one_hist = ipw_one_hist[ipw_one_hist != 0]
            ipw_one_entropy = -np.sum(ipw_one_hist * np.log(ipw_one_hist))

            ipw_cond_zero1 = ipw_y1.detach().cpu().numpy() * (1 - W_test.numpy())
            ipw_cond_zero1 = ipw_cond_zero1[ipw_cond_zero1 != 0]
            ipw_cond_zero0 = ipw_y0.detach().cpu().numpy() * (1 - W_test.numpy())
            ipw_cond_zero0 = ipw_cond_zero0[ipw_cond_zero0 != 0]
            ipw_zero_hist, _, _ = np.histogram2d(ipw_cond_zero1, ipw_cond_zero0, bins=[gt1_bins, gt0_bins])
            ipw_zero_hist = ipw_zero_hist / ipw_cond_zero1.shape[0] #/ ipw_cond_zero1.shape[0]
            ipw_zero_hist = ipw_zero_hist[ipw_zero_hist != 0]
            ipw_zero_entropy = -np.sum(ipw_zero_hist * np.log(ipw_zero_hist))

            ipw_expertise = ipw_uncond_entropy - num_ones * ipw_one_entropy - (1 - num_ones) * ipw_zero_entropy

            # assert ce_expertise > 0
            # assert tar_expertise > 0
            # assert cfr_expertise > 0
            # assert ipw_expertise > 0

            cate_test = sim.te(X_test)
            cepred = celearner.predict(X=X_test)
            cfrpred = cfrlearner.predict(X=X_test)
            tarpred = tarlearner.predict(X=X_test)
            ipwpred = ipwlearner.predict(X=X_test)
            cepehe = compute_pehe(cate_true=cate_test, cate_pred=cepred)
            cfrpehe = compute_pehe(cate_true=cate_test, cate_pred=cfrpred)
            tarpehe = compute_pehe(cate_true=cate_test, cate_pred=tarpred)
            ipwpehe = compute_pehe(cate_true=cate_test, cate_pred=ipwpred)

            cey = compute_pehe(po1_test, ce_y1, W_test) + compute_pehe(po0_test, ce_y0, 1 - W_test)
            tary = compute_pehe(po1_test, tar_y1, W_test) + compute_pehe(po0_test, tar_y0, 1 - W_test)
            cfry = compute_pehe(po1_test, cfr_y1, W_test) + compute_pehe(po0_test, cfr_y0, 1 - W_test)
            ipwy = compute_pehe(po1_test, ipw_y1, W_test) + compute_pehe(po0_test, ipw_y0, 1 - W_test)

            # with open("exp" + str(self.prop_scale) + ".txt", "a") as myfile:
            #     if shift == 0:
            #         myfile.write("seed: " + str(self.seed) + "\n")
            #     myfile.write(self.propensity_type + "gt" + str(gt_expertise) + "cond" + str(cond) + "av" + str(actentropy) + "\n")

            with open(str(self.propensity_type) + str(self.prop_scale) + "cate.txt", "a") as myfile:
                if shift == 0:
                    myfile.write("seed: " + str(self.seed) + "\n")
                myfile.write(self.propensity_type + " ce" + str(cepehe) + "cfr" + str(cfrpehe) + "tar" + str(tarpehe) + "ipw" + str(ipwpehe) + "\n")

            with open(str(self.propensity_type) + str(self.prop_scale) + "ypi.txt", "a") as myfile:
                if shift == 0:
                    myfile.write("seed: " + str(self.seed) + "\n")
                myfile.write(self.propensity_type + " ce" + str(cey) + "cfr" + str(cfry) + "tar" + str(tary) + "ipw" + str(ipwy) + "\n")

            with open("entropy" + str(self.propensity_type) + str(num_important_features) + "x" + str(self.prop_scale) + ".txt", "a") as myfile:
                if shift == 0:
                    myfile.write("seed: " + str(self.seed) + "\n")
                myfile.write(self.propensity_type + " ce" + str(ce_expertise) + "cfr" + str(cfr_expertise) + "tar" + str(tar_expertise) + "ipw" + str(ipw_expertise) + "gt" + str(gt_expertise) + "av" + str(actentropy) + "\n")
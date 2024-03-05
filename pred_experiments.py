from pathlib import Path

import catenets.models as cate_models
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append(sys.path[0] + '/datasets')
from data_loader import load
from synthetic_simulate import (
    SyntheticSimulatorLinear,
    SyntheticSimulatorModulatedNonLinear,
)
from utils import compute_pehe

class PredictivePropensitySensitivity:
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
        shift: int = 0,
        fully_synthetic: bool = True
    ) -> None:

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        # fully synthetic flag to test toy case
        # if fully_synthetic:
        #     X_raw_train = np.random.normal(size=X_raw_train.shape)
        #     X_raw_test = np.random.normal(size=X_raw_test.shape)

        # This is the ground truth, therefore d=0
<<<<<<< HEAD
        sim = SyntheticSimulatorLinear(
            X_raw_train,
            num_important_features=num_important_features,
            random_feature_selection=random_feature_selection,
            seed=self.seed,
            shift=0
        )
=======

        if self.synthetic_simulator_type == "linear":
            sim = SyntheticSimulatorLinear(
                X_raw_train,
                num_important_features=num_important_features,
                random_feature_selection=random_feature_selection,
                seed=self.seed,
                shift=0
            )
        elif self.synthetic_simulator_type == "nonlinear":
            sim = SyntheticSimulatorModulatedNonLinear(
                X_raw_train,
                num_important_features=num_important_features,
                non_linearity_scale=nonlinearity_scale,
                seed=self.seed,
                shift=0
            )
            
>>>>>>> 72bf8dc (First commit)
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

        # prognostic expertise calculation
        _, gt_bins = np.histogram(po1_test - po0_test, bins="auto")

        # Our training data will change based on d
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
<<<<<<< HEAD
                selection_type="random",
=======
>>>>>>> 72bf8dc (First commit)
                shift=shift
            )
        else:
            raise Exception("Unknown simulator type.")

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

        # cond = -np.mean(propensity_test * np.log(propensity_test) + (1 - propensity_test) * np.log(1 - propensity_test))

        gt_uncond_result = po1_test - po0_test
        gt_uncond_hist, gt_uncond_bins = np.histogram(gt_uncond_result, bins=gt_bins)
        gt_uncond_hist = gt_uncond_hist / gt_uncond_result.shape[0]
        gt_uncond_hist = gt_uncond_hist[gt_uncond_hist != 0]
        gt_uncond_entropy = -np.sum(gt_uncond_hist * np.log(gt_uncond_hist))

        gt_cond_one = (po1_test - po0_test) * W_test.numpy()
        gt_cond_one = gt_cond_one[gt_cond_one != 0]
        gt_one_hist, _ = np.histogram(gt_cond_one, bins=gt_bins)
        gt_one_hist = gt_one_hist / gt_cond_one.shape[0]
        gt_one_hist = gt_one_hist[gt_one_hist != 0]
        gt_one_entropy = -np.sum(gt_one_hist * np.log(gt_one_hist))

        gt_cond_zero = (po1_test - po0_test) * (1 - W_test).numpy()
        gt_cond_zero = gt_cond_zero[gt_cond_zero != 0]
        gt_zero_hist, _ = np.histogram(gt_cond_zero, bins=gt_bins)
        gt_zero_hist = gt_zero_hist / gt_cond_zero.shape[0]
        gt_zero_hist = gt_zero_hist[gt_zero_hist != 0]
        gt_zero_entropy = -np.sum(gt_zero_hist * np.log(gt_zero_hist))

        gt_expertise = gt_uncond_entropy - num_ones * gt_one_entropy - (1 - num_ones) * gt_zero_entropy

        _, y0, y1 = celearner.predict(X_test, return_po=True)
        ce_uncond_result = (y1 - y0).detach().cpu().numpy()
        ce_uncond_hist, ce_uncond_bins = np.histogram(ce_uncond_result, bins=gt_bins)
        ce_uncond_hist = ce_uncond_hist / ce_uncond_result.shape[0]
        ce_uncond_hist = ce_uncond_hist[ce_uncond_hist != 0]
        ce_uncond_entropy = -np.sum(ce_uncond_hist * np.log(ce_uncond_hist))

        ce_cond_one = (y1 - y0).detach().cpu().numpy() * W_test.numpy()
        ce_cond_one = ce_cond_one[ce_cond_one != 0]
        ce_one_hist, _ = np.histogram(ce_cond_one, bins=gt_bins)
        ce_one_hist = ce_one_hist / ce_cond_one.shape[0]
        ce_one_hist = ce_one_hist[ce_one_hist != 0]
        ce_one_entropy = -np.sum(ce_one_hist * np.log(ce_one_hist))

        ce_cond_zero = (y1 - y0).detach().cpu().numpy() * (1 - W_test.numpy())
        ce_cond_zero = ce_cond_zero[ce_cond_zero != 0]
        ce_zero_hist, _ = np.histogram(ce_cond_zero, bins=gt_bins)
        ce_zero_hist = ce_zero_hist / ce_cond_zero.shape[0]
        ce_zero_hist = ce_zero_hist[ce_zero_hist != 0]
        ce_zero_entropy = -np.sum(ce_zero_hist * np.log(ce_zero_hist))

        ce_expertise = ce_uncond_entropy - num_ones * ce_one_entropy - (1 - num_ones) * ce_zero_entropy

        _, y0, y1 = tarlearner.predict(X_test, return_po=True)
        tar_uncond_result = (y1 - y0).detach().cpu().numpy()
        tar_uncond_hist, _ = np.histogram(tar_uncond_result, bins=gt_bins)
        tar_uncond_hist = tar_uncond_hist / tar_uncond_result.shape[0]
        tar_uncond_hist = tar_uncond_hist[tar_uncond_hist != 0]
        tar_uncond_entropy = -np.sum(tar_uncond_hist * np.log(tar_uncond_hist))

        tar_cond_one = (y1 - y0).detach().cpu().numpy() * W_test.numpy()
        tar_cond_one = tar_cond_one[tar_cond_one != 0]
        tar_one_hist, _ = np.histogram(tar_cond_one, bins=gt_bins)
        tar_one_hist = tar_one_hist / tar_cond_one.shape[0]
        tar_one_hist = tar_one_hist[tar_one_hist != 0]
        tar_one_entropy = -np.sum(tar_one_hist * np.log(tar_one_hist))

        tar_cond_zero = (y1 - y0).detach().cpu().numpy() * (1 - W_test.numpy())
        tar_cond_zero = tar_cond_zero[tar_cond_zero != 0]
        tar_zero_hist, _ = np.histogram(tar_cond_zero, bins=gt_bins)
        tar_zero_hist = tar_zero_hist / tar_cond_zero.shape[0]
        tar_zero_hist = tar_zero_hist[tar_zero_hist != 0]
        tar_zero_entropy = -np.sum(tar_zero_hist * np.log(tar_zero_hist))

        tar_expertise = tar_uncond_entropy - num_ones * tar_one_entropy - (1 - num_ones) * tar_zero_entropy

        _, y0, y1 = cfrlearner.predict(X_test, return_po=True)
        cfr_uncond_result = (y1 - y0).detach().cpu().numpy()
        cfr_uncond_hist, _ = np.histogram(cfr_uncond_result, bins=gt_bins)
        cfr_uncond_hist = cfr_uncond_hist / cfr_uncond_result.shape[0]
        cfr_uncond_hist = cfr_uncond_hist[cfr_uncond_hist != 0]
        cfr_uncond_entropy = -np.sum(cfr_uncond_hist * np.log(cfr_uncond_hist))

        cfr_cond_one = (y1 - y0).detach().cpu().numpy() * W_test.numpy()
        cfr_cond_one = cfr_cond_one[cfr_cond_one != 0]
        cfr_one_hist, _ = np.histogram(cfr_cond_one, bins=gt_bins)
        cfr_one_hist = cfr_one_hist / cfr_cond_one.shape[0]
        cfr_one_hist = cfr_one_hist[cfr_one_hist != 0]
        cfr_one_entropy = -np.sum(cfr_one_hist * np.log(cfr_one_hist))

        cfr_cond_zero = (y1 - y0).detach().cpu().numpy() * (1 - W_test.numpy())
        cfr_cond_zero = cfr_cond_zero[cfr_cond_zero != 0]
        cfr_zero_hist, _ = np.histogram(cfr_cond_zero, bins=gt_bins)
        cfr_zero_hist = cfr_zero_hist / cfr_cond_zero.shape[0]
        cfr_zero_hist = cfr_zero_hist[cfr_zero_hist != 0]
        cfr_zero_entropy = -np.sum(cfr_zero_hist * np.log(cfr_zero_hist))

        cfr_expertise = cfr_uncond_entropy - num_ones * cfr_one_entropy - (1 - num_ones) * cfr_zero_entropy

        _, y0, y1 = ipwlearner.predict(X_test, return_po=True)
        ipw_uncond_result = (y1 - y0).detach().cpu().numpy()
        ipw_uncond_hist, _ = np.histogram(ipw_uncond_result, bins=gt_bins)
        ipw_uncond_hist = ipw_uncond_hist / ipw_uncond_result.shape[0]
        ipw_uncond_hist = ipw_uncond_hist[ipw_uncond_hist != 0]
        ipw_uncond_entropy = -np.sum(ipw_uncond_hist * np.log(ipw_uncond_hist))

        ipw_cond_one = (y1 - y0).detach().cpu().numpy() * W_test.numpy()
        ipw_cond_one = ipw_cond_one[ipw_cond_one != 0]
        ipw_one_hist, _ = np.histogram(ipw_cond_one, bins=gt_bins)
        ipw_one_hist = ipw_one_hist / ipw_cond_one.shape[0]
        ipw_one_hist = ipw_one_hist[ipw_one_hist != 0]
        ipw_one_entropy = -np.sum(ipw_one_hist * np.log(ipw_one_hist))

        ipw_cond_zero = (y1 - y0).detach().cpu().numpy() * (1 - W_test.numpy())
        ipw_cond_zero = ipw_cond_zero[ipw_cond_zero != 0]
        ipw_zero_hist, _ = np.histogram(ipw_cond_zero, bins=gt_bins)
        ipw_zero_hist = ipw_zero_hist / ipw_cond_zero.shape[0]
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

        with open(str(self.propensity_type) + str(self.synthetic_simulator_type) + str(dataset) + str(self.prop_scale) + str(shift) + ".txt", "a") as myfile:
            if shift == 0:
                myfile.write("seed: " + str(self.seed) + "\n")
            myfile.write("ce" + str(cepehe) + "cfr" + str(cfrpehe) + "tar" + str(tarpehe) + "ipw" + str(ipwpehe) + "\n")

        with open("entropy" + str(self.propensity_type) + str(self.synthetic_simulator_type) + str(dataset) + str(self.prop_scale) + str(shift) + ".txt", "a") as myfile:
            if shift == 0:
                myfile.write("seed: " + str(self.seed) + "\n")
            myfile.write("ce" + str(ce_expertise) + "cfr" + str(cfr_expertise) + "tar" + str(tar_expertise) + "ipw" + str(ipw_expertise) + "gt" + str(gt_expertise) + "av" + str(actentropy) + "\n")
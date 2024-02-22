# stdlib
import random
from typing import Optional

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error

def compute_pehe(
    cate_true: np.ndarray,
    cate_pred: torch.Tensor,
    w = None
) -> tuple:
    if w != None:
        pehe = np.sqrt(mean_squared_error(cate_true * w.numpy(), cate_pred.detach().cpu().numpy() * w.numpy()))
    else:
        pehe = np.sqrt(mean_squared_error(cate_true, cate_pred.detach().cpu().numpy()))
    return pehe

# def dataframe_line_plot(
#     df: pd.DataFrame,
#     x_axis: str,
#     y_axis: str,
#     explainers: list,
#     learners: list,
#     x_logscale: bool = True,
#     aggregate: bool = False,
#     aggregate_type: str = "mean",
# ) -> plt.Figure:
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     sns.set_style("white")
#     for learner_name in learners:
#         for explainer_name in explainers:
#             sub_df = df.loc[
#                 (df["Learner"] == learner_name) & (df["Explainer"] == explainer_name)
#             ]
#             if aggregate:
#                 sub_df = sub_df.groupby(x_axis).agg(aggregate_type).reset_index()
#             x_values = sub_df.loc[:, x_axis].values
#             y_values = sub_df.loc[:, y_axis].values
#             ax.plot(
#                 x_values,
#                 y_values,
#                 color=learner_colors[learner_name],
#                 marker=explainer_symbols[explainer_name],
#             )

#     learner_lines = [
#         Line2D([0], [0], color=learner_colors[learner_name], lw=2)
#         for learner_name in learners
#     ]
#     explainer_lines = [
#         Line2D([0], [0], color="black", marker=explainer_symbols[explainer_name])
#         for explainer_name in explainers
#     ]

#     legend_learners = plt.legend(
#         learner_lines, learners, loc="lower left", bbox_to_anchor=(1.04, 0.7)
#     )
#     legend_explainers = plt.legend(
#         explainer_lines,
#         [abbrev_dict[explainer_name] for explainer_name in explainers],
#         loc="lower left",
#         bbox_to_anchor=(1.04, 0),
#     )
#     plt.subplots_adjust(right=0.75)
#     ax.add_artist(legend_learners)
#     ax.add_artist(legend_explainers)
#     if x_logscale:
#         ax.set_xscale("log")
#     ax.set_xlabel(x_axis)
#     ax.set_ylabel(y_axis)
#     return fig
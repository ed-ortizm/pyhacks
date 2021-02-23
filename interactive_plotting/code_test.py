#! /usr/bin/env python3
import glob
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pickle

from interactive_lib import ExplanationData
from interactive_lib import PlotData
######################################################################
ti = time.time()
################################################################################
explanation_file = 'spec-2283-53729-0329_exp_dict.dill'
#'spec-8744-58124-0638_exp_dict.dill'
#'spec-0967-52636-0339_exp_dict.dill'
#'spec-7258-56605-0800_exp_dict.dill'
data_explanation = ExplanationData(explanation_file=explanation_file)
sdss_name = data_explanation.sdss_name
spec = data_explanation.spec
n_line = 0
(wave_exp, flux_exp, weights_exp, k_width,
    metric, feature_selection) = data_explanation.get_explanation_data(n_line)
################################################################################
# vmin vmax for colorbar
explanations_dict = data_explanation.get_serialized_data()

weights_explanation = []

for value in explanations_dict.values():
    array_explanation = value[1]
    weights_explanation.append(np.nanmin(array_explanation[:, 1]))
    weights_explanation.append(np.nanmax(array_explanation[:, 1]))

w_min = min(weights_explanation)
w_max = max(weights_explanation)
print(w_min, w_max)
################################################################################
# Data plot
explanation_plot = PlotData(spec, sdss_name, vmin=w_min, vmax=w_max)

fig, ax, ax_cb, line, scatter = explanation_plot.plot_explanation(
    wave_exp, flux_exp, weights_exp,
    k_width, feature_selection, metric,
    s=7., linewidth=1., alpha=0.1)
################################################################################

ax_line_plus = plt.axes([0.825, 0.8, 0.05, 0.05])
button_line_plus = Button(ax_line_plus, 'line +')

ax_line_minus = plt.axes([0.825, 0.725, 0.05, 0.05])
button_line_minus = Button(ax_line_minus, 'line -')

def line_plus(event):

    global n_line
    n_line += 1
    print(n_line)
    (wave_exp, flux_exp, weights_exp, k_width, metric,
        feature_selection) = data_explanation.get_explanation_data(n_line)

    # a = np.sort(weights_exp)
    # print([f'{i:.2E}' for i in a[:3]])
    # print([f'{i:.2E}' for i in a[-3:]])

    scatter_updated_data = np.array(([wave_exp,flux_exp])).T
    scatter.set_offsets(scatter_updated_data)

    norm = plt.Normalize(vmin=w_min, vmax=w_max)
    colors = plt.cm.plasma(norm(weights_exp))
    scatter.set_facecolor(colors)

    plt.draw()


def line_minus(event):

    global n_line
    n_line -= 1
    print(n_line)
    (wave_exp, flux_exp, weights_exp, k_width, metric,
        feature_selection) = data_explanation.get_explanation_data(n_line)

    scatter_updated_data = np.array(([wave_exp,flux_exp])).T
    scatter.set_offsets(scatter_updated_data)

    norm = plt.Normalize(vmin=w_min, vmax=w_max)
    colors = plt.cm.plasma(norm(weights_exp))
    scatter.set_facecolor(colors)

    plt.draw()

button_line_plus.on_clicked(line_plus)
button_line_minus.on_clicked(line_minus)

plt.show()
################################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")

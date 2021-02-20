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
from interactive_lib import plot_explanation
######################################################################
ti = time.time()
################################################################################
explanation_file = "spec-0967-52636-0339_exp_dict.dill"
"spec-7258-56605-0800_exp_dict.dill"
data = ExplanationData(explanation_file=explanation_file)
sdss_name = data.sdss_name
spec = data.spec
n_line = 0
(wave_exp, flux_exp, weights_exp,
    k_width, metric, feature_selection) = data.get_explanation_data(n_line)

################################################################################
fig, ax, scatter, cbar = plot_explanation(spec, wave_exp, flux_exp, weights_exp, sdss_name,
    k_width, feature_selection, metric,
    s=3., linewidth=1., alpha=0.7, cmap='plasma_r')

################################################################################
ax_key_plus = plt.axes([0.7, 0.05, 0.1, 0.075])
b_key_plus = Button(ax_key_plus, 'Key +')

ax_key_minus = plt.axes([0.81, 0.05, 0.1, 0.075])
b_key_minus = Button(ax_key_minus, 'Key -')

def key_plus(event):
    global n_line, scatter, cbar
    n_line += 2
    print(n_line)
    (wave_exp, flux_exp, weights_exp,
        k_width, metric, feature_selection) = data.get_explanation_data(n_line)

    scatter_updated_data = np.array(([wave_exp,flux_exp])).T
    scatter.set_offsets(scatter_updated_data)
    c = weights_exp/np.nanmax(weights_exp)
    print(c.min())
    scatter.set_array(c)
    cbar.update_normal()
    plt.draw()


def key_minus(event):
    global n_line
    n_line -= 1
    print(n_line)
    (wave_exp, flux_exp, weights_exp,
        k_width, metric, feature_selection) = data.get_explanation_data(n_line)

b_key_plus.on_clicked(key_plus)
b_key_minus.on_clicked(key_minus)

plt.show()
################################################################################
tf = time.time()
print(f"Running time: {tf-ti:.2f}")

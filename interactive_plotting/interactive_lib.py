import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pickle

class Interactive:
    def __init__(self, explanation_file=None):

        if explanation_file == None:
            self.explanation_file = '/home/edgar/zorro/AEsII/xAI/lime/results/spec-1246-54478-0144_exp_dict.dill'
        else:
            self_explanation_file = explanation_file

        self.spec, self.exp_dict, self.sdss_name = None, None, None
        self.line = None

    def _get_exp_data(self, key):

        exp_array = self.exp_dict[f'{key}'][1]
        wave_exp = exp_array[:, 0].astype(np.int)
        flx_exp = self.spec[wave_exp]
        weights_exp = exp_array[:, 1]

        k_width = float(self.exp_dict[f'{key}'][0])
        k_width = float(f'{k_width/1.:.2f}')
        feature_selection = self.exp_dict[f'{key}'][4]
        metric = self.exp_dict[f'{key}'][3]

        return wave_exp, flx_exp, weights_exp, k_width, feature_selection, metric

    def _get_exp_dict(self):

         sdss_directory = "/home/edgar/zorro/SDSSdata/data_proc"
         sdss_name = self.explanation_file.split('/')[-1].split('_')[0]
         with open(f'{self.explanation_file}', 'rb') as file:
             exp_dict = pickle.load(file)

         spec = np.load(f'{sdss_directory}/{sdss_name}.npy')

         return spec, exp_dict, sdss_name

    def increase_key(self, event):

        self.line.set_ydata(self.spec/self.spec)
        plt.draw()

    def decrease_key(self, event):

        self.line.set_ydata(self.spec/self.spec)
        plt.draw()

    def plot_explanation(self, s=3, linewidth=1., alpha=.3, cmap='plasma_r'):

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        self.spec, self.exp_dict, self.sdss_name = self._get_exp_dict()
        (wave_exp, flx_exp, weights_exp, k_width,
            feature_selection, metric) = self._get_exp_data(key=20)


        c = weights_exp/np.max(weights_exp)

        self.line, = plt.plot(self.spec, linewidth=linewidth, alpha=alpha)
        plt.scatter(wave_exp, flx_exp, s=s, c=c, cmap=cmap)
        plt.title(f'{self.sdss_name}: {metric}, {feature_selection}, k_width={k_width}')

        plt.colorbar()

        ax_increase = plt.axes([0.81, 0.05, 0.1, 0.075])
        button_increase = Button(ax_increase, 'key +')
        button_increase.on_clicked(self.increase_key)

        ax_decrease = plt.axes([0.7, 0.05, 0.1, 0.075])
        button_decrease = Button(ax_decrease, 'key -')
        button_decrease.on_clicked(self.decrease_key)

        plt.show()

###############################################################################
class ExplanationData:

    def __init__(self, explanation_file):

        self.explanation_file = explanation_file
        self.sdss_directory = "/home/edgar/Documents/pyhacks/interactive_plotting"
        # "/home/edgar/zorro/SDSSdata/data_proc"
        self.sdss_name = self.explanation_file.split('_')[0]
        # self.explanation_file.split('/')[-1].split('_')[0]
        self.spec = np.load(f'{self.sdss_directory}/{self.sdss_name}.npy')

    def get_explanation_data(self, n_line):

        explanation_dictionary = self._get_serialized_data()

        kernel_width = explanation_dictionary[f'{n_line}'][0]
        kernel_width = float(kernel_width)

        array_explanation = explanation_dictionary[f'{n_line}'][1]
        wave_explanation = array_explanation[:, 0].astype(np.int)
        flux_explanation = self.spec[wave_explanation]
        weights_explanation = array_explanation[:, 1]
        metric = explanation_dictionary[f'{n_line}'][3]
        feature_selection = explanation_dictionary[f'{n_line}'][4]

        return (wave_explanation,
                flux_explanation,
                weights_explanation,
                kernel_width, metric, feature_selection)

    def _get_serialized_data(self):

         with open(f'{self.explanation_file}', 'rb') as file:
             return pickle.load(file)

###############################################################################

def plot_explanation(spec, wave_exp, flx_exp, weights_exp, sdss_name,
    kernel_width, feature_selection, metric , s=3., linewidth=1.,
    alpha=0.7, cmap='plasma_r'):

    c = weights_exp/np.nanmax(weights_exp)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    line, = ax.plot(spec, linewidth=linewidth, alpha=alpha)
    scatter = ax.scatter(wave_exp, flx_exp, s=s, c=c, cmap=cmap)
    ax.set_title(
    f'{sdss_name}: {metric}, {feature_selection}, k_width={kernel_width}')

    cbar = fig.colorbar(scatter, orientation='vertical')
    return fig, ax, scatter, cbar

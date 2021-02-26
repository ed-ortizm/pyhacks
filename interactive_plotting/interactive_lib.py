import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pickle
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

        explanation_dictionary = self.get_serialized_data()

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

    def get_serialized_data(self):

         with open(f'{self.explanation_file}', 'rb') as file:
             return pickle.load(file)

###############################################################################
class PlotData:

    def __init__(self, spec, sdss_name, vmin, vmax):
        self.spec = spec
        self.sdss_name = sdss_name
        self.vmin = vmin
        self.vmax = vmax
        self._fig = None
        self._cmap = None

    def _colorbar_explanation(self):
        # Make axes with dimensions as desired.
        ax_cb = self._fig.add_axes([0.91, 0.05, 0.03, 0.9])

        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        self._cmap = mpl.cm.plasma
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=self._cmap,
                                        norm=norm,
                                        orientation='vertical', extend='both')
        cb.set_label('Normalized weights')


        return cb

    def plot_explanation(self,
        wave_exp, flx_exp, weights_explanation,
        kernel_width, feature_selection, metric,
        s=3., linewidth=1., alpha=0.7):

        a = np.sort(weights_explanation)
        print([f'{i:.2E}' for i in a[:2]])
        print([f'{i:.2E}' for i in a[-2:]])

        self._fig, ax = plt.subplots(figsize=(10, 5))
        plt.subplots_adjust(left=0.08, right=0.9)

        line, = ax.plot(self.spec, linewidth=linewidth, alpha=alpha)

        scatter = ax.scatter(wave_exp, flx_exp, s=s,
            c=weights_explanation, cmap='plasma',
            vmin=self.vmin, vmax=self.vmax, alpha=1.)

        ax_cb = self._colorbar_explanation()
        ax.set_title(
        f'{self.sdss_name}: {metric}, {feature_selection}, k_width={kernel_width}')

        # plt.tight_layout()

        return self._fig, ax, ax_cb, line, scatter

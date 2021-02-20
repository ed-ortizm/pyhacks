from functools import partial
import glob
from itertools import product
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

import lime
# import lime.lime_tabular
from lime import lime_tabular
from lime import lime_image
import multiprocessing as mp
import pickle
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from tensorflow.keras.models import load_model
###############################################################################
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
###############################################################################
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
def get_explanation_data(explanation_dictionary, key, spec):

    exp_array = explanation_dictionary[f'{key}'][1]
    wave_exp = exp_array[:, 0].astype(np.int)
    flx_exp = spec[wave_exp]
    weights_exp = exp_array[:, 1]

    k_width = float(explanation_dictionary[f'{key}'][0])
    k_width = float(f'{k_width/1.:.2f}')
    feature_selection = explanation_dictionary[f'{key}'][4]
    metric = explanation_dictionary[f'{key}'][3]

    return wave_exp, flx_exp, weights_exp, k_width, feature_selection, metric


def get_exp_dict(
    explanation_file=
        '/home/edgar/zorro/AEsII/xAI/lime/results/spec-1246-54478-0144_exp_dict.dill'):

     sdss_directory = "/home/edgar/zorro/SDSSdata/data_proc"
     sdss_name = explanation_file.split('/')[-1].split('_')[0]
     with open(f'{explanation_file}', 'rb') as file:
         exp_dict = pickle.load(file)

     spec = np.load(f'{sdss_directory}/{sdss_name}.npy')

     return spec, exp_dict, sdss_name

def plot_explanation(sdss_name, k_width, feature_selection, metric, spec, wave_exp, flx_exp, weights_exp, s=10, linewidth=1., alpha=1., cmap='plasma_r'):

    c = weights_exp/np.max(weights_exp)

    plt.plot(spec, linewidth=linewidth, alpha=alpha)
    plt.scatter(wave_exp, flx_exp, s=s, c=c, cmap=cmap)
    plt.title(f'{sdss_name}: {metric}, {feature_selection}, k_width={k_width}')

    plt.colorbar()

class Explainer_parallel:

    def __init__(self, explainer_type, training_data, training_labels,
        feature_names, kernel_widths, features_selection,
        sample_around_instance, training_data_stats=None,
        discretize_continuous=False, discretizer='decile', verbose=False,
        mode='regression', n_processes=None):
        # The input variable are lists

        self.k_widths = kernel_widths
        self.ftrs_slect = features_selection
        self.around_instance = sample_around_instance

        if n_processes == None:
            self.n_processes = mp.cpu_count()-1 or 1
        else:
            self.n_processes = n_processes


        # Fixed values

        self.xpl_type = explainer_type
        self.t_dat = training_data
        self.t_lbls = training_labels
        self.ftr_names = feature_names
        self.t_dat_stats = training_data_stats
        self.discretize = discretize_continuous
        self.discretizer = discretizer
        self.verbose = verbose
        self.mode = mode

        self.Ex_partial = partial(Explainer, explainer_type=self.xpl_type,
        training_data=self.t_dat, training_labels=self.t_lbls,
        feature_names=self.ftr_names, training_data_stats=self.t_dat_stats,
        discretize_continuous=self.discretize, discretizer=self.discretizer,
        verbose=self.verbose, mode=self.mode)

        self.explanations = None

    def get_explanations(self, x, regressor, sdss_name):

        params_grid = product([x], [regressor], [sdss_name],
            self.k_widths, self.ftrs_slect, self.around_instance)

        with mp.Pool(processes=self.n_processes) as pool:
            self.explanations = pool.starmap(self._get_explanation, params_grid)
            self._sizeof(self.explanations, itr_name='explanations')

        return self.explanations

    def _get_explanation(self, x, regressor, sdss_name,
        kernel_width, feature_selection, sample_around_instance):

        explainer = self.Ex_partial(kernel_width, feature_selection,
            sample_around_instance)

        self._sizeof(explainer, itr_name='explainer', is_itr=False)

        return [sdss_name, kernel_width, feature_selection, sample_around_instance,
            explainer.explanation(x, regressor)]

    def _sizeof(self, iterable, itr_name="iterable", is_itr=True):

        if is_itr:
            size = 0
            for itr in iterable:
                x = sys.getsizeof(itr)*1e-6
                print(f'The size of object from {itr_name} is: {x:.2f} Mbs')
                size += x
            print(f"The total size of {itr_name} is {size:.2f} Mbs")
        else:
            size =  sys.getsizeof(iterable)
            print(f"The total size of {itr_name} is {size:.2f} Mbs")

class Explainer:
    def __init__(self, kernel_width, feature_selection,
        sample_around_instance, explainer_type, training_data,
        training_labels, feature_names, training_data_stats=None,
        discretize_continuous=False, discretizer='decile', verbose=True,
        mode='regression'):

        self.xpl_type = explainer_type
        self.tr_data = training_data
        self.tr_labels = training_labels
        self.ftr_names = feature_names
        self.k_width = kernel_width
        self.ftr_select = feature_selection
        self.tr_data_stats = training_data_stats
        self.sar_instance = sample_around_instance
        self.discretize = discretize_continuous
        self.discretizer = discretizer
        self.verbose = verbose
        self.mode = mode

        if self.xpl_type == "tabular":

            self.explainer = self._tabular_explainer()
            x = sys.getsizeof(self.explainer)*1e-6
            print(f'The size of the explainer is: {x:.2f} Mbs')

    def _tabular_explainer(self):

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.tr_data, training_labels=self.tr_labels,
            feature_names=self.ftr_names, kernel_width=self.k_width,
            feature_selection=self.ftr_select,
            training_data_stats=self.tr_data_stats,
            sample_around_instance=self.sar_instance,
            discretize_continuous=self.discretize, discretizer=self.discretizer,
            verbose = self.verbose, mode=self.mode)

        return explainer

    def explanation(self, x, regressor):


        xpl = self.explainer.explain_instance(x, regressor,
            num_features=x.shape[0])
        return xpl.as_list()

class Explanation:

    def __init__(self, discretize_continuous=False):
        self.discretize_continuous = discretize_continuous

    def explanations_from_file(self, explanation_file_path: str, save=True):

        if not os.path.exists(explanation_file_path):

            print(f'There is no file {explanation_file_path}')
            return None

        sdss_name = explanation_file_path.split("/")[-1].split("_")[0]
        metric = explanation_file_path.split("/")[-1].split("_")[1].strip(".exp")
        explanation_dict = {}

        with open(explanation_file_path, "r") as file:

            explanation_lines = file.readlines()

            for idx, explanation_line in enumerate(explanation_lines):

                explanation_line = self._line_curation(explanation_line)

                k_width = explanation_line[1] # string
                feature_selection = explanation_line[2]
                sample_around_instance = explanation_line[3]

                explanation_array = self._fluxes_weights(
                    line=explanation_line[4:])

                explanation_dict[f'{idx}'] = [k_width, explanation_array,
                    sdss_name, metric,
                    f'{feature_selection}_{sample_around_instance}']

        return explanation_dict

    def _fluxes_weights(self, line):

        length = np.int(len(line)/2)
        fluxes_weights = np.empty((length,2))

        for idx, fw in enumerate(fluxes_weights):
            fw[0] = np.float(line[2*idx].strip("'flux "))
            fw[1] = np.float(line[2*idx+1])

        return fluxes_weights

    def _line_curation(self, line):
        for charater in "()[]'":
            line = line.replace(charater, "")
        return [element.strip(" \n") for element in line.split(",")]

    def plot_explanation(self, spec, wave_exp, flx_exp, weights_exp, s=10., linewidth=0.2, cmap='plasma_r', show=False, ipython=False):

        c = weights_exp/np.max(weights_exp)

        fig, ax = plt.subplots(figsize=(15, 5))

        ax.plot(spec, linewidth=linewidth)
        ax.scatter(wave_exp, flx_exp, s=s, c=c, cmap=cmap)

        fig.colorbar()

        fig.savefig(f'testing/test.png')
        fig.savefig(f'testing/test.pdf')
        if show:
            plt.show()
        if not ipython:
            plt.close()

class Outlier:
    """
    Class for dealing with the outliers based on a generative model trained with
    tensorflow.keras
    """

    def __init__(self, model_path, o_scores_path='.', metric='mse', p='p',
        n_spec=30, custom=False, custom_metric=None):
        """
        Init fucntion

        Args:
            model_path: path where the trained generative model is located

            o_scores_path: (str) path where the numpy arrays with the outlier scores
                is located. Its functionality is for the cases where the class
                will be implemented several times, offering therefore the
                possibility to load the scores instead of computing them over
                and over.

            metric: (str) the name of the metric used to compute the outlier score
                using the observed spectrum and its reconstruction. Possible

            p: (float > 0) in case the metric is the lp metric, p needs to be a non null
                possitive float [Aggarwal 2001]

            n_spec: (int > 0) this parameter contros the number of objects identifiers to
                return for the top reconstruction, that is the most oulying and
                the most normal objects
        """

        self.model_path = model_path
        self.o_scores_path = o_scores_path
        self.metric = metric
        self.p = p
        self.n_spec = n_spec
        self.custom = custom
        if self.custom:
            self.custom_metric = custom_metric

    def _get_OR(self, O, model):

        if len(O.shape) == 1:
            O = O.reshape(1,-1)

        R = model.predict(O)

        return O, R

    def score(self, O):
        print("Score working in parallelization")
        """
        Computes the outlier score according to the metric used to define
        instantiate the class.

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

        Returns:
            A one dimensional numpy array with the outlier scores for objects
            present in O
        """

        model_name = self.model_path.split('/')[-1]
        print(f'Loading model: {model_name}')
        model = load_model(f'{self.model_path}')

        O, R = self._get_OR(O, model)

        if self.custom:
            print(f'Computing the predictions of {model_name}')
            return self.user_metric(O=O, R=R)

        elif self.metric == 'mse':
            print(f'Computing the predictions of {model_name}')
            return self._mse(O=O, R=R)

        elif self.metric == 'chi2':
            print(f'Computing the predictions of {model_name}')
            return self._chi2(O=O, R=R)

        elif self.metric == 'mad':
            print(f'Computing the predictions of {model_name}')
            return self._mad(O=O, R=R)

        elif self.metric == 'lp':

            if self.p == 'p' or self.p <= 0:
                print(f'For the {self.metric} metric you need p')
                return None

            print(f'Computing the predictions of {model_name}')
            return self._lp(O=O, R=R)

        else:
            print(f'The provided metric: {self.metric} is not implemented yet')
            return None

    def _coscine_similarity(self, O, R):
        """
        Computes the coscine similarity between the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the cosine similarity between
            objects O and their reconstructiob
        """

        pass

    def _jaccard_index(self, O, R):
        """
        Computes the mean square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the mean square error for objects
            present in O
        """

        pass

    def _sorensen_dice_index(self, O, R):
        """
        Computes the mean square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the mean square error for objects
            present in O
        """
        pass
# Mahalanobis, Canberra, Braycurtis, and KL-divergence
    def _mse(self, O, R):
        """
        Computes the mean square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the mean square error for objects
            present in O
        """

        return np.square(R-O).mean(axis=1)

    def _chi2(self, O, R):
        """
        Computes the chi square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the chi square error for objects
            present in O
        """

        return (np.square(R-O)*(1/np.abs(R))).mean(axis=1)

    def _mad(self, O, R):
        """
        Computes the maximum absolute deviation from the reconstruction of the
        input objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the maximum absolute deviation
            from the objects present in O
        """

        return np.abs(R-O).mean(axis=1)

    def _lp(self, O, R):
        """
        Computes the lp distance from the reconstruction of the input objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the lp distance from the objects
            present in O
        """

        return (np.sum((np.abs(R-O))**self.p, axis=1))**(1/self.p)
# gotta code conditionals to make sure that the user inputs a "good one"
    def user_metric(self, custom_metric, O, R):
        """
        Computes the custom metric for the reconstruction of the input objects
        as defined by the user

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the score produced by the user
            defiend metric of objects present in O
        """

        return self.custom_metric(O, R)

    def metadata(self, spec_idx, training_data_files):
        """
        Generates the names and paths of the individual objects used to create
        the training data set.
        Note: this work according to the way the training data set was created

        Args:
            spec_idx: (int > 0) the location index of the spectrum in the
                training data set.

            training_data_files: (list of strs) a list with the paths of the
                individual objects used to create the training data set.

        Returns:
            sdss_name, sdss_name_path: (str, str) the sdss name of the objec,
                the path of the object in the files system
        """


        # print('Gathering name of data points used for training')

        sdss_names = [name.split('/')[-1].split('.')[0] for name in
            training_data_files]

        # print('Retrieving the sdss name of the desired spectrum')

        sdss_name = sdss_names[spec_idx]
        sdss_name_path = training_data_files[spec_idx]

        return sdss_name, sdss_name_path

    def top_reconstructions(self, O):
        """
        Selects the most normal and outlying objecs

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

        Returns:
            most_normal, most_oulying: (1D np.array, 1D np.array) numpy arrays
                with the location indexes of the most normal and outlying
                object in the training set.
        """

        if os.path.exists(f"{self.o_scores_path}/{self.metric}_o_score.npy"):
            scores= np.load(f"{self.o_scores_path}/{self.metric}_o_score.npy")
        else:
            scores = self.score(O)

        spec_idxs = np.argpartition(scores,
        [self.n_spec, -1*self.n_spec])

        most_normal = spec_idxs[: self.n_spec]
        most_oulying = spec_idxs[-1*self.n_spec:]

        return most_normal, most_oulying
################################################################################
class ImageExplainer:
    def __init__(self):
        pass
###############################################################################
## Case for spectra
class Spec_segmenter:

    def __init__(self, spec, segmenter="slic"):

        if len(spec.shape) == 1:
            self.spec = spec.reshape(1,-1)
        else:
            self.spec = spec

        self.segmenter = segmenter

    def slic(self, n_segments=30, compactness=100, sigma=1):

        segments = slic(self.spec, n_segments=n_segments,
            compactness=compactness, sigma=sigma, multichannel=False)

        boundaries = mark_boundaries(self.spec, segments)

        diff = boundaries[0, :, 0] - self.spec[0, :]

        idxs = np.nonzero(diff)[0]

        return idxs


    def plot(self, idxs, show=False):

        fig, ax = plt.subplots(figsize=(15, 5))

        wave = np.arange(self.spec.shape[1])

        for n, idx in enumerate(idxs):

            if n == 0:
                ax.plot(wave[:idxs[n]], self.spec[0, :idxs[n]], linewidth=0.7)
            else:
                ax.plot(wave[idxs[n-1]: idxs[n]],
                    self.spec[0, idxs[n-1]: idxs[n]], linewidth=0.7)

        if show:
            plt.show()

    ###########################################################################
    # print(f"Felzenszwalb's efficient graph based segmentation")
    # segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    # print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
    #
    # print(f'Quickshift image segmentation')
    # segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    # print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")
    #
    # print(f'Compact watershed segmentation of gradient images')
    # gradient = sobel(rgb2gray(img))
    # segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    # print(f"Watershed number of segments: {len(np.unique(segments_watershed))}")
    """
    ====================================================
    Comparison of segmentation and superpixel algorithms
    ====================================================

    This example compares four popular low-level image segmentation methods.  As
    it is difficult to obtain good segmentations, and the definition of "good"
    often depends on the application, these methods are usually used for obtaining
    an oversegmentation, also known as superpixels. These superpixels then serve as
    a basis for more sophisticated algorithms such as conditional random fields
    (CRF).


    Felzenszwalb's efficient graph based segmentation
    -------------------------------------------------
    This fast 2D image segmentation algorithm, proposed in [1]_ is popular in the
    computer vision community.
    The algorithm has a single ``scale`` parameter that influences the segment
    size. The actual size and number of segments can vary greatly, depending on
    local contrast.

    .. [1] Efficient graph-based image segmentation, Felzenszwalb, P.F. and
           Huttenlocher, D.P.  International Journal of Computer Vision, 2004


    Quickshift image segmentation
    -----------------------------

    Quickshift is a relatively recent 2D image segmentation algorithm, based on an
    approximation of kernelized mean-shift. Therefore it belongs to the family of
    local mode-seeking algorithms and is applied to the 5D space consisting of
    color information and image location [2]_.

    One of the benefits of quickshift is that it actually computes a
    hierarchical segmentation on multiple scales simultaneously.

    Quickshift has two main parameters: ``sigma`` controls the scale of the local
    density approximation, ``max_dist`` selects a level in the hierarchical
    segmentation that is produced. There is also a trade-off between distance in
    color-space and distance in image-space, given by ``ratio``.

    .. [2] Quick shift and kernel methods for mode seeking,
           Vedaldi, A. and Soatto, S.
           European Conference on Computer Vision, 2008


    SLIC - K-Means based image segmentation
    ---------------------------------------

    This algorithm simply performs K-means in the 5d space of color information and
    image location and is therefore closely related to quickshift. As the
    clustering method is simpler, it is very efficient. It is essential for this
    algorithm to work in Lab color space to obtain good results.  The algorithm
    quickly gained momentum and is now widely used. See [3]_ for details.  The
    ``compactness`` parameter trades off color-similarity and proximity, as in the
    case of Quickshift, while ``n_segments`` chooses the number of centers for
    kmeans.

    .. [3] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Suesstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.


    Compact watershed segmentation of gradient images
    -------------------------------------------------

    Instead of taking a color image as input, watershed requires a grayscale
    *gradient* image, where bright pixels denote a boundary between regions.
    The algorithm views the image as a landscape, with bright pixels forming high
    peaks. This landscape is then flooded from the given *markers*, until separate
    flood basins meet at the peaks. Each distinct basin then forms a different
    image segment. [4]_

    As with SLIC, there is an additional *compactness* argument that makes it
    harder for markers to flood faraway pixels. This makes the watershed regions
    more regularly shaped. [5]_

    .. [4] https://en.wikipedia.org/wiki/Watershed_%28image_processing%29

    .. [5] Peer Neubert & Peter Protzel (2014). Compact Watershed and
           Preemptive SLIC: On Improving Trade-offs of Superpixel Segmentation
           Algorithms. ICPR 2014, pp 996-1001. :DOI:`10.1109/ICPR.2014.181`
           https://www.tu-chemnitz.de/etit/proaut/publications/cws_pSLIC_ICPR.pdf
    """
def helper_idea():
    #  Explore this idea: like the setters and getters for the custom fucntion ;)
    # class Helper(object):
    #
    #     def add(self, a, b):
    #         return a + b
    #
    #     def mul(self, a, b):
    #         return a * b
    #
    #
    # class MyClass(Helper):
    #
    #     def __init__(self):
    #         Helper.__init__(self)
    #         print self.add(1, 1)
    #
    #
    # if __name__ == '__main__':
    #     obj = MyClass()
    pass

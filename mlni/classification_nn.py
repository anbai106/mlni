from mlni.base import WorkFlow, ClassificationAlgorithm, ClassificationValidation
import numpy as np
from sklearn.model_selection import ShuffleSplit
from mlni.utils import time_bar, neural_network_classification_3LinerLayers, neural_network_classification_5LinerLayers, train_network_classification
import torch
import copy

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020"
__credits__ = ["Junhao Wen, Jorge Samper-González"]
__license__ = "See LICENSE file"
__version__ = "0.1.5.1"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

class RB_RepeatedHoldOut_NN_Classification(WorkFlow):
    """
    The main class to run MLNI with repeated holdout CV for Classification.
    """

    def __init__(self, input, split_index, output_dir, n_threads=8, n_iterations=100, test_size=0.2,
                 batch_size=64, epochs=10, lr=0.0001, weight_decay=1e-4, optimizer='Adam', gpu=True, verbose=False):

        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._batch_size = batch_size
        self._epochs = epochs
        self._lr = lr
        self._weight_decay = weight_decay
        self._optimizer = optimizer
        self._gpu = gpu
        self._verbose = verbose
        self._test_size = test_size
        self._validation = None
        self._algorithm = None

    def run(self):
        x = self._input.get_x()
        y = self._input.get_y()
        df_header = self._input.get_participant_session_id()
        input_dim = x.shape[1]

        ### convert numpy array to Torch Tensor
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.LongTensor(y)

        self._algorithm = NNClassificationAlgorithm(x_tensor, y_tensor, self._output_dir, df_header, self._n_iterations,
                                                     batch_size=self._batch_size,
                                                     epochs=self._epochs,
                                                     lr=self._lr,
                                                     weight_decay=self._weight_decay, optimizer=self._optimizer,
                                                     gpu=self._gpu,
                                                     verbose=self._verbose)

        self._validation = RepeatedHoldOut(self._algorithm, input_dim, n_iterations=self._n_iterations, test_size=self._test_size)

        self._validation.validate(y, splits_indices=self._split_index, verbose=self._verbose)

class NNClassificationAlgorithm(ClassificationAlgorithm):
    '''
    NN Classification.
    '''
    def __init__(self, x, y, output_dir, df_header, n_iterations=100, batch_size=64, epochs=10, lr=0.0001, weight_decay=1e-4, optimizer='Adam', gpu=True,
                 verbose=False):
        self._x = x
        self._y = y
        self._output_dir = output_dir
        self._df_header = df_header
        self._n_iterations = n_iterations
        self._batch_size = batch_size
        self._epochs = epochs
        self._lr = lr
        self._weight_decay = weight_decay
        self._optimizer = optimizer
        self._gpu = gpu
        self._verbose = verbose

    def _lauch_nn(self, input_dim, x_train, x_test, y_train, y_test, fi):
        # model = neural_network_classification_3LinerLayers(input_dim)
        model = neural_network_classification_5LinerLayers(input_dim)
        if self._gpu:
            model.cuda()
        else:
            model.cpu()
        init_state = copy.deepcopy(model.state_dict())

        # train the NN
        train_network_classification(model, self._output_dir, fi, x_train, y_train, x_test, y_test, self._epochs,
                      self._batch_size, init_state, self._df_header, self._gpu, self._lr,  self._weight_decay, self._optimizer)

    def evaluate(self, input_dim, train_index, test_index, fi):
        x_train = self._x[train_index]
        y_train = self._y[train_index]
        x_test = self._x[test_index]
        y_test = self._y[test_index]

        results = self._lauch_nn(input_dim, x_train, x_test, y_train, y_test, fi)

        return results

class RepeatedHoldOut(ClassificationValidation):
    """
    Repeated holdout splits CV.
    """

    def __init__(self, ml_algorithm, input_dim, n_iterations=100, test_size=0.3):
        self._ml_algorithm = ml_algorithm
        self._input_dim = input_dim
        self._split_results = []
        self._cv = None
        self._n_iterations = n_iterations
        self._test_size = test_size

    def validate(self, y, splits_indices=None, verbose=False):

        if splits_indices is None:
            splits = ShuffleSplit(n_splits=self._n_iterations, test_size=self._test_size)
            self._cv = list(splits.split(np.zeros(len(y)), y))
        else:
            self._cv = splits_indices
        results = {}

        for i in range(self._n_iterations):
            time_bar(i, self._n_iterations)
            print()
            if verbose:
                print("Repetition %d of CV..." % i)
            train_index, test_index = self._cv[i]
            results[i] = self._ml_algorithm.evaluate(self._input_dim, train_index, test_index, i)
from pyhydra.base import WorkFlow, ClassificationAlgorithm, ClassificationValidation
import numpy as np
import pandas as pd
import os, json
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from multiprocessing.pool import ThreadPool
from pyhydra.utils import evaluate_prediction, gram_matrix_linear
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, RFE, SelectPercentile, SelectFromModel

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen, Jorge Samper-González"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

class RB_RepeatedHoldOut_DualSVM_Classification(WorkFlow):
    """
    The main class to run pyhydra with repeated holdout CV for classification.
    """

    def __init__(self, input, split_index, output_dir, n_threads=8, n_iterations=100, test_size=0.2,
                 grid_search_folds=10, balanced=True, c_range=np.logspace(-6, 2, 17), kernel=None, verbose=False):
        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range
        self._verbose = verbose
        self._test_size = test_size
        self._validation = None
        self._algorithm = None
        self._kernel = kernel

    def run(self):
        if self._kernel is None:
            kernel = self._input.get_kernel()
        else:
            kernel = self._kernel
        x = self._input.get_x()
        y = self._input.get_y()

        if self._verbose:
            if y[0] == 0:
                print('For classification, the negative coefficients in the weight map are more likely to be classified as the first label in the diagnose tsv')
            else:
                print('For classification, the positive coefficients in the weight map are more likely to be classified as the second label in the diagnose tsv')

        self._algorithm = LinearSVMAlgorithmWithPrecomputedKernel(kernel,
                                                     y,
                                                     balanced=self._balanced,
                                                     grid_search_folds=self._grid_search_folds,
                                                     c_range=self._c_range,
                                                     n_threads=self._n_threads, verbose=self._verbose)

        self._validation = RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations, test_size=self._test_size)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._split_index, verbose=self._verbose)
        classifier_dir = os.path.join(self._output_dir, 'classifier')
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        self._algorithm.save_weights(classifier, x, classifier_dir)
        self._validation.save_results(self._output_dir)

class RB_RepeatedHoldOut_DualSVM_Classification_Nested_Feature_Selection(WorkFlow):
    """
    The main class to run pyhydra with stritified KFold CV for classification with roi features and nested feature selection.
    """

    def __init__(self, input, split_index, output_dir, n_threads=8, n_iterations=100, test_size=0.2, grid_search_folds=10,
                 balanced=True, c_range=np.logspace(-6, 2, 17), feature_selection_method='RFE', top_k=50, verbose=False):
        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range
        self._verbose = verbose
        self._test_size = test_size
        self._validation = None
        self._algorithm = None
        self._feature_selection_method = feature_selection_method
        self._top_k = top_k

    def run(self):
        x = self._input.get_x()
        y = self._input.get_y()

        if self._verbose:
            if y[0] == 0:
                print('For classification, the negative coefficients in the weight map are more likely to be classified as the first label in the diagnose tsv')
            else:
                print('For classification, the positive coefficients in the weight map are more likely to be classified as the second label in the diagnose tsv')

        self._algorithm = LinearSVMAlgorithmWithPrecomputedKernelNestedFeatureSelection(x, y, self._feature_selection_method,
                     self._top_k, balanced=self._balanced, grid_search_folds=self._grid_search_folds, c_range=self._c_range,
                                                                        n_threads=self._n_threads, verbose=self._verbose)

        self._validation = RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations, test_size=self._test_size)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._split_index, verbose=self._verbose)
        classifier_dir = os.path.join(self._output_dir, 'classifier')
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        self._algorithm.save_weights(classifier, x, classifier_dir)
        self._validation.save_results(self._output_dir)

class VB_RepeatedHoldOut_DualSVM_Classification(WorkFlow):
    """
    The main class to run pyhydra with repeated holdout CV for classification with voxel-wise features.
    """
    def __init__(self, input, split_index, output_dir,  n_threads=8, n_iterations=100, test_size=0.2,
                 grid_search_folds=10, balanced=True, c_range=np.logspace(-6, 2, 17), verbose=False):

        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range
        self._verbose = verbose
        self._test_size = test_size
        self._validation = None
        self._algorithm = None

    def run(self):

        x = self._input.get_x()
        y = self._input.get_y()
        kernel = self._input.get_kernel()

        ## for voxel approach, we use kernel==precomputed to boost the algorithm
        self._algorithm = LinearSVMAlgorithmWithPrecomputedKernel(kernel,
                                                     y,
                                                     balanced=self._balanced,
                                                     grid_search_folds=self._grid_search_folds,
                                                     c_range=self._c_range,
                                                     n_threads=self._n_threads, verbose=self._verbose)

        self._validation = RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations, test_size=self._test_size)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._split_index, verbose=self._verbose)
        classifier_dir = os.path.join(self._output_dir, 'classifier')
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        weights = self._algorithm.save_weights(classifier, x, classifier_dir)
        self._validation.save_results(self._output_dir)
        self._input.save_weights_as_nifti(weights, classifier_dir)

class VB_RepeatedHoldOut_DualSVM_Classification_Nested_Feature_Selection(WorkFlow):
    """
    The main class to run pyhydra with repeated holdout CV for classification with voxel-wise features and nested feature selection.
    """
    def __init__(self, input, split_index, output_dir,  n_threads=8, n_iterations=100, test_size=0.2, grid_search_folds=10,
                 balanced=True, c_range=np.logspace(-6, 2, 17), feature_selection_method='RFE', top_k=50, verbose=False ):

        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range
        self._verbose = verbose
        self._test_size = test_size
        self._validation = None
        self._algorithm = None
        self._feature_selection_method = feature_selection_method
        self._top_k = top_k

    def run(self):

        x = self._input.get_x()
        y = self._input.get_y()

        ## for voxel approach, we use kernel==precomputed to boost the algorithm
        self._algorithm = LinearSVMAlgorithmWithPrecomputedKernelNestedFeatureSelection(x, y, self._feature_selection_method, self._top_k,
                          balanced=self._balanced, grid_search_folds=self._grid_search_folds, c_range=self._c_range,
                                                                  n_threads=self._n_threads, verbose=self._verbose)

        self._validation = RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations, test_size=self._test_size)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads, splits_indices=self._split_index,
                                                                     verbose=self._verbose)
        classifier_dir = os.path.join(self._output_dir, 'classifier')
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        weights = self._algorithm.save_weights(classifier, x, classifier_dir)
        self._validation.save_results(self._output_dir)

class RB_KFold_DualSVM_Classification(WorkFlow):
    """
    The main class to run pyhydra with stritified KFold CV for classification with ROI features.
    """

    def __init__(self, input, split_index, output_dir, n_folds, n_threads=8, grid_search_folds=10, balanced=True,
                 c_range=np.logspace(-6, 2, 17), kernel=None, verbose=False):
        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range
        self._verbose = verbose
        self._n_folds = n_folds
        self._validation = None
        self._algorithm = None
        self._kernel = kernel

    def run(self):
        if self._kernel is None:
            kernel = self._input.get_kernel()
        else:
            kernel = self._kernel
        x = self._input.get_x()
        y = self._input.get_y()
        if self._verbose:
            if y[0] == 0:
                print('For classification, the negative coefficients in the weight map are more likely to be classified as the first label in the diagnose tsv')
            else:
                print('For classification, the positive coefficients in the weight map are more likely to be classified as the second label in the diagnose tsv')

        self._algorithm = LinearSVMAlgorithmWithPrecomputedKernel(kernel, y, balanced=self._balanced,
                                                     grid_search_folds=self._grid_search_folds, c_range=self._c_range,
                                                     n_threads=self._n_threads, verbose=self._verbose)

        self._validation = KFoldCV(self._algorithm)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._split_index,
                                                                     n_folds=self._n_folds, verbose=self._verbose)
        classifier_dir = os.path.join(self._output_dir, 'classifier')
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        self._algorithm.save_weights(classifier, x, classifier_dir)
        self._validation.save_results(self._output_dir)

class VB_KFold_DualSVM_Classification(WorkFlow):
    """
    The main class to run pyhydra with stritified KFold CV for classification with voxel features.
    """

    def __init__(self, input, split_index, output_dir, n_folds, n_threads=8, grid_search_folds=10, balanced=True,
                 c_range=np.logspace(-6, 2, 17), verbose=False):
        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range
        self._verbose = verbose
        self._n_folds = n_folds
        self._validation = None
        self._algorithm = None

    def run(self):
        x = self._input.get_x()
        y = self._input.get_y()
        kernel = self._input.get_kernel()
        if self._verbose:
            if y[0] == 0:
                print('For classification, the negative coefficients in the weight map are more likely to be classified as the first label in the diagnose tsv')
            else:
                print('For classification, the positive coefficients in the weight map are more likely to be classified as the second label in the diagnose tsv')

        self._algorithm = LinearSVMAlgorithmWithPrecomputedKernel(kernel, y, balanced=self._balanced,
                                                     grid_search_folds=self._grid_search_folds, c_range=self._c_range,
                                                     n_threads=self._n_threads, verbose=self._verbose)

        self._validation = KFoldCV(self._algorithm)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._split_index,
                                                                     n_folds=self._n_folds, verbose=self._verbose)
        classifier_dir = os.path.join(self._output_dir, 'classifier')
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        weights = self._algorithm.save_weights(classifier, x, classifier_dir)
        self._validation.save_results(self._output_dir)
        self._input.save_weights_as_nifti(weights, classifier_dir)

class LinearSVMAlgorithmWithPrecomputedKernel(ClassificationAlgorithm):
    '''
    Dual SVM with precomputed linear kernel SVM.
    '''
    def __init__(self, kernel, y, balanced=True, grid_search_folds=10, c_range=np.logspace(-6, 2, 17), n_threads=15,
                 verbose=False):
        self._kernel = kernel
        self._y = y
        self._balanced = balanced
        self._grid_search_folds = grid_search_folds
        self._c_range = c_range
        self._n_threads = n_threads
        self._verbose = verbose

    def _launch_svc(self, kernel_train, x_test, y_train, y_test, c):

        if self._balanced:
            svc = SVC(C=c, kernel='precomputed', probability=True, tol=1e-6, class_weight='balanced')
        else:
            svc = SVC(C=c, kernel='precomputed', probability=True, tol=1e-6)

        svc.fit(kernel_train, y_train)
        y_hat_train = svc.predict(kernel_train)
        y_hat = svc.predict(x_test)
        proba_test_index1 = svc.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, proba_test_index1)

        return svc, y_hat, auc, y_hat_train, proba_test_index1

    def _grid_search(self, kernel_train, x_test, y_train, y_test, c):

        _, y_hat, _, _, _ = self._launch_svc(kernel_train, x_test, y_train, y_test, c)
        ba = evaluate_prediction(y_test, y_hat)['balanced_accuracy']

        return ba

    def _select_best_parameter(self, async_result):

        c_values = []
        accuracies = []
        for fold in async_result.keys():
            best_c = -1
            best_acc = -1

            for c, async_acc in async_result[fold].items():

                acc = async_acc.get()
                if acc > best_acc:
                    best_c = c
                    best_acc = acc
            c_values.append(best_c)
            accuracies.append(best_acc)

        best_acc = np.mean(accuracies)
        best_c = np.power(10, np.mean(np.log10(c_values)))

        return {'c': best_c, 'balanced_accuracy': best_acc}

    def evaluate(self, train_index, test_index):

        inner_pool = ThreadPool(self._n_threads)
        async_result = {}
        for i in range(self._grid_search_folds):
            async_result[i] = {}

        outer_kernel = self._kernel[train_index, :][:, train_index]
        y_train = self._y[train_index]

        skf = StratifiedKFold(n_splits=self._grid_search_folds, shuffle=True)
        inner_cv = list(skf.split(np.zeros(len(y_train)), y_train))

        for i in range(len(inner_cv)):
            inner_train_index, inner_test_index = inner_cv[i]

            inner_kernel = outer_kernel[inner_train_index, :][:, inner_train_index]
            x_test_inner = outer_kernel[inner_test_index, :][:, inner_train_index]
            y_train_inner, y_test_inner = y_train[inner_train_index], y_train[inner_test_index]

            for c in self._c_range:
                if self._verbose:
                    print("Inner CV for C=%f..." % c)
                async_result[i][c] = inner_pool.apply_async(self._grid_search, args=(inner_kernel, x_test_inner, y_train_inner,
                                                                          y_test_inner, c))
        inner_pool.close()
        inner_pool.join()

        best_parameter = self._select_best_parameter(async_result)
        x_test = self._kernel[test_index, :][:, train_index]
        y_train, y_test = self._y[train_index], self._y[test_index]

        _, y_hat, auc, y_hat_train, proba_test_index1 = self._launch_svc(outer_kernel, x_test, y_train, y_test, best_parameter['c'])

        result = dict()
        result['best_parameter'] = best_parameter
        result['evaluation'] = evaluate_prediction(y_test, y_hat)
        result['evaluation_train'] = evaluate_prediction(y_train, y_hat_train)
        result['y_hat'] = y_hat
        result['y_hat_train'] = y_hat_train
        result['y'] = y_test
        result['y_train'] = y_train
        result['y_index'] = test_index
        result['x_index'] = train_index
        result['auc'] = auc
        result['proba_test_index1'] = proba_test_index1

        return result

    def apply_best_parameters(self, results_list):

        best_c_list = []
        bal_acc_list = []

        for result in results_list:
            best_c_list.append(result['best_parameter']['c'])
            bal_acc_list.append(result['best_parameter']['balanced_accuracy'])

        # 10^(mean of log10 of best Cs of each fold) is selected
        best_c = np.power(10, np.mean(np.log10(best_c_list)))
        # Mean balanced accuracy
        mean_bal_acc = np.mean(bal_acc_list)

        if self._balanced:
            svc = SVC(C=best_c, kernel='precomputed', probability=True, tol=1e-6, class_weight='balanced')
        else:
            svc = SVC(C=best_c, kernel='precomputed', probability=True, tol=1e-6)

        svc.fit(self._kernel, self._y)

        return svc, {'c': best_c, 'balanced_accuracy': mean_bal_acc}

    def save_classifier(self, classifier, output_dir):

        np.savetxt(os.path.join(output_dir, 'dual_coefficients.txt'), classifier.dual_coef_)
        np.savetxt(os.path.join(output_dir, 'support_vectors_indices.txt'), classifier.support_)
        np.savetxt(os.path.join(output_dir, 'intersect.txt'), classifier.intercept_)

    def save_weights(self, classifier, x, output_dir):

        dual_coefficients = classifier.dual_coef_
        sv_indices = classifier.support_

        weighted_sv = dual_coefficients.transpose() * x[sv_indices]
        weights = np.sum(weighted_sv, 0)

        np.savetxt(os.path.join(output_dir, 'weights.txt'), weights)

        return weights

    def save_parameters(self, parameters_dict, output_dir):
        with open(os.path.join(output_dir, 'best_parameters.json'), 'w') as f:
            json.dump(parameters_dict, f)

class LinearSVMAlgorithmWithPrecomputedKernelNestedFeatureSelection(ClassificationAlgorithm):
    '''
    Dual SVM with precomputed linear kernel for nested feature selection.
    '''
    def __init__(self, x, y, feature_selection_method, top_k, balanced=True, grid_search_folds=10, c_range=np.logspace(-6, 2, 17), n_threads=15,
                 verbose=False):
        self._x = x
        self._y = y
        self._feature_selection_method = feature_selection_method
        self._top_k = top_k
        self._balanced = balanced
        self._grid_search_folds = grid_search_folds
        self._c_range = c_range
        self._n_threads = n_threads
        self._verbose = verbose

    def _launch_svc(self, kernel_train, x_test, y_train, y_test, c):

        if self._balanced:
            svc = SVC(C=c, kernel='precomputed', probability=True, tol=1e-6, class_weight='balanced')
        else:
            svc = SVC(C=c, kernel='precomputed', probability=True, tol=1e-6)

        svc.fit(kernel_train, y_train)
        y_hat_train = svc.predict(kernel_train)
        y_hat = svc.predict(x_test)
        proba_test_index1 = svc.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, proba_test_index1)

        return svc, y_hat, auc, y_hat_train, proba_test_index1

    def _grid_search(self, kernel_train, x_test, y_train, y_test, c):

        _, y_hat, _, _, _ = self._launch_svc(kernel_train, x_test, y_train, y_test, c)
        ba = evaluate_prediction(y_test, y_hat)['balanced_accuracy']

        return ba

    def _select_best_parameter(self, async_result):

        c_values = []
        accuracies = []
        for fold in async_result.keys():
            best_c = -1
            best_acc = -1

            for c, async_acc in async_result[fold].items():

                acc = async_acc.get()
                if acc > best_acc:
                    best_c = c
                    best_acc = acc
            c_values.append(best_c)
            accuracies.append(best_acc)

        best_acc = np.mean(accuracies)
        best_c = np.power(10, np.mean(np.log10(c_values)))

        return {'c': best_c, 'balanced_accuracy': best_acc}

    def evaluate(self, train_index, test_index):

        inner_pool = ThreadPool(self._n_threads)
        async_result = {}
        for i in range(self._grid_search_folds):
            async_result[i] = {}

        if self._feature_selection_method == 'ANOVA':
            selector = SelectPercentile(f_classif, percentile=self._top_k)
            selector.fit(self._x[train_index], self._y[train_index])
            x_after = selector.transform(self._x)
        elif self._feature_selection_method == 'RF':
            clf = RandomForestClassifier(n_estimators=250, random_state=0, n_jobs=-1)
            clf.fit(self._x[train_index], self._y[train_index])
            selector = SelectFromModel(clf, threshold= self._top_k)
            selector.fit(self._x[train_index], self._y[train_index])
            x_after = selector.transform(self._x)
        elif self._feature_selection_method == 'PCA':
            selector = PCA(n_components=self._top_k)
            selector.fit(self._x[train_index])
            x_after = selector.transform(self._x)
        elif self._feature_selection_method == 'RFE':
            svc = SVR(kernel="linear")
            selector = RFE(estimator=svc, n_features_to_select=int(0.01 * self._top_k * self._x[train_index].shape[1]), step=0.5)
            selector.fit(self._x[train_index], self._y[train_index])
            x_after = selector.transform(self._x)

        self._kernel = gram_matrix_linear(x_after)

        outer_kernel = self._kernel[train_index, :][:, train_index]
        y_train = self._y[train_index]

        skf = StratifiedKFold(n_splits=self._grid_search_folds, shuffle=True)
        inner_cv = list(skf.split(np.zeros(len(y_train)), y_train))

        for i in range(len(inner_cv)):
            inner_train_index, inner_test_index = inner_cv[i]

            inner_kernel = outer_kernel[inner_train_index, :][:, inner_train_index]
            x_test_inner = outer_kernel[inner_test_index, :][:, inner_train_index]
            y_train_inner, y_test_inner = y_train[inner_train_index], y_train[inner_test_index]

            for c in self._c_range:
                if self._verbose:
                    print("Inner CV for C=%f..." % c)
                async_result[i][c] = inner_pool.apply_async(self._grid_search, args=(inner_kernel, x_test_inner, y_train_inner,
                                                                          y_test_inner, c))
        inner_pool.close()
        inner_pool.join()

        best_parameter = self._select_best_parameter(async_result)
        x_test = self._kernel[test_index, :][:, train_index]
        y_train, y_test = self._y[train_index], self._y[test_index]

        _, y_hat, auc, y_hat_train, proba_test_index1 = self._launch_svc(outer_kernel, x_test, y_train, y_test, best_parameter['c'])

        result = dict()
        result['best_parameter'] = best_parameter
        result['evaluation'] = evaluate_prediction(y_test, y_hat)
        result['evaluation_train'] = evaluate_prediction(y_train, y_hat_train)
        result['y_hat'] = y_hat
        result['y_hat_train'] = y_hat_train
        result['y'] = y_test
        result['y_train'] = y_train
        result['y_index'] = test_index
        result['x_index'] = train_index
        result['auc'] = auc
        result['proba_test_index1'] = proba_test_index1

        return result

    def apply_best_parameters(self, results_list):

        best_c_list = []
        bal_acc_list = []

        for result in results_list:
            best_c_list.append(result['best_parameter']['c'])
            bal_acc_list.append(result['best_parameter']['balanced_accuracy'])

        # 10^(mean of log10 of best Cs of each fold) is selected
        best_c = np.power(10, np.mean(np.log10(best_c_list)))
        # Mean balanced accuracy
        mean_bal_acc = np.mean(bal_acc_list)

        if self._balanced:
            svc = SVC(C=best_c, kernel='precomputed', probability=True, tol=1e-6, class_weight='balanced')
        else:
            svc = SVC(C=best_c, kernel='precomputed', probability=True, tol=1e-6)

        svc.fit(self._kernel, self._y)

        return svc, {'c': best_c, 'balanced_accuracy': mean_bal_acc}

    def save_classifier(self, classifier, output_dir):

        np.savetxt(os.path.join(output_dir, 'dual_coefficients.txt'), classifier.dual_coef_)
        np.savetxt(os.path.join(output_dir, 'support_vectors_indices.txt'), classifier.support_)
        np.savetxt(os.path.join(output_dir, 'intersect.txt'), classifier.intercept_)

    def save_weights(self, classifier, x, output_dir):

        dual_coefficients = classifier.dual_coef_
        sv_indices = classifier.support_

        weighted_sv = dual_coefficients.transpose() * x[sv_indices]
        weights = np.sum(weighted_sv, 0)

        np.savetxt(os.path.join(output_dir, 'weights.txt'), weights)

        return weights

    def save_parameters(self, parameters_dict, output_dir):
        with open(os.path.join(output_dir, 'best_parameters.json'), 'w') as f:
            json.dump(parameters_dict, f)

class KFoldCV(ClassificationValidation):
    """
    KFold CV.
    """
    def __init__(self, ml_algorithm):
        self._ml_algorithm = ml_algorithm
        self._fold_results = []
        self._classifier = None
        self._best_params = None
        self._cv = None

    def validate(self, y, n_folds=10, n_threads=15, splits_indices=None, verbose=False):

        if splits_indices is None:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, )
            self._cv = list(skf.split(np.zeros(len(y)), y))
        else:
            self._cv = splits_indices

        async_pool = ThreadPool(n_threads)
        async_result = {}

        for i in range(n_folds):
            if verbose:
                print("Repetition %d of CV..." % i)
            train_index, test_index = self._cv[i]
            async_result[i] = async_pool.apply_async(self._ml_algorithm.evaluate, args=(train_index, test_index))
        async_pool.close()
        async_pool.join()

        for i in range(n_folds):
            self._fold_results.append(async_result[i].get())

        ## save the mean of the best models
        self._classifier, self._best_params = self._ml_algorithm.apply_best_parameters(self._fold_results)

        return self._classifier, self._best_params, self._fold_results

    def save_results(self, output_dir):
        if self._fold_results is None:
            raise Exception("No results to save. Method validate() must be run before save_results().")

        subjects_folds = []
        results_folds = []
        container_dir = os.path.join(output_dir, 'folds')

        if not os.path.exists(container_dir):
            os.makedirs(container_dir)

        for i in range(len(self._fold_results)):
            subjects_df = pd.DataFrame({'y': self._fold_results[i]['y'],
                                        'y_hat': self._fold_results[i]['y_hat'],
                                        'y_index': self._fold_results[i]['y_index']})
            subjects_df.to_csv(os.path.join(container_dir, 'subjects_fold-' + str(i) + '.tsv'),
                               index=False, sep='\t', encoding='utf-8')
            subjects_folds.append(subjects_df)

            results_df = pd.DataFrame({'balanced_accuracy': self._fold_results[i]['evaluation']['balanced_accuracy'],
                                       'auc': self._fold_results[i]['auc'],
                                       'accuracy': self._fold_results[i]['evaluation']['accuracy'],
                                       'sensitivity': self._fold_results[i]['evaluation']['sensitivity'],
                                       'specificity': self._fold_results[i]['evaluation']['specificity'],
                                       'ppv': self._fold_results[i]['evaluation']['ppv'],
                                       'npv': self._fold_results[i]['evaluation']['npv']}, index=['i', ])
            results_df.to_csv(os.path.join(container_dir, 'results_fold-' + str(i) + '.tsv'),
                              index=False, sep='\t', encoding='utf-8')
            results_folds.append(results_df)

        all_subjects = pd.concat(subjects_folds)
        all_subjects.to_csv(os.path.join(output_dir, 'subjects.tsv'),
                            index=False, sep='\t', encoding='utf-8')

        all_results = pd.concat(results_folds)
        all_results.to_csv(os.path.join(output_dir, 'results.tsv'),
                           index=False, sep='\t', encoding='utf-8')

        mean_results = pd.DataFrame(all_results.apply(np.nanmean).to_dict(), columns=all_results.columns, index=[0, ])
        mean_results.to_csv(os.path.join(output_dir, 'mean_results.tsv'),
                            index=False, sep='\t', encoding='utf-8')
        print("Mean results of the classification:")
        print("Balanced accuracy: %s" %(mean_results['balanced_accuracy'].to_string(index = False)))
        print("specificity: %s" % (mean_results['specificity'].to_string(index=False)))
        print("sensitivity: %s" % (mean_results['sensitivity'].to_string(index=False)))
        print("auc: %s" % (mean_results['auc'].to_string(index=False)))

class RepeatedHoldOut(ClassificationValidation):
    """
    Repeated holdout splits CV.
    """

    def __init__(self, ml_algorithm, n_iterations=100, test_size=0.3):
        self._ml_algorithm = ml_algorithm
        self._split_results = []
        self._classifier = None
        self._best_params = None
        self._cv = None
        self._n_iterations = n_iterations
        self._test_size = test_size
        self._error_resampled_t = None
        self._error_corrected_resampled_t = None
        self._bal_accuracy_resampled_t = None
        self._bal_accuracy_corrected_resampled_t = None

    def validate(self, y, n_threads=15, splits_indices=None, inner_cv=True, verbose=False):

        if splits_indices is None:
            splits = StratifiedShuffleSplit(n_splits=self._n_iterations, test_size=self._test_size)
            self._cv = list(splits.split(np.zeros(len(y)), y))
        else:
            self._cv = splits_indices
        async_pool = ThreadPool(n_threads)
        async_result = {}

        for i in range(self._n_iterations):
            if verbose:
                print("Repetition %d of CV..." % i)
            train_index, test_index = self._cv[i]
            if inner_cv:
                async_result[i] = async_pool.apply_async(self._ml_algorithm.evaluate, args=(train_index, test_index))
            else:
                raise Exception("We always do nested CV")

        async_pool.close()
        async_pool.join()

        for i in range(self._n_iterations):
            self._split_results.append(async_result[i].get())

        self._classifier, self._best_params = self._ml_algorithm.apply_best_parameters(self._split_results)
        return self._classifier, self._best_params, self._split_results

    def save_results(self, output_dir):
        if self._split_results is None:
            raise Exception("No results to save. Method validate() must be run before save_results().")

        all_results_list = []
        all_train_subjects_list = []
        all_test_subjects_list = []

        for iteration in range(len(self._split_results)):

            iteration_dir = os.path.join(output_dir, 'iteration-' + str(iteration))
            if not os.path.exists(iteration_dir):
                os.makedirs(iteration_dir)
            iteration_train_subjects_df = pd.DataFrame({'iteration': iteration,
                                                        'y': self._split_results[iteration]['y_train'],
                                                        'y_hat': self._split_results[iteration]['y_hat_train'],
                                                        'subject_index': self._split_results[iteration]['x_index']})
            iteration_train_subjects_df.to_csv(os.path.join(iteration_dir, 'train_subjects.tsv'),
                                               index=False, sep='\t', encoding='utf-8')
            all_train_subjects_list.append(iteration_train_subjects_df)

            iteration_test_subjects_df = pd.DataFrame({'iteration': iteration,
                                                       'y': self._split_results[iteration]['y'],
                                                       'y_hat': self._split_results[iteration]['y_hat'],
                                                       'subject_index': self._split_results[iteration]['y_index'],
                                                      'proba_test_index1': self._split_results[iteration]['proba_test_index1']})
            iteration_test_subjects_df.to_csv(os.path.join(iteration_dir, 'test_subjects.tsv'),
                                              index=False, sep='\t', encoding='utf-8')
            all_test_subjects_list.append(iteration_test_subjects_df)

            iteration_results_df = pd.DataFrame(
                    {'balanced_accuracy': self._split_results[iteration]['evaluation']['balanced_accuracy'],
                     'auc': self._split_results[iteration]['auc'],
                     'accuracy': self._split_results[iteration]['evaluation']['accuracy'],
                     'sensitivity': self._split_results[iteration]['evaluation']['sensitivity'],
                     'specificity': self._split_results[iteration]['evaluation']['specificity'],
                     'ppv': self._split_results[iteration]['evaluation']['ppv'],
                     'npv': self._split_results[iteration]['evaluation']['npv'],
                     'train_balanced_accuracy': self._split_results[iteration]['evaluation_train']['balanced_accuracy'],
                     'train_accuracy': self._split_results[iteration]['evaluation_train']['accuracy'],
                     'train_sensitivity': self._split_results[iteration]['evaluation_train']['sensitivity'],
                     'train_specificity': self._split_results[iteration]['evaluation_train']['specificity'],
                     'train_ppv': self._split_results[iteration]['evaluation_train']['ppv'],
                     'train_npv': self._split_results[iteration]['evaluation_train']['npv']
                     }, index=['i', ])
            iteration_results_df.to_csv(os.path.join(iteration_dir, 'results.tsv'),
                                        index=False, sep='\t', encoding='utf-8')

            all_results_list.append(iteration_results_df)

        all_train_subjects_df = pd.concat(all_train_subjects_list)
        all_train_subjects_df.to_csv(os.path.join(output_dir, 'train_subjects.tsv'),
                                     index=False, sep='\t', encoding='utf-8')

        all_test_subjects_df = pd.concat(all_test_subjects_list)
        all_test_subjects_df.to_csv(os.path.join(output_dir, 'test_subjects.tsv'),
                                    index=False, sep='\t', encoding='utf-8')

        all_results_df = pd.concat(all_results_list)
        all_results_df.to_csv(os.path.join(output_dir, 'results.tsv'),
                              index=False, sep='\t', encoding='utf-8')

        mean_results_df = pd.DataFrame(all_results_df.apply(np.nanmean).to_dict(),
                                       columns=all_results_df.columns, index=[0, ])
        mean_results_df.to_csv(os.path.join(output_dir, 'mean_results.tsv'),
                               index=False, sep='\t', encoding='utf-8')

        print("Mean results of the classification:")
        print("Balanced accuracy: %s" % (mean_results_df['balanced_accuracy'].to_string(index=False)))
        print("specificity: %s" % (mean_results_df['specificity'].to_string(index=False)))
        print("sensitivity: %s" % (mean_results_df['sensitivity'].to_string(index=False)))
        print("auc: %s" % (mean_results_df['auc'].to_string(index=False)))

        self.compute_error_variance()
        self.compute_accuracy_variance()

        variance_df = pd.DataFrame({'bal_accuracy_resampled_t': self._bal_accuracy_resampled_t,
                                    'bal_accuracy_corrected_resampled_t': self._bal_accuracy_corrected_resampled_t,
                                    'error_resampled_t': self._error_resampled_t,
                                    'error_corrected_resampled_t': self._error_corrected_resampled_t}, index=[0, ])

        variance_df.to_csv(os.path.join(output_dir, 'variance.tsv'),
                           index=False, sep='\t', encoding='utf-8')

    def _compute_variance(self, test_error_split):

        # compute average test error
        num_split = len(self._split_results)  # J in the paper

        # compute mu_{n_1}^{n_2}
        average_test_error = np.mean(test_error_split)

        approx_variance = np.sum((test_error_split - average_test_error)**2)/(num_split - 1)

        # compute variance (point 2 and 6 of Nadeau's paper)
        resampled_t = approx_variance / num_split
        corrected_resampled_t = (1/num_split + self._test_size/(1 - self._test_size)) * approx_variance

        return resampled_t, corrected_resampled_t

    def compute_error_variance(self):
        num_split = len(self._split_results)
        test_error_split = np.zeros((num_split, 1))  # this list will contain the list of mu_j hat for j = 1 to J
        for i in range(num_split):
            test_error_split[i] = self._compute_average_test_error(self._split_results[i]['y'],
                                                                   self._split_results[i]['y_hat'])

        self._error_resampled_t, self._error_corrected_resampled_t = self._compute_variance(test_error_split)

        return self._error_resampled_t, self._error_corrected_resampled_t

    def _compute_average_test_error(self, y_list, yhat_list):
        # return the average test error (denoted mu_j hat)
        return float(len(np.where(y_list != yhat_list)[0]))/float(len(y_list))

    def compute_accuracy_variance(self):
        num_split = len(self._split_results)
        test_accuracy_split = np.zeros((num_split, 1))  # this list will contain the list of mu_j hat for j = 1 to J
        for i in range(num_split):
            test_accuracy_split[i] = self._compute_average_test_accuracy(self._split_results[i]['y'],
                                                                         self._split_results[i]['y_hat'])

        self._bal_accuracy_resampled_t, self._bal_accuracy_corrected_resampled_t = self._compute_variance(test_accuracy_split)

        return self._bal_accuracy_resampled_t, self._bal_accuracy_corrected_resampled_t

    def _compute_average_test_accuracy(self, y_list, yhat_list):

        return evaluate_prediction(y_list, yhat_list)['balanced_accuracy']

class LinearSVMAlgorithmWithoutPrecomputedKernel(ClassificationAlgorithm):
    '''
    Linear SVM with input X, not with kernel method for regional features.
    '''

    def __init__(self, x, y, balanced=True, grid_search_folds=10, c_range=np.logspace(-6, 2, 17), n_threads=15,
                 verbose=False):
        self._x = x
        self._y = y
        self._balanced = balanced
        self._grid_search_folds = grid_search_folds
        self._c_range = c_range
        self._n_threads = n_threads
        self._verbose = verbose

    def _launch_svc(self, x_train, x_test, y_train, y_test, c):

        if self._balanced:
            svc = SVC(C=c, probability=True, tol=1e-6, class_weight='balanced', kernel='linear')
        else:
            svc = SVC(C=c, probability=True, tol=1e-6, kernel='linear')

        svc.fit(x_train, y_train)
        y_hat_train = svc.predict(x_train)
        y_hat = svc.predict(x_test)
        proba_test = svc.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, proba_test)

        return svc, y_hat, auc, y_hat_train

    def _grid_search(self, x_train, x_test, y_train, y_test, c):

        _, y_hat, _, _ = self._launch_svc(x_train, x_test, y_train, y_test, c)
        ba = evaluate_prediction(y_test, y_hat)['balanced_accuracy']

        return ba

    def _select_best_parameter(self, async_result):

        c_values = []
        accuracies = []
        for fold in async_result.keys():
            best_c = -1
            best_acc = -1

            for c, async_acc in async_result[fold].items():

                acc = async_acc.get()
                if acc > best_acc:
                    best_c = c
                    best_acc = acc
            c_values.append(best_c)
            accuracies.append(best_acc)

        best_acc = np.mean(accuracies)
        best_c = np.power(10, np.mean(np.log10(c_values)))

        return {'c': best_c, 'balanced_accuracy': best_acc}

    def evaluate(self, train_index, test_index):

        inner_pool = ThreadPool(self._n_threads)
        async_result = {}
        for i in range(self._grid_search_folds):
            async_result[i] = {}

        outer_x = self._x[train_index, :]
        y_train = self._y[train_index]

        skf = StratifiedKFold(n_splits=self._grid_search_folds, shuffle=True)
        inner_cv = list(skf.split(np.zeros(len(y_train)), y_train))

        for i in range(len(inner_cv)):
            inner_train_index, inner_test_index = inner_cv[i]

            inner_x = outer_x[inner_train_index, :]
            x_test_inner = outer_x[inner_test_index, :]
            y_train_inner, y_test_inner = y_train[inner_train_index], y_train[inner_test_index]

            for c in self._c_range:
                if self._verbose:
                    print("Inner CV for C=%f..." % c)
                async_result[i][c] = inner_pool.apply_async(self._grid_search, args=(inner_x, x_test_inner, y_train_inner,
                                                                                     y_test_inner, c))
        inner_pool.close()
        inner_pool.join() ##

        best_parameter = self._select_best_parameter(async_result)
        x_test = self._x[test_index, :]
        y_train, y_test = self._y[train_index], self._y[test_index]

        _, y_hat, auc, y_hat_train = self._launch_svc(outer_x, x_test, y_train, y_test, best_parameter['c'])

        result = dict()
        result['best_parameter'] = best_parameter
        result['evaluation'] = evaluate_prediction(y_test, y_hat)
        result['evaluation_train'] = evaluate_prediction(y_train, y_hat_train)
        result['y_hat'] = y_hat
        result['y_hat_train'] = y_hat_train
        result['y'] = y_test
        result['y_train'] = y_train
        result['y_index'] = test_index
        result['x_index'] = train_index
        result['auc'] = auc

        return result

    def apply_best_parameters(self, results_list):

        best_c_list = []
        bal_acc_list = []

        for result in results_list:
            best_c_list.append(result['best_parameter']['c'])
            bal_acc_list.append(result['best_parameter']['balanced_accuracy'])

        # 10^(mean of log10 of best Cs of each fold) is selected
        best_c = np.power(10, np.mean(np.log10(best_c_list)))
        # Mean balanced accuracy
        mean_bal_acc = np.mean(bal_acc_list)

        if self._balanced:
            svc = SVC(C=best_c, probability=True, tol=1e-6, class_weight='balanced', kernel='linear')
        else:
            svc = SVC(C=best_c, probability=True, tol=1e-6, kernel='linear')

        svc.fit(self._x, self._y)

        return svc, {'c': best_c, 'balanced_accuracy': mean_bal_acc}

    def save_classifier(self, classifier, output_dir):

        np.savetxt(os.path.join(output_dir, 'dual_coefficients.txt'), classifier.dual_coef_)
        np.savetxt(os.path.join(output_dir, 'support_vectors_indices.txt'), classifier.support_)
        np.savetxt(os.path.join(output_dir, 'intersect.txt'), classifier.intercept_)

    def save_weights(self, classifier, x, output_dir):

        dual_coefficients = classifier.dual_coef_
        sv_indices = classifier.support_

        weighted_sv = dual_coefficients.transpose() * x[sv_indices]
        weights = np.sum(weighted_sv, 0)

        np.savetxt(os.path.join(output_dir, 'weights.txt'), weights)

        return weights

    def save_parameters(self, parameters_dict, output_dir):
        with open(os.path.join(output_dir, 'best_parameters.json'), 'w') as f:
            json.dump(parameters_dict, f)

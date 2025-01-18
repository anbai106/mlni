from mlni.base import WorkFlow, RegressionAlgorithm, RegressionValidation
import numpy as np
import pandas as pd
import os, json
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, ShuffleSplit
from multiprocessing.pool import ThreadPool
from mlni.utils import time_bar
from joblib import dump
from sklearn.linear_model import Lasso

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020"
__credits__ = ["Junhao Wen, Jorge Samper-González"]
__license__ = "See LICENSE file"
__version__ = "0.1.4"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

class RB_RepeatedHoldOut_Lasso_Regression(WorkFlow):
    """
    The main class to run MLNI with repeated holdout CV for regression.
    """

    def __init__(self, input, split_index, output_dir, n_threads=8, n_iterations=100, test_size=0.2,
                 grid_search_folds=10, alpha_range=np.logspace(-6, 2, 17), verbose=False):
        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._grid_search_folds = grid_search_folds
        self._alpha_range = alpha_range
        self._verbose = verbose
        self._test_size = test_size
        self._validation = None
        self._algorithm = None

    def run(self):
        x = self._input.get_x()
        y = self._input.get_y_raw()

        self._algorithm = LassoRegressionAlgorithm(x, y,
                                                     grid_search_folds=self._grid_search_folds,
                                                     alpha_range=self._alpha_range,
                                                     n_threads=self._n_threads, verbose=self._verbose)

        self._validation = RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations, test_size=self._test_size)

        regressor, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._split_index, verbose=self._verbose)
        regressor_dir = os.path.join(self._output_dir, 'regressor')
        if not os.path.exists(regressor_dir):
            os.makedirs(regressor_dir)

        self._algorithm.save_regressor(regressor, regressor_dir)
        self._algorithm.save_parameters(best_params, regressor_dir)
        self._validation.save_results(self._output_dir)

class LassoRegressionAlgorithm(RegressionAlgorithm):
    '''
    Lasso linear regression.
    '''
    def __init__(self, x, y, grid_search_folds=10, alpha_range=np.logspace(-6, 2, 17), n_threads=15,
                 verbose=False):
        self._x = x
        self._y = y
        self._grid_search_folds = grid_search_folds
        self._alpha_range = alpha_range
        self._n_threads = n_threads
        self._verbose = verbose

    def _lauch_lasso(self, x_train, x_test, y_train, y_test, alpha):
        lasso = Lasso(alpha=alpha, tol=1e-6)
        lasso.fit(x_train, y_train)
        y_hat_train = lasso.predict(x_train)
        y_hat = lasso.predict(x_test)
        ## compute the mae
        mae = mean_absolute_error(y_test, y_hat)

        return lasso, y_hat, y_hat_train, mae

    def _grid_search(self, x_train, x_test, y_train, y_test, alpha):

        _, _, _, mae = self._lauch_lasso(x_train, x_test, y_train, y_test, alpha)

        return mae

    def _select_best_parameter(self, async_result):

        alpha_values = []
        maes = []
        for fold in async_result.keys():
            best_alpha = -1
            best_mae = np.inf

            for c, async_mae in async_result[fold].items():
                mae = async_mae.get()
                if mae < best_mae:
                    best_alpha = c
                    best_mae = mae
            alpha_values.append(best_alpha)
            maes.append(best_mae)

        best_mae_avg = np.mean(maes)
        best_alpha_mean = np.power(10, np.mean(np.log10(alpha_values)))

        best_mae_single = min(maes)
        best_alpha_single = alpha_values[maes.index(min(maes))]

        return {'alpha_mean': best_alpha_mean, 'mae_mean': best_mae_avg, 'alpha_single': best_alpha_single, 'mae_single': best_mae_single}

    def evaluate(self, train_index, test_index):
        inner_pool = ThreadPool(processes=self._n_threads)
        async_result = {}
        for i in range(self._grid_search_folds):
            async_result[i] = {}

        x_train = self._x[train_index]
        y_train = self._y[train_index]

        skf = KFold(n_splits=self._grid_search_folds, shuffle=True)
        inner_cv = list(skf.split(np.zeros(len(y_train)), y_train))

        for i in range(len(inner_cv)):
            inner_train_index, inner_test_index = inner_cv[i]
            x_train_inner = x_train[inner_train_index]
            x_test_inner = x_train[inner_test_index]
            y_train_inner, y_test_inner = y_train[inner_train_index], y_train[inner_test_index]

            for alpha in self._alpha_range:
                if self._verbose:
                    print("Inner CV for alpha=%f..." % alpha)
                async_result[i][alpha] = inner_pool.apply_async(self._grid_search, args=(x_train_inner, x_test_inner, y_train_inner, y_test_inner, alpha))
        inner_pool.close()
        inner_pool.join()

        best_parameter = self._select_best_parameter(async_result)
        x_train = self._x[train_index]
        x_test = self._x[test_index]
        y_train, y_test = self._y[train_index], self._y[test_index]

        _, y_hat, y_hat_train, mae_mean = self._lauch_lasso(x_train, x_test, y_train, y_test, best_parameter['alpha_mean'])
        _, _, _, mae_single = self._lauch_lasso(x_train, x_test, y_train, y_test, best_parameter['alpha_single'])

        result = dict()
        result['best_parameter'] = best_parameter
        result['y_hat'] = y_hat
        result['y_hat_train'] = y_hat_train
        result['y'] = y_test
        result['y_train'] = y_train
        result['y_index'] = test_index
        result['x_index'] = train_index
        result['mae_mean'] = mae_mean
        result['mae_single'] = mae_single

        return result

    def apply_best_parameters(self, results_list):

        best_alpha_list_mean = []
        bal_mae_list_mean = []
        best_alpha_list_single = []
        bal_mae_list_single = []

        for result in results_list:
            best_alpha_list_mean.append(result['best_parameter']['alpha_mean'])
            bal_mae_list_mean.append(result['best_parameter']['mae_mean'])
            best_alpha_list_single.append(result['best_parameter']['alpha_single'])
            bal_mae_list_single.append(result['best_parameter']['mae_single'])

        # 10^(mean of log10 of best Cs of each fold) is selected
        best_alpha = np.power(10, np.mean(np.log10(best_alpha_list_mean)))
        # MAE
        mean_mae = np.mean(bal_mae_list_mean)
        lasso_mean = Lasso(alpha=best_alpha, tol=1e-6)
        lasso_mean.fit(self._x, self._y)

        ### also save the single model with the lowest MAE
        single_mae = min(bal_mae_list_single)
        min_mae_index = bal_mae_list_single.index(min(bal_mae_list_single))
        best_alpha_single = best_alpha_list_single[min_mae_index]
        lasso_single = Lasso(alpha=best_alpha_single, tol=1e-6)
        lasso_single.fit(self._x, self._y)

        lasso = [lasso_mean, lasso_single]

        return lasso, {'alpha_mean': best_alpha, 'alpha_single': best_alpha_single, 'mae_mean': mean_mae, 'mae_single': single_mae}

    def save_regressor(self, regressor, output_dir):

        np.savetxt(os.path.join(output_dir, 'coefficients_mean.txt'), regressor[0].coef_)
        np.savetxt(os.path.join(output_dir, 'coefficients_single.txt'), regressor[1].coef_)
        np.savetxt(os.path.join(output_dir, 'intercept_mean.txt'),  np.array([regressor[0].intercept_]))
        np.savetxt(os.path.join(output_dir, 'intercept_single.txt'),  np.array([regressor[1].intercept_]))
        ## save the svr instance to apply external test data
        dump(regressor[0], os.path.join(output_dir, 'lasso_mean.joblib'))
        dump(regressor[1], os.path.join(output_dir, 'lasso_single.joblib'))

    def save_parameters(self, parameters_dict, output_dir):
        with open(os.path.join(output_dir, 'best_parameters.json'), 'w') as f:
            json.dump(parameters_dict, f)

class RepeatedHoldOut(RegressionValidation):
    """
    Repeated holdout splits CV.
    """

    def __init__(self, ml_algorithm, n_iterations=100, test_size=0.3):
        self._ml_algorithm = ml_algorithm
        self._split_results = []
        self._regressor = None ## this should be a list of two sklearn models
        self._best_params = None
        self._cv = None
        self._n_iterations = n_iterations
        self._test_size = test_size

    def validate(self, y, n_threads=15, splits_indices=None, inner_cv=True, verbose=False):

        if splits_indices is None:
            splits = ShuffleSplit(n_splits=self._n_iterations, test_size=self._test_size)
            self._cv = list(splits.split(np.zeros(len(y)), y))
        else:
            self._cv = splits_indices
        async_pool = ThreadPool(processes=n_threads)
        async_result = {}

        for i in range(self._n_iterations):
            time_bar(i, self._n_iterations)
            print()
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

        self._regressor, self._best_params = self._ml_algorithm.apply_best_parameters(self._split_results)
        return self._regressor, self._best_params, self._split_results

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
                                                       'subject_index': self._split_results[iteration]['y_index']})
            iteration_test_subjects_df.to_csv(os.path.join(iteration_dir, 'test_subjects.tsv'),
                                              index=False, sep='\t', encoding='utf-8')
            all_test_subjects_list.append(iteration_test_subjects_df)

            iteration_results_df = pd.DataFrame(
                    {'mae_mean': self._split_results[iteration]['mae_mean'],
                     'mae_single': self._split_results[iteration]['mae_single'],
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

        print("Mean results of the regression:")
        print("Mean absolute error for the average model: %s" % (mean_results_df['mae_mean'].to_string(index=False)))
        print("Mean absolute error for the single model: %s" % (mean_results_df['mae_single'].to_string(index=False)))

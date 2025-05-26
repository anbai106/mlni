from mlni.base import WorkFlow, RegressionAlgorithm, RegressionValidation
import numpy as np
import pandas as pd
import os, json
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, ShuffleSplit
from multiprocessing.pool import ThreadPool
from mlni.utils import time_bar
from joblib import dump
from sklearn.linear_model import ElasticNet
from itertools import product

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020"
__credits__ = ["Junhao Wen, Jorge Samper-González"]
__license__ = "See LICENSE file"
__version__ = "0.1.5.1"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

class RB_RepeatedHoldOut_ElasticNet_Regression(WorkFlow):
    """
    The main class to run MLNI with repeated holdout CV for regression.
    """

    def __init__(self, input, split_index, output_dir, n_threads=8, n_iterations=100, test_size=0.2,
                 grid_search_folds=10, alpha_range=np.logspace(-4, 0, 5), l1_ratio_range=np.linspace(0.1, 0.9, 5), verbose=False):
        self._input = input
        self._split_index = split_index
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._grid_search_folds = grid_search_folds
        self._alpha_range = alpha_range
        self._l1_ratio_range = l1_ratio_range
        self._verbose = verbose
        self._test_size = test_size
        self._validation = None
        self._algorithm = None

    def run(self):
        x = self._input.get_x()
        y = self._input.get_y_raw()

        self._algorithm = ElasticNetRegressionAlgorithm(x, y,
                                                     grid_search_folds=self._grid_search_folds,
                                                     alpha_range=self._alpha_range,
                                                     l1_ratio_range=self._l1_ratio_range,
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

class ElasticNetRegressionAlgorithm(RegressionAlgorithm):
    '''
    ElasticNet linear regression with nested cross-validation.
    '''

    def __init__(self, x, y, grid_search_folds=10,
                 alpha_range=np.logspace(-4, 0, 5),
                 l1_ratio_range=np.linspace(0.1, 0.9, 5),
                 n_threads=15, verbose=False):
        self._x = x
        self._y = y
        self._grid_search_folds = grid_search_folds
        self._alpha_range = alpha_range
        self._l1_ratio_range = l1_ratio_range
        self._n_threads = n_threads
        self._verbose = verbose

    def _launch_model(self, x_train, x_test, y_train, y_test, alpha, l1_ratio):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-6, max_iter=10000)
        model.fit(x_train, y_train)
        y_hat_train = model.predict(x_train)
        y_hat = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_hat)
        return model, y_hat, y_hat_train, mae

    def _grid_search(self, x_train, x_test, y_train, y_test, alpha, l1_ratio):
        _, _, _, mae = self._launch_model(x_train, x_test, y_train, y_test, alpha, l1_ratio)
        return mae

    def _select_best_parameter(self, async_result):
        best_combos = []
        maes = []
        for fold in async_result:
            best_combo = None
            best_mae = np.inf
            for (alpha, l1_ratio), async_mae in async_result[fold].items():
                mae = async_mae.get()
                if mae < best_mae:
                    best_combo = (alpha, l1_ratio)
                    best_mae = mae
            best_combos.append(best_combo)
            maes.append(best_mae)

        best_mae_avg = np.mean(maes)
        log_alphas = np.log10([c[0] for c in best_combos])
        l1_ratios = [c[1] for c in best_combos]
        best_alpha_mean = np.power(10, np.mean(log_alphas))
        best_l1_ratio_mean = np.mean(l1_ratios)

        best_alpha_single, best_l1_ratio_single = best_combos[np.argmin(maes)]
        best_mae_single = min(maes)

        return {
            'alpha_mean': best_alpha_mean, 'l1_ratio_mean': best_l1_ratio_mean,
            'mae_mean': best_mae_avg,
            'alpha_single': best_alpha_single, 'l1_ratio_single': best_l1_ratio_single,
            'mae_single': best_mae_single
        }

    def evaluate(self, train_index, test_index):
        inner_pool = ThreadPool(processes=self._n_threads)
        async_result = {i: {} for i in range(self._grid_search_folds)}

        x_train, y_train = self._x[train_index], self._y[train_index]
        skf = KFold(n_splits=self._grid_search_folds, shuffle=True)
        inner_cv = list(skf.split(np.zeros(len(y_train)), y_train))

        for i, (train_idx, test_idx) in enumerate(inner_cv):
            x_tr, x_te = x_train[train_idx], x_train[test_idx]
            y_tr, y_te = y_train[train_idx], y_train[test_idx]
            for alpha, l1_ratio in product(self._alpha_range, self._l1_ratio_range):
                if self._verbose:
                    print(f"Inner CV: alpha={alpha:.1e}, l1_ratio={l1_ratio:.2f}")
                async_result[i][(alpha, l1_ratio)] = inner_pool.apply_async(
                    self._grid_search, args=(x_tr, x_te, y_tr, y_te, alpha, l1_ratio))

        inner_pool.close()
        inner_pool.join()

        best_params = self._select_best_parameter(async_result)
        x_tr, x_te = self._x[train_index], self._x[test_index]
        y_tr, y_te = self._y[train_index], self._y[test_index]

        _, y_hat, y_hat_train, mae_mean = self._launch_model(x_tr, x_te, y_tr, y_te,
                                                             best_params['alpha_mean'], best_params['l1_ratio_mean'])
        _, _, _, mae_single = self._launch_model(x_tr, x_te, y_tr, y_te,
                                                 best_params['alpha_single'], best_params['l1_ratio_single'])

        return {
            'best_parameter': best_params,
            'y_hat': y_hat,
            'y_hat_train': y_hat_train,
            'y': y_te,
            'y_train': y_tr,
            'y_index': test_index,
            'x_index': train_index,
            'mae_mean': mae_mean,
            'mae_single': mae_single
        }

    def apply_best_parameters(self, results_list):
        alphas_mean, l1s_mean, maes_mean = [], [], []
        alphas_single, l1s_single, maes_single = [], [], []

        for r in results_list:
            p = r['best_parameter']
            alphas_mean.append(p['alpha_mean'])
            l1s_mean.append(p['l1_ratio_mean'])
            maes_mean.append(p['mae_mean'])
            alphas_single.append(p['alpha_single'])
            l1s_single.append(p['l1_ratio_single'])
            maes_single.append(p['mae_single'])

        best_alpha = np.power(10, np.mean(np.log10(alphas_mean)))
        best_l1_ratio = np.mean(l1s_mean)
        mean_mae = np.mean(maes_mean)
        model_mean = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, tol=1e-6, max_iter=10000)
        model_mean.fit(self._x, self._y)

        idx_best = np.argmin(maes_single)
        model_single = ElasticNet(alpha=alphas_single[idx_best], l1_ratio=l1s_single[idx_best], tol=1e-6, max_iter=10000)
        model_single.fit(self._x, self._y)

        return [model_mean, model_single], {
            'alpha_mean': best_alpha, 'l1_ratio_mean': best_l1_ratio, 'mae_mean': mean_mae,
            'alpha_single': alphas_single[idx_best], 'l1_ratio_single': l1s_single[idx_best],
            'mae_single': maes_single[idx_best]
        }

    def save_regressor(self, regressor, output_dir):
        np.savetxt(os.path.join(output_dir, 'coefficients_mean.txt'), regressor[0].coef_)
        np.savetxt(os.path.join(output_dir, 'coefficients_single.txt'), regressor[1].coef_)
        np.savetxt(os.path.join(output_dir, 'intercept_mean.txt'), np.array([regressor[0].intercept_]))
        np.savetxt(os.path.join(output_dir, 'intercept_single.txt'), np.array([regressor[1].intercept_]))
        dump(regressor[0], os.path.join(output_dir, 'elasticnet_mean.joblib'))
        dump(regressor[1], os.path.join(output_dir, 'elasticnet_single.joblib'))

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
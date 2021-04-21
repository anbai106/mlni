import os
import numpy as np
import pandas as pd
from pyhydra.utils import consensus_clustering, cv_cluster_stability, hydra_solver_svm
from pyhydra.base import WorkFlow

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen, Erdem Varol"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

class RB_DualSVM_Subtype(WorkFlow):
    """
    The main class to run pyhydra with repeated holdout CV for clustering.
    """

    def __init__(self, input, feature_tsv, split_index, cv_repetition, k_min, k_max, output_dir, balanced=True,
                 n_iterations=100, test_size=0.2, num_consensus=20, num_iteration=50, tol=1e-6, predefined_c=None,
                 weight_initialization_type='DPP', n_threads=8, save_models=False, verbose=True):

        self._input = input
        self._feature_tsv = feature_tsv
        self._split_index = split_index
        self._cv_repetition = cv_repetition
        self._n_iterations = n_iterations
        self._output_dir = output_dir
        self._k_min = k_min
        self._k_max = k_max
        self._balanced = balanced
        self._test_size = test_size
        self._num_consensus = num_consensus
        self._num_iteration = num_iteration
        self._tol = tol
        self._predefined_c = predefined_c
        self._weight_initialization_type = weight_initialization_type
        self._k_range_list = list(range(k_min, k_max + 1))
        self._n_threads = n_threads
        self._save_models = save_models
        self._verbose = verbose

    def run(self):
        x = self._input.get_x()
        y = self._input.get_y_raw()
        data_label_folds_ks = np.zeros((y.shape[0], self._cv_repetition, self._k_max - self._k_min + 1)).astype(int)

        for i in range(self._cv_repetition):
            for j in self._k_range_list:
                print('Applying pyHRDRA for finding %d clusters. Repetition: %d / %d...\n' % (j, i+1, self._cv_repetition))
                training_final_prediction = hydra_solver_svm(i, x[self._split_index[i][0]], y[self._split_index[i][0]], j, self._output_dir,
                                                         self._num_consensus, self._num_iteration, self._tol, self._balanced, self._predefined_c,
                                                         self._weight_initialization_type, self._n_threads, self._save_models, self._verbose)


                # change the final prediction's label: test data to be 0, the rest training data will b e updated by the model's prediction
                data_label_fold = y.copy()
                data_label_fold[self._split_index[i][1]] = 0 # all test data to be 0
                data_label_fold[self._split_index[i][0]] = training_final_prediction ## assign the training prediction
                data_label_folds_ks[:, i, j - self._k_min] = data_label_fold

        print('Estimating clustering stability...\n')
        ## for the adjusted rand index, only consider the PT results
        adjusted_rand_index_results = np.zeros(self._k_max - self._k_min + 1)
        index_pt = np.where(y == 1)[0]  # index for PTs
        for m in range(self._k_max - self._k_min + 1):
            result = data_label_folds_ks[:, :, m][index_pt]
            adjusted_rand_index_result = cv_cluster_stability(result, self._k_range_list[m])
            # saving each k result into the final adjusted_rand_index_results
            adjusted_rand_index_results[m] = adjusted_rand_index_result

        print('Computing the final consensus group membership...\n')
        final_assignment_ks = -np.ones((self._input.get_y_raw().shape[0], self._k_max - self._k_min + 1)).astype(int)
        for n in range(self._k_max - self._k_min + 1):
            result = data_label_folds_ks[:, :, n][index_pt]
            final_assignment_ks_pt = consensus_clustering(result, n + self._k_min)
            final_assignment_ks[index_pt, n] = final_assignment_ks_pt + 1

        print('Saving the final results...\n')
        # save_cluster_results(adjusted_rand_index_results, final_assignment_ks)
        columns = ['ari_' + str(i) + '_subtypes' for i in self._k_range_list]
        ari_df = pd.DataFrame(adjusted_rand_index_results[:, np.newaxis].transpose(), columns=columns)
        ari_df.to_csv(os.path.join(self._output_dir, 'adjusted_rand_index.tsv'), index=False, sep='\t',
                      encoding='utf-8')

        # save the final assignment for consensus clustering across different folds
        df_feature = pd.read_csv(self._feature_tsv, sep='\t')
        columns = ['assignment_' + str(i) for i in self._k_range_list]
        participant_df = df_feature.iloc[:, :3]
        cluster_df = pd.DataFrame(final_assignment_ks, columns=columns)
        all_df = pd.concat([participant_df, cluster_df], axis=1)
        all_df.to_csv(os.path.join(self._output_dir, 'clustering_assignment.tsv'), index=False,
                      sep='\t', encoding='utf-8')
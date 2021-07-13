from mlni.regression_rbf import RB_RepeatedHoldOut_DualSVM_Regression, RB_KFold_DualSVM_Regression, \
    VB_RepeatedHoldOut_DualSVM_Regression, VB_KFold_DualSVM_Regression
from mlni.base import RB_Input, VB_Input
import os, pickle
from mlni.utils import make_cv_partition

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def regression_roi(feature_tsv, output_dir, cv_repetition, cv_strategy='hold_out', n_threads=8, seed=None, verbose=False):
    """
    Core function for regression with ROI-based features

    Args:
        feature_tsv:str, path to the tsv containing extracted feature, following the BIDS convention. The tsv contains
        the following headers: "
                                 "i) the first column is the participant_id;"
                                 "ii) the second column should be the session_id;"
                                 "iii) the third column should be the diagnosis;"
                                 "The following column should be the extracted features. e.g., the ROI features"
        output_dir: str, path to store the regression results.
        cv_repetition: int, number of repetitions for cross-validation (CV)
        cv_strategy: str, cross validation strategy used. Default is hold_out. choices=['k_fold', 'hold_out']
        n_threads: int, default is 8. The number of threads to run model in parallel.
        verbose: Bool, default is False. If the output message is verbose.

    Returns: regression outputs.

    """
    print('MLNI for a regression with nested CV...')
    input_data = RB_Input(feature_tsv, standardization_method="minmax")

    ## data split
    print('Data split was performed based on validation strategy: %s...\n' % cv_strategy)
    ## check if data split has been done, if yes, the pickle file is there
    if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')):
        split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl'), 'rb'))
    else:
        split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition, seed=seed)
    print('Data split has been done!\n')

    print('Starts regression with SVR...')
    ## Here, we perform a nested CV (outer CV with defined CV method, inner CV with 10-fold grid search) for regression.
    if cv_strategy == 'hold_out':
        wf_regression = RB_RepeatedHoldOut_DualSVM_Regression(input_data, split_index, os.path.join(output_dir, 'regression'),
                                        n_threads=n_threads, n_iterations=cv_repetition, verbose=verbose)
        wf_regression.run()
    elif cv_strategy == 'k_fold':
        wf_regression = RB_KFold_DualSVM_Regression(input_data, split_index, os.path.join(output_dir, 'regression'),
                                        cv_repetition, n_threads=n_threads, verbose=verbose)
        wf_regression.run()
    else:
        raise Exception("CV methods have not been implemented")

    print('Finish...')

def regression_voxel(participant_tsv, output_dir, cv_repetition, cv_strategy='hold_out', n_threads=8, seed=None, verbose=False):
    """
    Core function for regression with voxel-wise images

    Args:
        participant_tsv:str, path to the tsv containing extracted feature, following the BIDS convention. The tsv contains
        the following headers: "
                                 "i) the first column is the participant_id;"
                                 "ii) the second column should be the session_id;"
                                 "iii) the third column should be the diagnosis;"
                                 "iv) the forth column should be the path to each image;"
        output_dir: str, path to store the regression results.
        cv_repetition: int, number of repetitions for cross-validation (CV)
        cv_strategy: str, cross validation strategy used. Default is hold_out. choices=['k_fold', 'hold_out']
        n_threads: int, default is 8. The number of threads to run model in parallel.
        verbose: Bool, default is False. If the output message is verbose.

    Returns: regression outputs.

    """
    print('MLNI for a regression with nested CV...')
    input_data = VB_Input(participant_tsv)

    ## data split
    print('Data split was performed based on validation strategy: %s...\n' % cv_strategy)
    ## check if data split has been done, if yes, the pickle file is there
    if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')):
        split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl'), 'rb'))
    else:
        split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition, seed=seed)
    print('Data split has been done!\n')

    print('Starts regression with SVR...')
    ## Here, we perform a nested CV (outer CV with defined CV method, inner CV with 10-fold grid search) for regression.
    if cv_strategy == 'hold_out':
        wf_regression = VB_RepeatedHoldOut_DualSVM_Regression(input_data, split_index, os.path.join(output_dir, 'regression'),
                                        n_threads=n_threads, n_iterations=cv_repetition, verbose=verbose)
        wf_regression.run()
    elif cv_strategy == 'k_fold':
        wf_regression = VB_KFold_DualSVM_Regression(input_data, split_index, os.path.join(output_dir, 'regression'),
                                        cv_repetition, n_threads=n_threads, verbose=verbose)
        wf_regression.run()
    else:
        raise Exception("CV methods have not been implemented")

    print('Finish...')
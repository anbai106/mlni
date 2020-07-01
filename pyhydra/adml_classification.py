from .classification import RB_RepeatedHoldOut_DualSVM_Classification, RB_KFold_DualSVM_Classification
from .base import RB_Input
import os, pickle
from .utils import make_cv_partition

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def classification(feature_tsv, output_dir, cv_repetition, cv_strategy='hold_out', class_weight_balanced=True,
                           n_threads=8, verbose=False):
    """
    pyhydra core function for classification

    Args:
        feature_tsv:str, path to the tsv containing extracted feature, following the BIDS convention. The tsv contains
        the following headers: "
                                 "i) the first column is the participant_id;"
                                 "ii) the second column should be the session_id;"
                                 "iii) the third column should be the diagnosis;"
                                 "The following column should be the extracted features. e.g., the ROI features"
        output_dir: str, path to store the classification results.
        cv_repetition: int, number of repetitions for cross-validation (CV)
        cv_strategy: str, cross validation strategy used. Default is hold_out. choices=['k_fold', 'hold_out']
        class_weight_balanced: Bool, default is True. If the two groups are balanced.
        n_threads: int, default is 8. The number of threads to run model in parallel.
        verbose: Bool, default is False. If the output message is verbose.

    Returns: classification outputs.

    """
    print('pyhydra for a binary classification with nested CV...')
    input_data = RB_Input(feature_tsv, covariate_tsv=None)

    ## data split
    print('Data split was performed based on validation strategy: %s...\n' % cv_strategy)
    ## check if data split has been done, if yes, the pickle file is there
    if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')):
        split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl'), 'rb'))
    else:
        split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition)
    print('Data split has been done!\n')

    print('Starts binary classification...')
    ## Here, we perform a nested CV (outer CV with defined CV method, inner CV with 10-fold grid search) for classification.
    if cv_strategy == 'hold_out':
        wf_classification = RB_RepeatedHoldOut_DualSVM_Classification(input_data, split_index, os.path.join(output_dir, 'classification'),
                                        n_threads=n_threads, n_iterations=cv_repetition, balanced=class_weight_balanced, verbose=verbose)
        wf_classification.run()
    elif cv_strategy == 'k_fold':
        wf_classification = RB_KFold_DualSVM_Classification(input_data, split_index, os.path.join(output_dir, 'classification'),
                                        cv_repetition, n_threads=n_threads, balanced=class_weight_balanced, verbose=verbose)
        wf_classification.run()
    else:
        raise Exception("CV methods have not been implemented")

    print('Finish...')

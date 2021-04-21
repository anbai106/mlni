from pyhydra.clustering import RB_DualSVM_Subtype
from pyhydra.base import RB_Input
import os, pickle
from pyhydra.utils import make_cv_partition

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen, Erdem Varol"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def clustering(feature_tsv, output_dir, k_min, k_max, cv_repetition, covariate_tsv=None, cv_strategy='hold_out', save_models=False,
            cluster_predefined_c=0.25, class_weight_balanced=True, weight_initialization_type='DPP', num_iteration=50,
            num_consensus=20, tol=1e-8, n_threads=8, verbose=False):
    """
    pyhydra core function for clustering
    Args:
        feature_tsv:str, path to the tsv containing extracted feature, following the BIDS convention. The tsv contains
        the following headers: "
                                 "i) the first column is the participant_id;"
                                 "ii) the second column should be the session_id;"
                                 "iii) the third column should be the diagnosis;"
                                 "The following column should be the extracted features. e.g., the ROI features"
        output_dir: str, path to store the clustering results
        k_min: int, minimum k (number of clusters)
        k_max: int, maximum k (number of clusters)
        cv_repetition: int, number of repetitions for cross-validation (CV)
        covariate_tsv: str, path to the tsv containing the covariates, eg., age or sex. The header (first 3 columns) of
                     the tsv file is the same as the feature_tsv, following the BIDS convention.
        cv_strategy: str, cross validation strategy used. Default is hold_out. choices=['k_fold', 'hold_out']
        save_models: Bool, if save all models during CV. Default is False to save space.
                      Set true only if you are going to apply the trained model to unseen data.
        cluster_predefined_c: Float, default is 0.25. The predefined best c if you do not want to perform a nested CV to
                             find it. If used, it should be a float number
        class_weight_balanced: Bool, default is True. If the two groups are balanced.
        weight_initialization_type: str, default is DPP. The strategy for initializing the weight to control the
                                    hyperplances and the subpopulation of patients. choices=["random_hyperplane", "random_assign", "k_means", "DPP"]
        num_iteration: int, default is 50. The number of iterations to iteratively optimize the polytope.
        num_consensus: int, default is 20. The number of repeats for consensus clustering to eliminate the unstable clustering.
        tol: float, default is 1e-8. Clustering stopping criterion.
        n_threads: int, default is 8. The number of threads to run model in parallel.
        verbose: Bool, default is False. If the output message is verbose.

    Returns: clustering outputs.

    """
    print('pyhydra for semi-supervised clustering...')
    if covariate_tsv == None:
        input_data = RB_Input(feature_tsv, covariate_tsv=None)
    else:
        input_data = RB_Input(feature_tsv, covariate_tsv=covariate_tsv)

    ## data split
    print('Data split was performed based on validation strategy: %s...\n' % cv_strategy)
    if cv_strategy == "hold_out":
        ## check if data split has been done, if yes, the pickle file is there
        if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')):
            split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl'), 'rb'))
        else:
            split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition)
    elif cv_strategy == "k_fold":
        ## check if data split has been done, if yes, the pickle file is there
        if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-fold.pkl')):
            split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-fold.pkl'), 'rb'))
        else:
            split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition)

    print('Data split has been done!\n')

    print('Starts semi-supervised clustering...')
    ## Here, semi-supervised clustering
    wf_clustering = RB_DualSVM_Subtype(input_data, feature_tsv, split_index, cv_repetition, k_min, k_max,
                                                       os.path.join(output_dir, 'clustering'), balanced=class_weight_balanced,
                                                       num_consensus=num_consensus, num_iteration=num_iteration,
                                                       tol=tol, predefined_c=cluster_predefined_c,
                                                       weight_initialization_type=weight_initialization_type,
                                                       n_threads=n_threads, save_models=save_models, verbose=verbose)

    wf_clustering.run()
    print('Finish...')

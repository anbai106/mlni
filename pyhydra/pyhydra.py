import argparse
from clustering import RB_RepeatedHoldOut_DualSVM_Subtype
from classification import RB_RepeatedHoldOut_DualSVM_Classification
from base import RB_Input
import os, pickle
from utils import make_cv_partition

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

# parser = argparse.ArgumentParser(description="Argparser for HYDRA, which is BIDS compliant")
#
# # Mandatory argument
# parser.add_argument("--feature_tsv", type=str, default="/home/hao/Project/pyhydra/data/test_feature.tsv",
#                            help="Path to the tsv containing extracted feature. The tsv contains the following headers: "
#                                 "ii) the first column is the participant_id;"
#                                 "iii) the second column should be the session_id;"
#                                 "iv) the third column should be the diagnosis;"
#                                 "v) the following column should be the extracted features. e.g., the ROI features")
# parser.add_argument("--output_dir", type=str, default="/home/hao/test/pyhydra",
#                            help="Path to store the clustering results")
#
# # Optional argument
# parser.add_argument("--covariate_tsv", default='/home/hao/Project/pyhydra/data/test_covariate.tsv',
#                     help="Path to the tsv containing the information of the covariates for correction."
#                          "The first three columns has the same header as the feature_tsv. Covariate effect can be age or sex")
# parser.add_argument("--cv_strategy", default='hold_out', type=str, choices=['k_fold', 'hold_out'],
#                     help="cross validation strategy used. Default is hold_out")
# parser.add_argument("--classification", default=False, action='store_true',
#                     help="If SVM classification should be performed")
# parser.add_argument("--save_models", default=False, action='store_true',
#                     help="If save all models during CV. Default is False to save space. Set true only if you are going to apply the trained model to unseen data")
# parser.add_argument("--cluster_predefined_c", default=0.25,
#                     help="the predefined best c if you do not want to perform a nested CV to find it. If used, it should be a float number")
# parser.add_argument("--class_weight_balanced", default=True, action='store_true',
#                     help="If the two groups are balanced")
# parser.add_argument("--weight_initialization_type", default="DPP", type=str, choices=["random_hyperplane", "random_assign", "k_means", "DPP"],
#                     help="The strategy for initializing the weight to control the hyperplances and the subpopulation of patients")
# parser.add_argument("--num_iteration", default=50, type=int,
#                     help="the number of iterations to iteratively optimize the polytope")
# parser.add_argument("--num_consensus", default=20, type=int,
#                     help="the number of repeats for consensus clustering to eliminate the unstable clustering")
# parser.add_argument("--k_min", default=2, type=int,
#                     help="the minimum k for clustering solutions to evaluate")
# parser.add_argument("--k_max", default=8, type=int,
#                     help="the maximum k for clustering solutions to evaluate")
# parser.add_argument("--cv_repetition", default=2, type=int,
#                     help="number of repetitions or folds for cross validation, depending on the cross validation strategy")
# parser.add_argument("--tol", default=1e-8, type=float,
#                     help="clustering stopping criteria")
# parser.add_argument("--n_threads", default=8, type=int,
#                     help="number of threads to run in parallel for classification")
# parser.add_argument("--verbose", default=False, action='store_true',
#                     help="If the output message is verbose")

def pyhydra(feature_tsv, output_dir, k_min, k_max, cv_repetition, covariate_tsv=None, cv_strategy='hold_out', classification=False, save_models=False,
            cluster_predefined_c=0.25, class_weight_balanced=True, weight_initialization_type='DPP',num_iteration=50,
            num_consensus=20, tol=1e-8, n_threads=8, verbose=False):
    """
    pyhydra core function
    Args:
        feature_tsv:Path to the tsv containing extracted feature. The tsv contains the following headers: "
                                 "ii) the first column is the participant_id;"
                                 "iii) the second column should be the session_id;"
                                 "iv) the third column should be the diagnosis;"
                                 "v) the following column should be the extracted features. e.g., the ROI features"
        output_dir:
        k_min:
        k_max:
        cv_repetition:
        covariate_tsv:
        cv_strategy:
        classification:
        save_models:
        cluster_predefined_c:
        class_weight_balanced:
        weight_initialization_type:
        num_iteration:
        num_consensus:
        tol:
        n_threads:
        verbose:

    Returns:

    """
    print('pyhydra for a binary classification or semi-supervised clustering...')
    if covariate_tsv == None:
        input_data = RB_Input(feature_tsv, covariate_tsv=None)
    else:
        input_data = RB_Input(feature_tsv, covariate_tsv=covariate_tsv)

    ## data split
    print('Data split was performed based on validation strategy: %s...\n' % cv_strategy)
    ## check if data split has been done, if yes, the pickle file is there
    if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')):
        split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl'), 'rb'))
    else:
        split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition)
    print('Data split has been done!\n')

    ## Check if classification should be done with nested CV followed by this paper: 'Reproducible Evaluation of Diffusion MRI Features for Automatic Classification of Patients with Alzheimer’s Disease'
    if classification:
        print('Starts binary classification...')
        ## Here, we perform a nested CV (outer CV with defined times repeated hold out, inner CV with 10-fold grid search) for global classification.
        wf_classification = RB_RepeatedHoldOut_DualSVM_Classification(input_data, split_index, os.path.join(output_dir, 'classification'),
                                        n_threads, cv_repetition, balanced=class_weight_balanced)
        wf_classification.run()
    else:
        print('Starts semi-supervised clustering...')
        ## Here, semi-supervised clustering
        wf_clustering = RB_RepeatedHoldOut_DualSVM_Subtype(input_data, feature_tsv, split_index, cv_repetition, k_min, k_max,
                                                           os.path.join(output_dir, 'clustering'), balanced=class_weight_balanced,
                                                           num_consensus=num_consensus, num_iteration=num_iteration,
                                                           tol=tol, predefined_c=cluster_predefined_c,
                                                           weight_initialization_type=weight_initialization_type,
                                                           save_models=save_models, verbose=verbose)

        wf_clustering.run()

# if __name__ == "__main__":
#     commandline = parser.parse_known_args()
#     options = commandline[0]
#     if commandline[1]:
#         raise Exception("unknown arguments: %s" % parser.parse_known_args()[1])
#     main(options)

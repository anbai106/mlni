from .classification import RB_RepeatedHoldOut_DualSVM_Classification, RB_KFold_DualSVM_Classification, \
    VB_RepeatedHoldOut_DualSVM_Classification, VB_KFold_DualSVM_Classification, RB_RepeatedHoldOut_DualSVM_Classification_Nested_Feature_Selection, \
    VB_RepeatedHoldOut_DualSVM_Classification_Nested_Feature_Selection
from .base import RB_Input, VB_Input
import os, pickle
from .utils import make_cv_partition, soft_majority_voting
import pandas as pd

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def classification_roi(feature_tsv, output_dir, cv_repetition, cv_strategy='hold_out', class_weight_balanced=True,
                           n_threads=8, seed=None, verbose=False):
    """
    pyhydra core function for classification for ROI-based features

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
    input_data = RB_Input(feature_tsv)

    ## data split
    print('Data split was performed based on validation strategy: %s...\n' % cv_strategy)
    ## check if data split has been done, if yes, the pickle file is there
    if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')):
        split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl'), 'rb'))
    else:
        split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition, seed=seed)
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

def classification_roi_feature_selection(feature_tsv, output_dir, cv_repetition, cv_strategy='hold_out',
                           class_weight_balanced=True, feature_selection_method='RFE', top_k=50, n_threads=8, seed=None, verbose=False):
    """
    pyhydra core function for classification for ROI-based features with nested feature selection

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
        feature_selection_method: str, default is RFE. choices=['ANOVA', 'RF', 'PCA', 'RFE'].
        top_k: int, default is 50 (50%). Percentage of original feature that the method want to select.
        n_threads: int, default is 8. The number of threads to run model in parallel.
        verbose: Bool, default is False. If the output message is verbose.

    Returns: classification outputs.

    """
    print('pyhydra for a binary classification with nested CV and nested feature selection method...')
    input_data = RB_Input(feature_tsv)

    ## data split
    print('Data split was performed based on validation strategy: %s...\n' % cv_strategy)
    ## check if data split has been done, if yes, the pickle file is there
    if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')):
        split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl'), 'rb'))
    else:
        split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition, seed=seed)
    print('Data split has been done!\n')

    print('Starts binary classification...')
    ## Here, we perform a nested CV (outer CV with defined CV method, inner CV with 10-fold grid search) for classification.
    if cv_strategy == 'hold_out':
        wf_classification = RB_RepeatedHoldOut_DualSVM_Classification_Nested_Feature_Selection(input_data, split_index,
                           os.path.join(output_dir, 'classification'), n_threads=n_threads, n_iterations=cv_repetition,
       balanced=class_weight_balanced, feature_selection_method=feature_selection_method, top_k=top_k, verbose=verbose)
        wf_classification.run()
    elif cv_strategy == 'k_fold':
        raise Exception("Non-nested feature selection is currently only supported for repeated hold-out CV")
    else:
        raise Exception("CV methods have not been implemented")

    print('Finish...')

def classification_voxel_feature_selection(feature_tsv, output_dir, cv_repetition, cv_strategy='hold_out', class_weight_balanced=True,
                                           feature_selection_method='RFE', top_k=50, n_threads=8, seed=None, verbose=False):
    """
    pyhydra core function for classification with voxel-wise features

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
        feature_selection_method: str, default is RFE. choices=['ANOVA', 'RF', 'PCA', 'RFE'].
        top_k: int, default is 50 (50%). Percentage of original feature that the method want to select.
        n_threads: int, default is 8. The number of threads to run model in parallel.
        verbose: Bool, default is False. If the output message is verbose.

    Returns: classification outputs.

    """
    print('pyhydra for a binary classification with nested CV and nested feature selection method...')
    input_data =VB_Input(feature_tsv)

    ## data split
    print('Data split was performed based on validation strategy: %s...\n' % cv_strategy)
    ## check if data split has been done, if yes, the pickle file is there
    if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')):
        split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl'), 'rb'))
    else:
        split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition, seed=seed)
    print('Data split has been done!\n')

    print('Starts binary classification...')
    ## Here, we perform a nested CV (outer CV with defined CV method, inner CV with 10-fold grid search) for classification.
    if cv_strategy == 'hold_out':
        wf_classification = VB_RepeatedHoldOut_DualSVM_Classification_Nested_Feature_Selection(input_data, split_index, os.path.join(output_dir, 'classification'),
        n_threads=n_threads, n_iterations=cv_repetition, balanced=class_weight_balanced, feature_selection_method=feature_selection_method, top_k=top_k,
        verbose=verbose)
        wf_classification.run()
    elif cv_strategy == 'k_fold':
        raise Exception("Non-nested feature selection is currently only supported for repeated hold-out CV")
    else:
        raise Exception("CV methods have not been implemented")

    print('Finish...')

def classification_voxel(feature_tsv, output_dir, cv_repetition, cv_strategy='hold_out', class_weight_balanced=True, n_threads=8, seed=None, verbose=False):
    """
    pyhydra core function for classification with voxel-wise features

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
    input_data =VB_Input(feature_tsv)

    ## data split
    print('Data split was performed based on validation strategy: %s...\n' % cv_strategy)
    ## check if data split has been done, if yes, the pickle file is there
    if os.path.isfile(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')):
        split_index = pickle.load(open(os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl'), 'rb'))
    else:
        split_index, _ = make_cv_partition(input_data.get_y(), cv_strategy, output_dir, cv_repetition, seed=seed)
    print('Data split has been done!\n')

    print('Starts binary classification...')
    ## Here, we perform a nested CV (outer CV with defined CV method, inner CV with 10-fold grid search) for classification.
    if cv_strategy == 'hold_out':
        wf_classification = VB_RepeatedHoldOut_DualSVM_Classification(input_data, split_index, os.path.join(output_dir, 'classification'),
                                        n_threads=n_threads, n_iterations=cv_repetition, balanced=class_weight_balanced, verbose=verbose)
        wf_classification.run()
    elif cv_strategy == 'k_fold':
        wf_classification = VB_KFold_DualSVM_Classification(input_data, split_index, os.path.join(output_dir, 'classification'),
                                        cv_repetition, n_threads=n_threads, balanced=class_weight_balanced, verbose=verbose)
        wf_classification.run()
    else:
        raise Exception("CV methods have not been implemented")

    print('Finish...')

def classification_multiscale_opnmf(participant_tsv, opnmf_dir, output_dir, num_components_min, num_components_max, num_components_step, cv_repetition, cv_strategy='hold_out',
            class_weight_balanced=True, n_threads=8, verbose=False):
    """
    classification based on the multi-scale feature extracted from opNMF
    Args:
    Returns: classification outputs.

    """
    ### For voxel approach
    print('Multi-scale ensemble classification...')
    print('Starts classification for each specific scale...')
    ### Here, semi-supervised clustering with multi-scale feature reduction learning
    if (num_components_max - num_components_min) % num_components_step != 0:
        raise Exception('Number of componnets step should be divisible!')

    ## read the participant tsv
    df_participant = pd.read_csv(participant_tsv, sep='\t')

    ## create a temp file in the output_dir to save the intermediate tsv files
    if not os.path.exists(os.path.join(output_dir, 'intermediate')):
        os.makedirs(os.path.join(output_dir, 'intermediate'))

    ## make the final reuslts folder
    if not os.path.exists(os.path.join(output_dir, 'ensemble')):
        os.makedirs(os.path.join(output_dir, 'ensemble'))

    ## C lists
    C_list = list(range(num_components_min, num_components_max+num_components_step, num_components_step))
    ## first loop on different initial C.
    for i in range(len(C_list)):
        ## create a temp file in the output_dir to save the intermediate tsv files
        component_output_dir = os.path.join(output_dir, 'component_' + str(i))
        if not os.path.exists(component_output_dir):
            os.makedirs(component_output_dir)
        ### grab the output tsv of each C from opNMF
        opnmf_tsv = os.path.join(opnmf_dir, 'NMF', 'component_' + str(i), 'loading_coefficient.tsv')
        df_opnmf = pd.read_csv(opnmf_tsv, sep='\t')
        if df_participant.shape[0] != df_opnmf.shape[0]:
            raise Exception("The dimension of the participant_tsv and opNMF are not consistent!")
        ### make sure the row order is consistent with the participant_tsv
        df_opnmf = df_opnmf.set_index('participant_id')
        df_opnmf = df_opnmf.reindex(index=df_participant['participant_id'])
        df_opnmf = df_opnmf.reset_index()
        ## replace the path column in df_opnmf to be diagnosis, and save it to temp path for pyHYDRA classification
        diagnosis_list = list(df_participant['diagnosis'])
        df_opnmf["path"] = diagnosis_list
        df_opnmf.rename(columns={'path': 'diagnosis'}, inplace=True)
        ## save to tsv in a temporal folder
        opnmf_component_tsv = os.path.join(output_dir, 'intermediate', 'opnmf_component_' + str(i) + '.tsv')
        df_opnmf.to_csv(opnmf_component_tsv, index=False, sep='\t', encoding='utf-8')
        ## run the classification for K features and set the seed of CV split to 0, so that the splits are the same for
        ## different scales. Then finally, the ensemble voting can be done.
        classification_roi(opnmf_component_tsv, component_output_dir, cv_repetition=cv_repetition, cv_strategy=cv_strategy,
            class_weight_balanced=class_weight_balanced, n_threads=n_threads, verbose=verbose, seed=0)

    ## ensemble soft voting to determine the final classification results
    print('Computing the final classification with soft voting!\n')
    soft_majority_voting(output_dir, C_list)
    print('Finish...')
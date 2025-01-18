import numpy as np
import scipy
import os, pickle
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold, ShuffleSplit
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score
from packaging import version
import sklearn
if version.parse(sklearn.__version__) < version.parse("0.22.0"):
    from sklearn.metrics.ranking import roc_auc_score
else:
    from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from joblib import dump
import pandas as pd
from multiprocessing.pool import ThreadPool
import nibabel as nib
import sys
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock
from torch import nn
import math

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020"
__credits__ = ["Junhao Wen, Erdem Varol"]
__license__ = "See LICENSE file"
__version__ = "0.1.4"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def GLMcorrection(X_train, Y_train, covar_train, X_test, covar_test):
    """
    Eliminate the confound of covariate, such as age and sex, from the disease-based changes.
    Ref: "Age Correction in Dementia Matching to a Healthy Brain"
    :param X_train: array, training features
    :param Y_train: array, training labels
    :param covar_train: array, ttraining covariate data
    :param X_test: array, test labels
    :param covar_test: array, ttest covariate data
    :return: corrected training & test feature data
    """
    Yc = X_train[Y_train == -1]
    Xc = covar_train[Y_train == -1]
    Xc = np.concatenate((Xc, np.ones((Xc.shape[0], 1))), axis=1)
    beta = np.matmul(np.matmul(Yc.transpose(), Xc), np.linalg.inv(np.matmul(Xc.transpose(), Xc)))
    num_col = beta.shape[1]
    X_train_cor = (X_train.transpose() - np.matmul(beta[:, : num_col - 1], covar_train.transpose())).transpose()
    X_test_cor = (X_test.transpose() - np.matmul(beta[:, : num_col - 1], covar_test.transpose())).transpose()

    return X_train_cor, X_test_cor

def make_cv_partition(diagnosis, cv_strategy, output_dir, cv_repetition, seed=None):
    """
    Randomly generate the data split index for different CV strategy.

    :param diagnosis: the list for labels
    :param cv_repetition: the number of repetitions or folds
    :param output_dir: the output folder path
    :param cv_repetition: the number of repetitions for CV
    :param seed: random seed for sklearn split generator. Default is None
    :return:
    """
    unique = list(set(diagnosis))
    y = np.array(diagnosis)
    if len(unique) == 2: ### CV for classification and clustering
        if cv_strategy == 'k_fold':
            splits_indices_pickle = os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-fold.pkl')
            ## try to see if the shuffle has been done
            if os.path.isfile(splits_indices_pickle):
                splits_indices = pickle.load(open(splits_indices_pickle, 'rb'))
            else:
                splits = StratifiedKFold(n_splits=cv_repetition, random_state=seed)
                splits_indices = list(splits.split(np.zeros(len(y)), y))
        elif cv_strategy == 'hold_out':
            splits_indices_pickle = os.path.join(output_dir, 'data_split_stratified_' + str(cv_repetition) + '-holdout.pkl')
            ## try to see if the shuffle has been done
            if os.path.isfile(splits_indices_pickle):
                splits_indices = pickle.load(open(splits_indices_pickle, 'rb'))
            else:
                splits = StratifiedShuffleSplit(n_splits=cv_repetition, test_size=0.2, random_state=seed)
                splits_indices = list(splits.split(np.zeros(len(y)), y))
        else:
            raise Exception("this cross validation strategy has not been implemented!")
    elif len(unique) == 1:
        raise Exception("Diagnosis cannot be the same for all participants...")
    else: ### CV for regression, no need to be stratified
        if cv_strategy == 'k_fold':
            splits_indices_pickle = os.path.join(output_dir, 'data_split_' + str(cv_repetition) + '-fold.pkl')

            ## try to see if the shuffle has been done
            if os.path.isfile(splits_indices_pickle):
                splits_indices = pickle.load(open(splits_indices_pickle, 'rb'))
            else:
                splits = KFold(n_splits=cv_repetition, random_state=seed)
                splits_indices = list(splits.split(np.zeros(len(y)), y))
        elif cv_strategy == 'hold_out':
            splits_indices_pickle = os.path.join(output_dir, 'data_split_' + str(cv_repetition) + '-holdout.pkl')
            ## try to see if the shuffle has been done
            if os.path.isfile(splits_indices_pickle):
                splits_indices = pickle.load(open(splits_indices_pickle, 'rb'))
            else:
                splits = ShuffleSplit(n_splits=cv_repetition, test_size=0.2, random_state=seed)
                splits_indices = list(splits.split(np.zeros(len(y)), y))
        else:
            raise Exception("this cross validation strategy has not been implemented!")

    with open(splits_indices_pickle, 'wb') as s:
        pickle.dump(splits_indices, s)

    return splits_indices, splits_indices_pickle

def sample_dpp(evalue, evector, k=None):
    """
    sample a set Y from a dpp.  evalue, evector are a decomposed kernel, and k is (optionally) the size of the set to return
    :param evalue: eigenvalue
    :param evector: normalized eigenvector
    :param k: number of cluster
    :return:
    """
    if k == None:
        # choose eigenvectors randomly
        evalue = np.divide(evalue, (1 + evalue))
        evector = np.where(np.random.random(evalue.shape[0]) <= evalue)[0]
    else:
        v = sample_k(evalue, k) ## v here is a 1d array with size: k

    k = v.shape[0]
    v = v.astype(int)
    v = [i - 1 for i in v.tolist()]  ## due to the index difference between matlab & python, here, the element of v is for matlab
    V = evector[:, v]

    ## iterate
    y = np.zeros(k)
    for i in range(k, 0, -1):
        ## compute probabilities for each item
        P = np.sum(np.square(V), axis=1)
        P = P / np.sum(P)

        # choose a new item to include
        y[i-1] = np.where(np.random.rand(1) < np.cumsum(P))[0][0]
        y = y.astype(int)

        # choose a vector to eliminate
        j = np.where(V[y[i-1], :])[0][0]
        Vj = V[:, j]
        V = np.delete(V, j, 1)

        ## Update V
        if V.size == 0:
           pass
        else:
            V = np.subtract(V, np.multiply(Vj, (V[y[i-1], :] / Vj[y[i-1]])[:, np.newaxis]).transpose())  ## watch out the dimension here

        ## orthogonalize
        for m in range(i - 1):
            for n in range(m):
                V[:, m] = np.subtract(V[:, m], np.matmul(V[:, m].transpose(), V[:, n]) * V[:, n])

            V[:, m] = V[:, m] / np.linalg.norm(V[:, m])

    y = np.sort(y)

    return y

def sample_k(lambda_value, k):
    """
    Pick k lambdas according to p(S) \propto prod(lambda \in S)
    :param lambda_value: the corresponding eigenvalues
    :param k: the number of clusters
    :return:
    """

    ## compute elementary symmetric polynomials
    E = elem_sym_poly(lambda_value, k)

    ## ietrate over the lambda value
    num = lambda_value.shape[0]
    remaining = k
    S = np.zeros(k)
    while remaining > 0:
        #compute marginal of num given that we choose remaining values from 0:num-1
        if num == remaining:
            marg = 1
        else:
            marg = lambda_value[num-1] * E[remaining-1, num-1] / E[remaining, num]

        # sample marginal
        if np.random.rand(1) < marg:
            S[remaining-1] = num
            remaining = remaining - 1
        num = num - 1
    return S

def elem_sym_poly(lambda_value, k):
    """
    given a vector of lambdas and a maximum size k, determine the value of
    the elementary symmetric polynomials:
    E(l+1,n+1) = sum_{J \subseteq 1..n,|J| = l} prod_{i \in J} lambda(i)
    :param lambda_value: the corresponding eigenvalues
    :param k: number of clusters
    :return:
    """
    N = lambda_value.shape[0]
    E = np.zeros((k + 1, N + 1))
    E[0, :] = 1

    for i in range(1, k+1):
        for j in range(1, N+1):
            E[i, j] = E[i, j - 1] + lambda_value[j-1] * E[i - 1, j - 1]

    return E

def proportional_assign(l, d):
    """
    Proportional assignment based on margin
    :param l: int
    :param d: int
    :return:
    """
    np.seterr(divide='ignore', invalid='ignore')
    invL = np.divide(1, l)
    idx = np.isinf(invL)
    invL[idx] = d[idx]

    for i in range(l.shape[0]):
        pos = np.where(invL[i, :] > 0)[0]
        neg = np.where(invL[i, :] < 0)[0]
        if pos.size != 0:
            invL[i, neg] = 0
        else:
            invL[i, :] = np.divide(invL[i, :], np.amin(invL[i, :]))
            invL[i, invL[i, :] < 1] = 0

    S = np.multiply(invL, np.divide(1, np.sum(invL, axis=1))[:, np.newaxis])

    return S

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Check if the numpy array is symmetric or not
    Args:
        a:
        rtol:
        atol:

    Returns:

    """
    result = np.allclose(a, a.T, rtol=rtol, atol=atol)
    return result

def consensus_clustering(clustering_results, k):
    """
    This function performs consensus clustering on a co-occurence matrix
    :param clustering_results: an array containing all the clustering results across different iterations, in order to
    perform
    :param k: number of clusters
    :return:
    """

    num_pt = clustering_results.shape[0]
    cooccurence_matrix = np.zeros((num_pt, num_pt))

    for i in range(num_pt - 1):
        for j in range(i + 1, num_pt):
            cooccurence_matrix[i, j] = sum(clustering_results[i, :] == clustering_results[j, :])

    cooccurence_matrix = np.add(cooccurence_matrix, cooccurence_matrix.transpose())
    ## here is to compute the Laplacian matrix
    Laplacian = np.subtract(np.diag(np.sum(cooccurence_matrix, axis=1)), cooccurence_matrix)

    Laplacian_norm = np.subtract(np.eye(num_pt), np.matmul(np.matmul(np.diag(1 / np.sqrt(np.sum(cooccurence_matrix, axis=1))), cooccurence_matrix), np.diag(1 / np.sqrt(np.sum(cooccurence_matrix, axis=1)))))
    ## replace the nan with 0
    Laplacian_norm = np.nan_to_num(Laplacian_norm)

    ## check if the Laplacian norm is symmetric or not, because matlab eig function will automatically check this, but not in numpy or scipy
    if check_symmetric(Laplacian_norm):
        ## extract the eigen value and vector
        ## matlab eig equivalence is eigh, not eig from numpy or scipy, see this post: https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
        ## Note, the eigenvector is not unique, thus the matlab and python eigenvector may be different, but this will not affect the results.
        evalue, evector = scipy.linalg.eigh(Laplacian_norm)
    else:
        # evalue, evector = np.linalg.eig(Laplacian_norm)
        raise Exception("The Laplacian matrix should be symmetric here...")

    ## check if the eigen vector is complex
    if np.any(np.iscomplex(evector)):
        evalue, evector = scipy.linalg.eigh(Laplacian)

    ## create the kmean algorithm with sklearn
    kmeans = KMeans(n_clusters=k, n_init=20).fit(evector.real[:, 0: k])
    final_predict = kmeans.labels_

    return final_predict

def cv_cluster_stability(result, k):
    """
    To compute the adjusted rand index across different pair of 2 folds cross CV
    Args:
        result: results containing all clustering assignment across all repetitions for CV
        k: number of clusters
    Returns:

    """
    num_pair = 0
    aris = []
    if k == 1:
        adjusted_rand_index = 0  ## note, here, we manually set it to be 0, because it does not make sense when k==1.
    else:
        for i in range(result.shape[1] - 1):
            for j in range(i+1, result.shape[1]):
                num_pair += 1
                non_zero_index = np.all(result[:, [i, j]], axis=1)
                pair_result = result[:, [i, j]][non_zero_index]
                ari = adjusted_rand_score(pair_result[:, 0], pair_result[:, 1])
                aris.append(ari)

        adjusted_rand_index = np.mean(np.asarray(aris))

    return adjusted_rand_index

def evaluate_prediction(y, y_hat):
    """
    Caculate the performance for classification.
    Note: positive value is 1 (PT) and negative is 0 (CN).
    Args:
        y: ground truth for label
        y_hat: predicted value for label

    Returns:

    """
    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0

    tp = []
    tn = []
    fp = []
    fn = []

    for i in range(len(y)):
        if y[i] == 1:
            if y_hat[i] == 1:
                true_positive += 1
                tp.append(i)
            else:
                false_negative += 1
                fn.append(i)
        else:  # -1
            if y_hat[i] == 0:
                true_negative += 1
                tn.append(i)
            else:
                false_positive += 1
                fp.append(i)

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    if (true_positive + false_negative) != 0:
        sensitivity = true_positive / (true_positive + false_negative)
    else:
        sensitivity = 0.0

    if (false_positive + true_negative) != 0:
        specificity = true_negative / (false_positive + true_negative)
    else:
        specificity = 0.0

    if (true_positive + false_positive) != 0:
        ppv = true_positive / (true_positive + false_positive)
    else:
        ppv = 0.0

    if (true_negative + false_negative) != 0:
        npv = true_negative / (true_negative + false_negative)
    else:
        npv = 0.0

    balanced_accuracy = (sensitivity + specificity) / 2

    results = {'accuracy': accuracy,
               'balanced_accuracy': balanced_accuracy,
               'sensitivity': sensitivity,
               'specificity': specificity,
               'ppv': ppv,
               'npv': npv,
               'confusion_matrix': {'tp': len(tp), 'tn': len(tn), 'fp': len(fp), 'fn': len(fn)}
               }

    return results

def hydra_solver_svm(num_repetition, X, y, k, output_dir, num_consensus, num_iteration, tol, balanced, predefined_c, weight_initialization_type, n_threads, save_models, verbose):
    """
    This is the main function of HYDRA, which find the convex polytope using a supervised classification fashion.
    Args:
        num_repetition: int, number of repetitions for CV
        X: input matrix for features
        y: input for group label
        k: number of clusters
        output_dir: the path for output
        num_consensus: int, number of runs for consensus clustering
        num_iteration: int, number of maximum iterations for running HYDRA
        tol: float, tolerance value for model convergence
        balanced: if sample imbalance should be considered during model optimization
        predefined_c: predefined c for SVM for clustering
        weight_initialization_type: the type of initialization of the weighted sample matrix
        n_threads: number of threads used
        save_models: if save all models during CV
        verbose: if output is verbose

    Returns:

    """
    censensus_assignment = np.zeros((y[y == 1].shape[0], num_consensus)) ## only consider the PTs

    index_pt = np.where(y == 1)[0]  # index for PTs
    index_cn = np.where(y == -1)[0]  # index for CNs

    for i in range(num_consensus):
        weight_sample = np.ones((y.shape[0], k)) / k
        ## depending on the weight initialization strategy, random hyperplanes were initialized with maximum diversity to constitute the convex polytope
        weight_sample_pt = hydra_init_weight(X, y, k, index_pt, index_cn, weight_initialization_type)
        weight_sample[index_pt] = weight_sample_pt  ## only replace the sample weight of the PT group
        ## cluster assignment is based on this svm scores across different SVM/hyperplanes
        svm_scores = np.zeros((weight_sample.shape[0], weight_sample.shape[1]))
        update_weights_pool = ThreadPool(processes=n_threads)

        for j in range(num_iteration):
            for m in range(k):
                sample_weight = np.ascontiguousarray(weight_sample[:, m])
                if np.count_nonzero(sample_weight[index_pt]) == 0:
                    if verbose == True:
                        print("Cluster dropped, meaning that all PT has been assigned to one single hyperplane in iteration: %d" % (j-1))
                        print("Be careful, this could cause problem because of the ill-posed solution. Especially when k==2")
                else:
                    results = update_weights_pool.apply_async(launch_svc, args=(X, y, predefined_c, sample_weight, balanced))
                    weight_coef = results.get()[0]
                    intesept = results.get()[1]
                    ## Apply the data again the trained model to get the final SVM scores
                    svm_scores[:, m] = (np.matmul(weight_coef, X.transpose()) + intesept).transpose().squeeze()

            cluster_index = np.argmax(svm_scores[index_pt], axis=1)

            ## decide the converge of the polytope based on the toleration
            weight_sample_hold = weight_sample.copy()
            # after each iteration, first set the weight of patient rows to be 0
            weight_sample[index_pt, :] = 0
            # then set the pt's weight to be 1 for the assigned hyperplane
            for n in range(len(index_pt)):
                weight_sample[index_pt[n], cluster_index[n]] = 1

            ## check the loss comparted to the tolorence for stopping criteria
            loss = np.linalg.norm(np.subtract(weight_sample, weight_sample_hold), ord='fro')
            if verbose == True:
                print("The loss is: %f" % loss)
            if loss < tol:
                if verbose == True:
                    print("The polytope has been converged for iteration %d in finding %d clusters in consensus running: %d" % (j, k, i))
                break
        update_weights_pool.close()
        update_weights_pool.join()

        ## update the cluster index for the consensus clustering
        censensus_assignment[:, i] = cluster_index + 1

    ## do censensus clustering
    final_predict = consensus_clustering(censensus_assignment.astype(int), k)

    ## after deciding the final convex polytope, we refit the training data once to save the best model
    weight_sample_final = np.zeros((y.shape[0], k))
    ## change the weight of PTs to be 1, CNs to be 1/k

    # then set the pt's weight to be 1 for the assigned hyperplane
    for n in range(len(index_pt)):
        weight_sample_final[index_pt[n], final_predict[n]] = 1

    weight_sample_final[index_cn] = 1 / k
    update_weights_pool_final = ThreadPool(processes=n_threads)
    ## create the final polytope by applying all weighted subjects
    for o in range(k):
        sample_weight = np.ascontiguousarray(weight_sample_final[:, o])
        results = update_weights_pool_final.apply_async(launch_svc, args=(X, y, predefined_c, sample_weight, balanced))
        
        if not os.path.exists(os.path.join(output_dir, str(k) + '_clusters', 'models')):
            os.makedirs(os.path.join(output_dir, str(k) + '_clusters', 'models'))

        ## save the final model for the k SVMs/hyperplanes
        if save_models == True:
            if not os.path.exists(os.path.join(output_dir, str(k) + '_clusters', 'models')):
                os.makedirs(os.path.join(output_dir, str(k) + '_clusters', 'models'))

            dump(results.get()[2], os.path.join(output_dir,  str(k) + '_clusters', 'models', 'svm-' + str(o) + '_cv_' + str(num_repetition) + '.joblib'))
        else: 
        ## only save the last repetition
            if not os.path.isfile(os.path.join(output_dir,  str(k) + '_clusters', 'models', 'svm-' + str(o) + '_last_repetition.joblib')):
                dump(results.get()[2], os.path.join(output_dir,  str(k) + '_clusters', 'models', 'svm-' + str(o) + '_last_repetition.joblib'))    
    update_weights_pool_final.close()
    update_weights_pool_final.join()

    y[index_pt] = final_predict + 1

    if not os.path.exists(os.path.join(output_dir, str(k) + '_clusters', 'tsv')):
        os.makedirs(os.path.join(output_dir, str(k) + '_clusters', 'tsv'))

    ### save results also in tsv file for each repetition
    ## save the assigned weight for each subject across k-fold
    columns = ['hyperplane' + str(i) for i in range(k)]
    weight_sample_df = pd.DataFrame(weight_sample_final, columns=columns)
    weight_sample_df.to_csv(os.path.join(output_dir, str(k) + '_clusters', 'tsv', 'weight_sample_cv_' + str(num_repetition) + '.tsv'), index=False, sep='\t', encoding='utf-8')

    ## save the final_predict_all
    columns = ['y_hat']
    y_hat_df = pd.DataFrame(y, columns=columns)
    y_hat_df.to_csv(os.path.join(output_dir, str(k) + '_clusters', 'tsv', 'y_hat_cv_' + str(num_repetition) + '.tsv'), index=False, sep='\t', encoding='utf-8')

    ## save the pt index
    columns = ['pt_index']
    pt_df = pd.DataFrame(index_pt, columns=columns)
    pt_df.to_csv(os.path.join(output_dir, str(k) + '_clusters', 'tsv', 'pt_index_cv_' + str(num_repetition) + '.tsv'), index=False, sep='\t', encoding='utf-8')

    return y

def hydra_init_weight(X, y, k, index_pt, index_cn, weight_initialization_type):
    """
    Function performs initialization for the polytope of mlni
    Args:
        X: the input features
        y: the label
        k: number of predefined clusters
        index_pt: list, the index for patient subjects
        index_cn: list, the index for control subjects
        weight_initialization_type: the type of chosen initialization method
    Returns:

    """
    if weight_initialization_type == "DPP":  ##
        num_subject = y.shape[0]
        W = np.zeros((num_subject, X.shape[1]))
        for j in range(num_subject):
            ipt = np.random.randint(index_pt.shape[0])
            icn = np.random.randint(index_cn.shape[0])
            W[j, :] = X[index_pt[ipt], :] - X[index_cn[icn], :]

        KW = np.matmul(W, W.transpose())
        KW = np.divide(KW, np.sqrt(np.multiply(np.diag(KW)[:, np.newaxis], np.diag(KW)[:, np.newaxis].transpose())))
        evalue, evector = np.linalg.eig(KW)
        Widx = sample_dpp(np.real(evalue), np.real(evector), k)
        prob = np.zeros((len(index_pt), k))  # only consider the PTs

        for i in range(k):
            prob[:, i] = np.matmul(np.multiply(X[index_pt, :], np.divide(1, np.linalg.norm(X[index_pt, :], axis=1))[:, np.newaxis]), W[Widx[i], :].transpose())

        l = np.minimum(prob - 1, 0)
        d = prob - 1
        S = proportional_assign(l, d)

    elif weight_initialization_type == "random_hyperplane":
        print("TODO")

    elif weight_initialization_type == "random_assign":
        S = random_init_dirichlet(k, len(index_pt))

    elif weight_initialization_type == "k_means":
        print("TODO")
    else:
        raise Exception("Not implemented yet!")

    return S

def launch_svc(X, y, predefined_c, sample_weight, balanced):
    """
    Lauch svc classifier of sklearn
    Args:
        X: input matrix for features
        y: input matrix for label
        predefined_c: predefined C
        sample_weight: the weighted sample matrix
        balanced:

    Returns:

    """
    if not balanced:
        model = SVC(kernel='linear', C=predefined_c)
    else:
        model = SVC(kernel='linear', C=predefined_c, class_weight='balanced')

    ## fit the different SVM/hyperplanes
    model.fit(X, y, sample_weight=sample_weight)

    weight_coef = model.coef_
    intesept = model.intercept_

    return weight_coef, intesept, model

def random_init_dirichlet(k, num_pt):
    """
    a sample from a dirichlet distribution
    :param k: number of clusters
    :param num_pt: number of PT
    :return:
    """
    a = np.ones(k)
    s = np.random.dirichlet(a, num_pt)

    return s

def load_data(image_list, mask=True):
    """
    Load the image data with/without mask
    Args:
        image_list: list, containing all image path
        mask: if mask out the background or not

    Returns:

    """
    data = None
    shape = None
    data_mask = None
    first = True

    for i in range(len(image_list)):
        subj = nib.load(image_list[i])
        subj_data = np.nan_to_num(subj.get_data(caching='unchanged'))
        shape = subj_data.shape
        ## change dtype to float32 to save memory, in case number of images is huge, consider downsample the image resolution.
        subj_data = subj_data.flatten().astype('float32')
        subj_data[subj_data < 0] = 0

        # Memory allocation for ndarray containing all data to avoid copying the array for each new subject
        if first:
            data = np.ndarray(shape=(len(image_list), subj_data.shape[0]), dtype=float, order='C')
            first = False

        data[i, :] = subj_data

    if mask:
        data_mask = (data != 0).sum(axis=0) != 0
        data = data[:, data_mask]

    return data, shape, data_mask

def revert_mask(weights, mask, shape):
    """
    Args:
        weights:
        mask:
        shape:
    Returns:
    """

    z = np.zeros(np.prod(shape))
    z[mask] = weights

    new_weights = np.reshape(z, shape)

    return new_weights

def gram_matrix_linear(data):
    return np.dot(data, data.transpose())


def soft_voting(output_dir, C_list, cv_repetition):
    """
    This is to perform soft majority voting for the final classification across different scales of opNMF.
    Note that soft voting is only recommended if the classifiers are well-calibrated, though SVM is not in this case.
    Args:
        output_dir:
        C_list:

    Returns:

    """
    ### take the mean results
    resutls_repetitions = []
    ### mkdir the iteration folder
    for i in range(cv_repetition):
        iteration_dir = os.path.join(output_dir, 'ensemble', 'iteration-' + str(i))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)
        ## read the test_subjects.tsv from each component
        for j in C_list:
            test_results_tsv = os.path.join(output_dir, 'component_' + str(j), 'classification', 'iteration-' + str(i),
                                            'test_subjects.tsv')
            df = pd.read_csv(test_results_tsv, sep='\t')
            if j == C_list[0]:
                df_final = df.copy()
            else:
                ### concatenate the new df to previous df
                df_final = pd.concat([df_final, df['proba_test_index1']], axis=1)
                ## rename the proba_test_index1
            df_final.rename({'proba_test_index1': 'proba_test_index1_C' + str(j)}, axis=1, inplace=True)

        ### perform soft voting to decide the final probability
        columns_to_mean = ['proba_test_index1_C' + str(k) for k in C_list]
        df_final['proba_test_index1'] = df_final[columns_to_mean].mean(axis=1)
        df_final.drop(columns=columns_to_mean, inplace=True)
        ## decide the finla y_hat
        df_final['y_hat_ensemble'] = (df_final['proba_test_index1'] > 0.5).astype(int)
        del df_final['y_hat']
        columns_to_reorder = ['iteration', 'y', 'y_hat_ensemble', 'subject_index', 'proba_test_index1']
        df_final = df_final[columns_to_reorder]
        auc = roc_auc_score(df_final['y'].to_numpy(), df_final['proba_test_index1'].to_numpy())
        results_dic = evaluate_prediction(list(df_final['y'].to_numpy()), list(df_final['y_hat_ensemble'].to_numpy()))
        ## calculate the results.tsv
        results_df = pd.DataFrame({'balanced_accuracy': results_dic['balanced_accuracy'],
                                   'auc': auc,
                                   'accuracy': results_dic['accuracy'],
                                   'sensitivity': results_dic['sensitivity'],
                                   'specificity': results_dic['specificity'],
                                   'ppv': results_dic['ppv'],
                                   'npv': results_dic['npv']}, index=['i', ])
        results_df.to_csv(os.path.join(iteration_dir, 'results.tsv'), index=False, sep='\t', encoding='utf-8')
        resutls_repetitions.append(results_df)

    all_results = pd.concat(resutls_repetitions)
    all_results.to_csv(os.path.join(output_dir, 'ensemble', 'results.tsv'), index=False, sep='\t', encoding='utf-8')
    mean_results = pd.DataFrame(all_results.apply(np.nanmean).to_dict(), columns=all_results.columns, index=[0, ])
    mean_results.to_csv(os.path.join(output_dir, 'ensemble', 'mean_results.tsv'), index=False, sep='\t',
                        encoding='utf-8')

    print("Mean results of the classification after voting:")
    print("Balanced accuracy: %s" % (mean_results['balanced_accuracy'].to_string(index=False)))
    print("specificity: %s" % (mean_results['specificity'].to_string(index=False)))
    print("sensitivity: %s" % (mean_results['sensitivity'].to_string(index=False)))
    print("auc: %s" % (mean_results['auc'].to_string(index=False)))

    return None

def weighted_soft_voting(output_dir, C_list, cv_repetition):
    """
    This is to perform weighted soft majority voting for the final classification across different scales of opNMF.
    Note that soft voting is only recommended if the classifiers are well-calibrated, though SVM is not in this case.
    Args:
        output_dir:
        C_list:

    Returns:

    """
    ### take the mean results
    resutls_repetitions = []

    ### calculate the weighted based on the number of components
    C_list_weight = [i / sum(C_list) for i in C_list]

    ### mkdir the iteration folder
    for i in range(cv_repetition):
        iteration_dir = os.path.join(output_dir, 'ensemble', 'iteration-' + str(i))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)
        ## read the test_subjects.tsv from each component
        for j in C_list:
            test_results_tsv = os.path.join(output_dir, 'component_' + str(j), 'classification', 'iteration-' + str(i),
                                            'test_subjects.tsv')
            df = pd.read_csv(test_results_tsv, sep='\t')
            if j == C_list[0]:
                df_final = df.copy()
                ## calculate the proba for index 0
                prob_test_index0 = 1 - df['proba_test_index1']
                df_final['proba_test_index0'] = prob_test_index0
            else:
                ### concatenate the new df to previous df
                df_final = pd.concat([df_final, df['proba_test_index1']], axis=1)
                ## calculate the proba for index 0
                prob_test_index0 = 1 - df['proba_test_index1']
                df_final['proba_test_index0'] = prob_test_index0
                ## rename the proba_test_index1
            df_final.rename({'proba_test_index1': 'proba_test_index1_C' + str(j)}, axis=1, inplace=True)
            df_final.rename({'proba_test_index0': 'proba_test_index0_C' + str(j)}, axis=1, inplace=True)

        ### perform weighted soft voting to decide the final probability
        columns_to_mean_pos = ['proba_test_index1_C' + str(k) for k in C_list]
        for k in range(len(columns_to_mean_pos)):
            df_final[columns_to_mean_pos[k]] = df_final[columns_to_mean_pos[k]] * C_list_weight[k]
        df_final['proba_test_index1'] = df_final[columns_to_mean_pos].sum(axis=1)
        df_final.drop(columns=columns_to_mean_pos, inplace=True)
        columns_to_mean_neg = ['proba_test_index0_C' + str(k) for k in C_list]
        for k in range(len(columns_to_mean_neg)):
            df_final[columns_to_mean_neg[k]] = df_final[columns_to_mean_neg[k]] * C_list_weight[k]
        df_final['proba_test_index0'] = df_final[columns_to_mean_neg].sum(axis=1)
        df_final.drop(columns=columns_to_mean_neg, inplace=True)

        ## decide the finla y_hat
        df_final['y_hat_ensemble'] = (df_final['proba_test_index1'] > df_final['proba_test_index0']).astype(int)
        del df_final['y_hat']
        columns_to_reorder = ['iteration', 'y', 'y_hat_ensemble', 'subject_index', 'proba_test_index1']
        df_final = df_final[columns_to_reorder]
        auc = roc_auc_score(df_final['y'].to_numpy(), df_final['proba_test_index1'].to_numpy())
        results_dic = evaluate_prediction(list(df_final['y'].to_numpy()), list(df_final['y_hat_ensemble'].to_numpy()))
        ## calculate the results.tsv
        results_df = pd.DataFrame({'balanced_accuracy': results_dic['balanced_accuracy'],
                                   'auc': auc,
                                   'accuracy': results_dic['accuracy'],
                                   'sensitivity': results_dic['sensitivity'],
                                   'specificity': results_dic['specificity'],
                                   'ppv': results_dic['ppv'],
                                   'npv': results_dic['npv']}, index=['i', ])
        results_df.to_csv(os.path.join(iteration_dir, 'results.tsv'), index=False, sep='\t', encoding='utf-8')
        resutls_repetitions.append(results_df)

    all_results = pd.concat(resutls_repetitions)
    all_results.to_csv(os.path.join(output_dir, 'ensemble', 'results.tsv'), index=False, sep='\t', encoding='utf-8')
    mean_results = pd.DataFrame(all_results.apply(np.nanmean).to_dict(), columns=all_results.columns, index=[0, ])
    mean_results.to_csv(os.path.join(output_dir, 'ensemble', 'mean_results.tsv'), index=False, sep='\t',
                        encoding='utf-8')

    print("Mean results of the classification after voting:")
    print("Balanced accuracy: %s" % (mean_results['balanced_accuracy'].to_string(index=False)))
    print("specificity: %s" % (mean_results['specificity'].to_string(index=False)))
    print("sensitivity: %s" % (mean_results['sensitivity'].to_string(index=False)))
    print("auc: %s" % (mean_results['auc'].to_string(index=False)))

    return None

def hard_majority_voting(output_dir, C_list, cv_repetition):
    """
    This is to perform hard majority voting for the final classification across different scales of opNMF.
    Args:
        output_dir:
        C_list:

    Returns:

    """
    ### take the mean results
    resutls_repetitions = []
    ### mkdir the iteration folder
    for i in range(cv_repetition):
        iteration_dir = os.path.join(output_dir, 'ensemble', 'iteration-' + str(i))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)
        ## read the test_subjects.tsv from each component
        for j in C_list:
            test_results_tsv = os.path.join(output_dir, 'component_' + str(j), 'classification', 'iteration-' + str(i), 'test_subjects.tsv')
            df = pd.read_csv(test_results_tsv, sep='\t')
            if j == C_list[0]:
                df_final = df.copy()
            else:
                ### concatenate the new df to previous df
                df_final = pd.concat([df_final, df['y_hat']], axis=1)
                ## rename the proba_test_index1
            df_final.rename({'y_hat': 'y_hat_C' + str(j)}, axis=1, inplace=True)

        ### perform hard voting to decide the final probability
        columns_to_mode = ['y_hat_C' + str(k) for k in C_list]
        df_final['y_hat_ensemble'] = df_final[columns_to_mode].mode(axis=1).iloc[:, 0]
        df_final.drop(columns=columns_to_mode, inplace=True)
        columns_to_reorder = ['iteration', 'y', 'y_hat_ensemble', 'subject_index']
        df_final = df_final[columns_to_reorder]
        results_dic = evaluate_prediction(list(df_final['y'].to_numpy()), list(df_final['y_hat_ensemble'].to_numpy()))
        ## calculate the results.tsv
        results_df = pd.DataFrame({'balanced_accuracy': results_dic['balanced_accuracy'],
                                   'accuracy': results_dic['accuracy'],
                                   'sensitivity': results_dic['sensitivity'],
                                   'specificity': results_dic['specificity'],
                                   'ppv': results_dic['ppv'],
                                   'npv': results_dic['npv']}, index=['i', ])
        results_df.to_csv(os.path.join(iteration_dir, 'results.tsv'), index=False, sep='\t', encoding='utf-8')
        resutls_repetitions.append(results_df)

    all_results = pd.concat(resutls_repetitions)
    all_results.to_csv(os.path.join(output_dir, 'ensemble', 'results.tsv'), index=False, sep='\t', encoding='utf-8')
    mean_results = pd.DataFrame(all_results.apply(np.nanmean).to_dict(), columns=all_results.columns, index=[0, ])
    mean_results.to_csv(os.path.join(output_dir, 'ensemble', 'mean_results.tsv'), index=False, sep='\t', encoding='utf-8')

    print("Mean results of the classification after voting:")
    print("Balanced accuracy: %s" % (mean_results['balanced_accuracy'].to_string(index=False)))
    print("specificity: %s" % (mean_results['specificity'].to_string(index=False)))
    print("sensitivity: %s" % (mean_results['sensitivity'].to_string(index=False)))

    return None

def consensus_classification(classification_results, k, ground_truth):
    """
    This function performs consensus classification based on a co-occurence matrix
    :param classification_results: an array containing all the clustering results across different iterations, in order to
    perform
    :param k: number of clusters
    :return:
    """

    num_pt = classification_results.shape[0]
    cooccurence_matrix = np.zeros((num_pt, num_pt))

    for i in range(num_pt - 1):
        for j in range(i + 1, num_pt):
            cooccurence_matrix[i, j] = sum(classification_results[i, :] == classification_results[j, :])

    cooccurence_matrix = np.add(cooccurence_matrix, cooccurence_matrix.transpose())
    ## here is to compute the Laplacian matrix
    Laplacian = np.subtract(np.diag(np.sum(cooccurence_matrix, axis=1)), cooccurence_matrix)

    Laplacian_norm = np.subtract(np.eye(num_pt), np.matmul(np.matmul(np.diag(1 / np.sqrt(np.sum(cooccurence_matrix, axis=1))), cooccurence_matrix), np.diag(1 / np.sqrt(np.sum(cooccurence_matrix, axis=1)))))
    ## replace the nan with 0
    Laplacian_norm = np.nan_to_num(Laplacian_norm)

    ## check if the Laplacian norm is symmetric or not, because matlab eig function will automatically check this, but not in numpy or scipy
    if check_symmetric(Laplacian_norm):
        ## extract the eigen value and vector
        ## matlab eig equivalence is eigh, not eig from numpy or scipy, see this post: https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
        ## Note, the eigenvector is not unique, thus the matlab and python eigenvector may be different, but this will not affect the results.
        evalue, evector = scipy.linalg.eigh(Laplacian_norm)
    else:
        # evalue, evector = np.linalg.eig(Laplacian_norm)
        raise Exception("The Laplacian matrix should be symmetric here...")

    ## check if the eigen vector is complex
    if np.any(np.iscomplex(evector)):
        evalue, evector = scipy.linalg.eigh(Laplacian)

    ## create the kmean algorithm with sklearn
    kmeans = KMeans(n_clusters=k, n_init=20).fit(evector.real[:, 0: k])
    final_predict = kmeans.labels_

    ### since the order of the clustering does not make any sense, we will decide the final predict based on permutation of the final_predict.
    ## WARN: we assume that the original classification perform bettern than random chance, i.e., accuracy > 0.5
    acc_1 = accuracy_score(ground_truth, final_predict)
    ## swap the cluster labels
    final_predict_swapped = []
    for i in list(final_predict):
        if i == 0:
            final_predict_swapped.append(1)
        elif i ==1:
            final_predict_swapped.append(0)
        else:
            raise Exception("Something wrong here!")

    if acc_1 > 0.5:
        pass
    else:
        final_predict = final_predict_swapped

    return final_predict

def consensus_voting(output_dir, C_list, cv_repetition):
    """
    This is to perform consensus voting based on Spectral clustering.
    Args:
        output_dir:
        C_list:

    Returns:

    """
    ### take the mean results
    resutls_repetitions = []
    ### mkdir the iteration folder
    for i in range(cv_repetition):
        iteration_dir = os.path.join(output_dir, 'ensemble', 'iteration-' + str(i))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)
        ## read the test_subjects.tsv from each component
        for j in C_list:
            test_results_tsv = os.path.join(output_dir, 'component_' + str(j), 'classification', 'iteration-' + str(i), 'test_subjects.tsv')
            df = pd.read_csv(test_results_tsv, sep='\t')
            if j == C_list[0]:
                df_final = df.copy()
            else:
                ### concatenate the new df to previous df
                df_final = pd.concat([df_final, df['y_hat']], axis=1)
                ## rename the proba_test_index1
            df_final.rename({'y_hat': 'y_hat_C' + str(j)}, axis=1, inplace=True)

        ### perform consensus voting to decide the final probability
        columns_to_consensus = ['y_hat_C' + str(k) for k in C_list]
        classification_results = df_final[columns_to_consensus].to_numpy()
        ground_truth = df_final['y'].to_numpy()
        final_predict = consensus_classification(classification_results, 2, ground_truth)
        df_final['y_hat_ensemble'] = final_predict

        df_final.drop(columns=columns_to_consensus, inplace=True)
        columns_to_reorder = ['iteration', 'y', 'y_hat_ensemble', 'subject_index']
        df_final = df_final[columns_to_reorder]
        results_dic = evaluate_prediction(list(df_final['y'].to_numpy()), list(df_final['y_hat_ensemble'].to_numpy()))
        ## calculate the results.tsv
        results_df = pd.DataFrame({'balanced_accuracy': results_dic['balanced_accuracy'],
                                   'accuracy': results_dic['accuracy'],
                                   'sensitivity': results_dic['sensitivity'],
                                   'specificity': results_dic['specificity'],
                                   'ppv': results_dic['ppv'],
                                   'npv': results_dic['npv']}, index=['i', ])
        results_df.to_csv(os.path.join(iteration_dir, 'results.tsv'), index=False, sep='\t', encoding='utf-8')
        resutls_repetitions.append(results_df)

    all_results = pd.concat(resutls_repetitions)
    all_results.to_csv(os.path.join(output_dir, 'ensemble', 'results.tsv'), index=False, sep='\t', encoding='utf-8')
    mean_results = pd.DataFrame(all_results.apply(np.nanmean).to_dict(), columns=all_results.columns, index=[0, ])
    mean_results.to_csv(os.path.join(output_dir, 'ensemble', 'mean_results.tsv'), index=False, sep='\t', encoding='utf-8')

    print("Mean results of the classification after voting:")
    print("Balanced accuracy: %s" % (mean_results['balanced_accuracy'].to_string(index=False)))
    print("specificity: %s" % (mean_results['specificity'].to_string(index=False)))
    print("sensitivity: %s" % (mean_results['sensitivity'].to_string(index=False)))

    return None

def prepare_opnmf_tsv_voting(output_dir, opnmf_dir, i, df_participant):
    """
    Prepare the tsv from opNMF for pyHYDRA classification voting.
    Args:
        output_dir:
        opnmf_dir:
        i:
        df_participant:

    Returns:

    """
    ## create a temp file in the output_dir to save the intermediate tsv files
    component_output_dir = os.path.join(output_dir, 'component_' + str(i))
    if not os.path.exists(component_output_dir):
        os.makedirs(component_output_dir)
    ### grab the output tsv of each C from opNMF
    opnmf_tsv = os.path.join(opnmf_dir, 'NMF', 'component_' + str(i), 'atlas_components_signal.tsv')
    df_opnmf = pd.read_csv(opnmf_tsv, sep='\t')
    ### only take the rows in opnmf_tsv which are in common in participant_tsv
    df_opnmf = df_opnmf.loc[df_opnmf['participant_id'].isin(df_participant['participant_id'])]
    ## now check the dimensions
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

    return component_output_dir, opnmf_component_tsv

def voting_system(voting_method, output_dir, components_list, cv_repetition):
    """
    Perform voting based on different methods
    Args:
        voting_method:
        output_dir:
        components_list:
        cv_repetition:

    Returns:

    """
    if voting_method == "soft_voting":
        print('Computing the final classification with soft voting!\n')
        soft_voting(output_dir, components_list, cv_repetition)
    elif voting_method == "weighted_soft_voting":
        print('Computing the final classification with weighted soft voting!\n')
        weighted_soft_voting(output_dir, components_list, cv_repetition)
    elif voting_method == "hard_voting":
        print('Computing the final classification with hard voting!\n')
        hard_majority_voting(output_dir, components_list, cv_repetition)
    elif voting_method == "consensus_voting":
        print('Computing the final classification with consensus voting!\n')
        consensus_voting(output_dir, components_list, cv_repetition)
    else:
        raise Exception("Method not implemented yet！")

def time_bar(i,num):
    sys.stdout.write('\r')
    progress = (i+1)*1.0/num
    prog_int = int(progress*50)
    sys.stdout.write('\t\t[%s%s] %.2f%%' % ('='*prog_int,' '*(50-prog_int), progress*100))
    sys.stdout.flush()

###### Neural Network for classification, e.g., age prediction
class neural_network_classification_3LinerLayers(nn.Module):
    def __init__(self, input_dim):
        super(neural_network_classification_3LinerLayers, self).__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.hidden_layer1 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        out = self.output_layer(out)
        return out

class neural_network_classification_5LinerLayers(nn.Module):
    def __init__(self, input_dim):
        super(neural_network_classification_5LinerLayers, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.Dropout2d(p=0.5),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout2d(p=0.5),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Dropout2d(p=0.5),
            nn.ReLU(),

            nn.Linear(16, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.features(x)
        return out

###### Neural Network for Regression, e.g., age prediction
class neural_network_regression_3LinerLayers(nn.Module):
    def __init__(self, input_dim):
        super(neural_network_regression_3LinerLayers, self).__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.hidden_layer1 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        out = self.output_layer(out)
        return out

class neural_network_regression_5LinerLayers(nn.Module):
    def __init__(self, input_dim):
        super(neural_network_regression_5LinerLayers, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.features(x)
        return out


##### Resnet adapted to regerssion
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}

class DatasetShuffler(Dataset):
    """
    This class reads the batch size of samples

    Return: a Pytorch Dataset objective
    """

    def __init__(self, data_tensor_x, data_tensor_y, df_header, transformations=False):
        """
        Args:
            data_tensor (string): data tensor.
            transformations (callable, optional): if the data sample should be done some transformations or not, such as resize the image.
        """
        self.data_tensor_x = data_tensor_x
        self.data_tensor_y = data_tensor_y
        self.df_header = df_header
        self.transformations = transformations
        self.df_x = data_tensor_x
        self.df_y = data_tensor_y

    def __len__(self):
        return self.df_x.shape[0]

    def __getitem__(self, idx):
        x = self.data_tensor_x[idx, :]
        y = self.data_tensor_y[idx]
        participant_id = self.df_header.iloc[idx].iloc[0]
        session_id = self.df_header.iloc[idx].iloc[1]

        if self.transformations:
            x = self.transformations(x)

        sample = {'x': x, 'y': y, 'participant_id': participant_id, 'session_id': session_id}

        return sample

def train(model, data_loader, use_cuda, loss_func, optimizer, writer, epoch, model_mode="train"):
    """
    This is the function to train, validate or test the model, depending on the model_mode parameter.
    :param model:
    :param data_loader:
    :param use_cuda:
    :param loss_func:
    :param optimizer:
    :param writer:
    :param epoch:
    :return:
    """
    global_step = None

    if model_mode == "train":
        columns = ['participant_id', 'session_id', 'true_label', 'predicted_label']
        results_df = pd.DataFrame(columns=columns)
        total_loss = 0.0

        model.train()  # set the model to training mode
        print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))

        for i, data in enumerate(data_loader):
            # update the global step
            global_step = i + epoch * len(data_loader)

            if use_cuda:
                imgs, labels = data['x'].cuda(), data['y'].cuda()
            else:
                imgs, labels = data['x'], data['y']


            predicted = torch.squeeze(model(imgs))
            batch_loss = loss_func(predicted, labels)
            total_loss += batch_loss.item()
            writer.add_scalar('loss', batch_loss.item(), global_step)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                row = [sub, data['session_id'][idx],
                       labels[idx].item(), predicted[idx].item()]
                row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
                results_df = pd.concat([results_df, row_df])

            # delete the temporary variables taking the GPU memory
            del imgs, labels, predicted, batch_loss
            torch.cuda.empty_cache()

        loss_batch_mean = total_loss / len(data_loader)
        torch.cuda.empty_cache()

    elif model_mode == "valid":
        results_df, total_loss = test(model, data_loader, use_cuda, loss_func)
        loss_batch_mean = total_loss / len(data_loader)
        writer.add_scalar('loss', loss_batch_mean, epoch)
        torch.cuda.empty_cache()

    else:
        raise ValueError('This mode %s was not implemented. Please choose between train and valid' % model_mode)

    return results_df, loss_batch_mean, global_step

def test(model, data_loader, use_cuda, loss_func):
    """
    The function to evaluate the testing data for the trained classifiers
    :param model:
    :param data_loader:
    :param use_cuda:
    :return:
    """

    columns = ['participant_id', 'session_id', 'true_label', 'predicted_label']
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    if use_cuda:
        model.cuda()

    model.eval()  # set the model to evaluation mode
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if use_cuda:
                imgs, labels = data['x'].cuda(), data['y'].cuda()
            else:
                imgs, labels = data['x'], data['y']

            predicted = torch.squeeze(model(imgs))
            loss = loss_func(predicted, labels)
            total_loss += loss.item()

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                row = [sub, data['session_id'][idx],
                       labels[idx].item(), predicted[idx].item()]

                row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
                results_df = pd.concat([results_df, row_df])

            del imgs, labels
            torch.cuda.empty_cache()

        results_df.reset_index(inplace=True, drop=True)
        torch.cuda.empty_cache()

    return results_df, total_loss

def save_checkpoint(state, loss_is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    import torch
    import os

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if loss_is_best:
        torch.save(state, os.path.join(checkpoint_dir, filename))

def save_checkpoint_calssification(state, accuracy_is_best, loss_is_best, checkpoint_dir, filename='checkpoint.pth.tar',
                    best_accuracy='best_acc', best_loss='best_loss'):
    import torch
    import os
    import shutil

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save(state, os.path.join(checkpoint_dir, filename))
    if accuracy_is_best:
        best_accuracy_path = os.path.join(checkpoint_dir, best_accuracy)
        if not os.path.exists(best_accuracy_path):
            os.makedirs(best_accuracy_path)
        shutil.copyfile(os.path.join(checkpoint_dir, filename),  os.path.join(best_accuracy_path, 'model_best.pth.tar'))

    if loss_is_best:
        best_loss_path = os.path.join(checkpoint_dir, best_loss)
        if not os.path.exists(best_loss_path):
            os.makedirs(best_loss_path)
        shutil.copyfile(os.path.join(checkpoint_dir, filename), os.path.join(best_loss_path, 'model_best.pth.tar'))

def load_model(model, checkpoint_dir, gpu, filename='model_best.pth.tar'):
    """
    Load the weights written in checkpoint_dir in the model object.

    :param model: (Module) CNN in which the weights will be loaded.
    :param checkpoint_dir: (str) path to the folder containing the parameters to loaded.
    :param gpu: (bool) if True a gpu is used.
    :param filename: (str) Name of the file containing the parameters to loaded.
    :return: (Module) the update model.
    """
    from copy import deepcopy
    import torch
    import os

    best_model = deepcopy(model)
    param_dict = torch.load(os.path.join(checkpoint_dir, filename), map_location="cpu")
    best_model.load_state_dict(param_dict['model'])

    if gpu:
        best_model = best_model.cuda()

    return best_model, param_dict['epoch']

def to_tsvs(output_dir, results_df, result, fold, dataset='train'):
    """
    Allows to save the outputs of the test function.

    :param output_dir: (str) path to the output directory.
    :param results_df: (DataFrame) the individual results per slice.
    :param result: (float) mae per batch.
    :param fold: (int) the fold for which the performances were obtained.
    :param selection: (str) the metrics on which the model was selected (best_acc, best_loss)
    :param dataset: (str) the dataset on which the evaluation was performed.
    :return:
    """
    performance_dir = os.path.join(output_dir, 'performances', 'fold_%i' % fold)

    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir)

    results_df.to_csv(os.path.join(performance_dir, dataset + '.tsv'), index=False, sep='\t')
    pd.DataFrame([result], index=[0]).to_csv(os.path.join(performance_dir, dataset + '_mae.tsv'), index=False, sep='\t')

def train_network(model, output_dir, fi, X_train, y_train, X_test, y_test, num_epochs,
                  batch_size, init_state, df_header, gpu, lr, weight_decay, opt):

    print("Running for the %d-th repetition of the repeated holdout CV" % fi)
    writer_train_batch = SummaryWriter(log_dir=(os.path.join(output_dir, "log_dir", "fold_%i" % fi,
                                                             "train_batch")))
    writer_train_all_data = SummaryWriter(log_dir=(os.path.join(output_dir, "log_dir", "fold_%i" % fi,
                                                                "train_all_data")))
    writer_valid = SummaryWriter(log_dir=(os.path.join(output_dir, "log_dir", "fold_%i" % fi, "valid")))

    data_train = DatasetShuffler(X_train, y_train, df_header)
    data_valid = DatasetShuffler(X_test, y_test, df_header)

    # Use argument load to distinguish training and testing
    train_loader = DataLoader(data_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    valid_loader = DataLoader(data_valid,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True)

    # Define Optimizer and Loss Function
    optimizer = eval("torch.optim." + opt)(filter(lambda x: x.requires_grad, model.parameters()),
                                                        lr=lr, weight_decay=weight_decay)
    loss = torch.nn.L1Loss()

    model.load_state_dict(init_state)

    # parameters used in training
    best_loss_valid = np.inf

    for epoch in range(num_epochs):
        print("At %i-th epoch." % epoch)

        # train the model
        train_df, mean_mae_train, global_step = train(model, train_loader, gpu, loss, optimizer, writer_train_batch, epoch, model_mode='train')
        print("For training, mean MAE loss: %f during each epoch using batch data %d" % (mean_mae_train, epoch))

        # calculate the accuracy with the whole training data for subject level balanced accuracy
        train_all_df, mean_mae_train_all, _ = train(model, train_loader, gpu, loss, optimizer, writer_train_all_data, epoch, model_mode='valid')
        print("For training, mean MAE loss: %f at the end of epoch by applying all training data %d" % (mean_mae_train_all, epoch))

        # at then end of each epoch, we validate one time for the model with the validation data
        valid_df, mean_mae_valid, _ = train(model, valid_loader, gpu, loss, optimizer, writer_valid, epoch, model_mode='valid')
        print("For validation, mean MAE loss: %f at the end of epoch %d" % (mean_mae_valid, epoch))

        # save the best model based on the best loss
        loss_is_best = mean_mae_valid < best_loss_valid
        best_loss_valid = min(mean_mae_valid, best_loss_valid)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'loss': mean_mae_valid,
            'optimizer': optimizer.state_dict(),
            'global_step': global_step},
            loss_is_best,
            os.path.join(output_dir, "best_model_dir", "fold_" + str(fi), "NN", "best_loss"),
            filename="model_best.pth.tar")

    # Final evaluation for all criteria
    model, best_epoch = load_model(model, os.path.join(output_dir, 'best_model_dir', 'fold_%i' % fi, 'NN', 'best_loss'),
                                   gpu=gpu, filename='model_best.pth.tar')

    train_df, mae_train = test(model, train_loader, gpu, loss)
    valid_df, mae_valid = test(model, valid_loader, gpu, loss)

    # write the information of subjects and performances into tsv files.
    to_tsvs(output_dir, train_df, mae_train/len(train_loader), fi, dataset='train')
    to_tsvs(output_dir, valid_df, mae_valid/len(valid_loader), fi, dataset='validation')

    del optimizer, writer_train_batch, writer_train_all_data, writer_valid
    torch.cuda.empty_cache()

def train_network_classification(model, output_dir, fi, X_train, y_train, X_test, y_test, num_epochs,
                  batch_size, init_state, df_header, gpu, lr, weight_decay, opt):

    print("Running for the %d-th repetition of the repeated holdout CV" % fi)
    writer_train_batch = SummaryWriter(log_dir=(os.path.join(output_dir, "log_dir", "fold_%i" % fi,
                                                             "train_batch")))
    writer_train_all_data = SummaryWriter(log_dir=(os.path.join(output_dir, "log_dir", "fold_%i" % fi,
                                                                "train_all_data")))
    writer_valid = SummaryWriter(log_dir=(os.path.join(output_dir, "log_dir", "fold_%i" % fi, "valid")))

    data_train = DatasetShuffler(X_train, y_train, df_header)
    data_valid = DatasetShuffler(X_test, y_test, df_header)

    # Use argument load to distinguish training and testing
    train_loader = DataLoader(data_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    valid_loader = DataLoader(data_valid,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True)

    # Define Optimizer and Loss Function
    optimizer = eval("torch.optim." + opt)(filter(lambda x: x.requires_grad, model.parameters()),
                                                        lr=lr, weight_decay=weight_decay)
    loss = torch.nn.CrossEntropyLoss()

    model.load_state_dict(init_state)

    # parameters used in training
    best_loss_valid = np.inf
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print("At %i-th epoch." % epoch)

        # train the model
        train_df, mean_loss_train, batch_accuracy_train, global_step = train_classification(model, train_loader, gpu, loss, optimizer, writer_train_batch, epoch, model_mode='train')
        print("For training, mean loss loss: %f during each epoch using batch data %d" % (mean_loss_train, epoch))

        # calculate the accuracy with the whole training data for subject level balanced accuracy
        train_all_df, mean_loss_train_all, batch_accuracy_train_all, _ = train_classification(model, train_loader, gpu, loss, optimizer, writer_train_all_data, epoch, model_mode='valid')
        print("For training, mean loss loss: %f at the end of epoch by applying all training data %d" % (mean_loss_train_all, epoch))

        # at then end of each epoch, we validate one time for the model with the validation data
        valid_df, mean_loss_valid, batch_accuracy_valid, _ = train_classification(model, valid_loader, gpu, loss, optimizer, writer_valid, epoch, model_mode='valid')
        print("For validation, mean loss loss: %f at the end of epoch %d" % (mean_loss_valid, epoch))

        # save the best model based on the best loss
        acc_is_best = batch_accuracy_valid > best_accuracy
        best_accuracy = max(best_accuracy, batch_accuracy_valid)
        loss_is_best = mean_loss_valid < best_loss_valid
        best_loss_valid = min(mean_loss_valid, best_loss_valid)

        save_checkpoint_calssification({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'loss': mean_loss_valid,
            'accuracy': batch_accuracy_valid,
            'optimizer': optimizer.state_dict(),
            'global_step': global_step},
            acc_is_best, loss_is_best,
            os.path.join(output_dir, "best_model_dir", "fold_" + str(fi), "NN"),
            filename="model_best.pth.tar")

    # Final evaluation for all criteria
    model, best_epoch = load_model(model, os.path.join(output_dir, 'best_model_dir', 'fold_%i' % fi, 'NN', 'best_acc'),
                                   gpu=gpu, filename='model_best.pth.tar')

    train_df, results_train = test_classification(model, train_loader, gpu, loss)
    valid_df, results_valid = test_classification(model, valid_loader, gpu, loss)

    ### change the participand_id column
    valid_df['participant_id'] = valid_df['participant_id'].astype(str)
    valid_df['participant_id'] = valid_df['participant_id'].str.split('tensor').str[1].str.split(')').str[0].str.split('(').str[1]
    valid_df['participant_id'] = valid_df['participant_id'].astype(int)

    train_df['participant_id'] = train_df['participant_id'].astype(str)
    train_df['participant_id'] = train_df['participant_id'].str.split('tensor').str[1].str.split(')').str[0].str.split('(').str[1]
    train_df['participant_id'] = train_df['participant_id'].astype(int)

    # write the information of subjects and performances into tsv files.
    to_tsvs_classification(output_dir, train_df, results_train, fi, dataset='train')
    to_tsvs_classification(output_dir, valid_df, results_valid, fi, dataset='validation')

    del optimizer, writer_train_batch, writer_train_all_data, writer_valid
    torch.cuda.empty_cache()

def test_classification(model, data_loader, use_cuda, loss_func):
    """
    The function to evaluate the testing data for the trained classifiers
    :param model:
    :param data_loader:
    :param use_cuda:
    :return:
    """

    columns = ['participant_id', 'session_id', 'true_label', 'predicted_label']
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0
    softmax = torch.nn.Softmax(dim=1)

    if use_cuda:
        model.cuda()

    model.eval()  # set the model to evaluation mode
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if use_cuda:
                imgs, labels = data['x'].cuda(), data['y'].cuda()
            else:
                imgs, labels = data['x'], data['y']

            output = model(imgs)
            normalized_output = softmax(output)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                row = [sub, data['session_id'][idx],
                       labels[idx].item(), predicted[idx].item()]

                row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
                results_df = pd.concat([results_df, row_df])

            del imgs, labels
            torch.cuda.empty_cache()


        # calculate the balanced accuracy
        results = evaluate_prediction(results_df.true_label.values.astype(int),
                                      results_df.predicted_label.values.astype(int))
        results_df.reset_index(inplace=True, drop=True)
        results['total_loss'] = total_loss
        torch.cuda.empty_cache()

    return results_df, results

def train_classification(model, data_loader, use_cuda, loss_func, optimizer, writer, epoch, model_mode="train"):
    """
    This is the function to train, validate or test the model, depending on the model_mode parameter.
    :param model:
    :param data_loader:
    :param use_cuda:
    :param loss_func:
    :param optimizer:
    :param writer:
    :param epoch:
    :return:
    """
    global_step = None
    softmax = torch.nn.Softmax(dim=1)
    if model_mode == "train":
        columns = ['participant_id', 'session_id', 'true_label', 'predicted_label']
        results_df = pd.DataFrame(columns=columns)
        total_loss = 0.0

        model.train()  # set the model to training mode
        print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))

        for i, data in enumerate(data_loader):
            # update the global step
            global_step = i + epoch * len(data_loader)

            if use_cuda:
                imgs, labels = data['x'].cuda(), data['y'].cuda()
            else:
                imgs, labels = data['x'], data['y']

            gound_truth_list = labels.data.cpu().numpy().tolist()
            output = model(imgs)
            normalized_output = softmax(output)
            _, predicted = torch.max(output.data, 1)
            predict_list = predicted.data.cpu().numpy().tolist()
            batch_loss = loss_func(output, labels)
            total_loss += batch_loss.item()

            # calculate the batch balanced accuracy and loss
            batch_metrics = evaluate_prediction(gound_truth_list, predict_list)
            batch_accuracy = batch_metrics['balanced_accuracy']

            writer.add_scalar('classification accuracy', batch_accuracy, global_step)
            writer.add_scalar('loss', batch_loss.item(), global_step)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                row = [sub, data['session_id'][idx],
                       labels[idx].item(), predicted[idx].item()]
                row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
                results_df = pd.concat([results_df, row_df])

            # delete the temporary variables taking the GPU memory
            del imgs, labels, predicted, batch_loss
            torch.cuda.empty_cache()

        loss_batch_mean = total_loss / len(data_loader)
        torch.cuda.empty_cache()

    elif model_mode == "valid":
        results_df, results_val = test_classification(model, data_loader, use_cuda, loss_func)
        batch_accuracy = results_val['balanced_accuracy']
        total_loss = results_val['total_loss']
        loss_batch_mean = total_loss / len(data_loader)
        writer.add_scalar('classification accuracy', batch_accuracy, epoch)
        writer.add_scalar('loss', loss_batch_mean, epoch)

        torch.cuda.empty_cache()

    else:
        raise ValueError('This mode %s was not implemented. Please choose between train and valid' % model_mode)

    return results_df, loss_batch_mean, batch_accuracy, global_step

def to_tsvs_classification(output_dir, results_df, result, fold, dataset='train'):
    """
    Allows to save the outputs of the test function.

    :param output_dir: (str) path to the output directory.
    :param results_df: (DataFrame) the individual results per slice.
    :param result: (float) mae per batch.
    :param fold: (int) the fold for which the performances were obtained.
    :param selection: (str) the metrics on which the model was selected (best_acc, best_loss)
    :param dataset: (str) the dataset on which the evaluation was performed.
    :return:
    """
    performance_dir = os.path.join(output_dir, 'performances', 'fold_%i' % fold)

    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir)

    results_df.to_csv(os.path.join(performance_dir, dataset + '.tsv'), index=False, sep='\t')
    pd.DataFrame([result], index=[0]).to_csv(os.path.join(performance_dir, dataset + '_metrics.tsv'), index=False, sep='\t')

def apply_2_external_data_hydra(output_model_folder, participant_tsv_validation, cov_tsv_validation, output_tsv_folder, num_subtype):
    """
    This is to apply the trained HYDRA model to external data (should be harmonized to the training data if from different studies/sites)
    Args:
        output_model_folder: The model folder: {output_dir}/clustering/{k}_clusters/models
        participant_tsv_validation: the participant tsv for the validation set; should follow the same header as the training data
        cov_tsv_validation: the covariate tsv for the validation set; should follow the same header as the training data
        output_tsv_folder: {output_dir}/clustering/{k}_clusters/tsv

    Returns:
    """
    from mlni.hydra_clustering import RB_Input
    from joblib import load

    df_header_val = pd.read_csv(participant_tsv_validation, sep='\t')[['participant_id', 'session_id', 'diagnosis']]
    if cov_tsv_validation == None:
        input_data = RB_Input(participant_tsv_validation, covariate_tsv=None)
    else:
        input_data = RB_Input(participant_tsv_validation, covariate_tsv=cov_tsv_validation)

    feature = input_data.get_x()

    ## load the saved hyperplanes for the k subtypes
    for i in range(num_subtype):
        ## calculate the HYDRA scores
        model = load(os.path.join(output_model_folder, 'svm-' + str(i) + '_last_repetition.joblib'))
        weights = model.coef_
        bias = model.intercept_
        y_hat = np.dot(feature, weights.transpose()) + bias
        df_header_val['subtype' + '_' + str(i)] = y_hat

    df_header_val.to_csv(os.path.join(output_tsv_folder, 'Validation_external_hydra_scores.tsv'), index=False, sep='\t', encoding='utf-8')

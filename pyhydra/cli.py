import argparse

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def classification_roi_func(args):
    """
    The default function to run classificaiton.
    Args:
        args: args from parser

    Returns:

    """
    from .adml_classification import classification_roi
    classification_roi(
        args.feature_tsv,
        args.output_dir,
        args.cv_repetition,
        args.cv_strategy,
        args.class_weight_balanced,
        args.n_threads,
        args.verbose
    )

def classification_voxel_func(args):
    """
    The default function to run classificaiton.
    Args:
        args: args from parser

    Returns:

    """
    from .adml_classification import classification_voxel
    classification_voxel(
        args.feature_tsv,
        args.output_dir,
        args.cv_repetition,
        args.cv_strategy,
        args.class_weight_balanced,
        args.n_threads,
        args.verbose
    )

def clustering_func(args):
    """
    The default function to run classificaiton.
    Args:
        args: args from parser

    Returns:

    """
    from .hydra_clustering import clustering
    clustering(
        args.feature_tsv,
        args.output_dir,
        args.k_min,
        args.k_max,
        args.cv_repetition,
        args.covariate_tsv,
        args.cv_strategy,
        args.save_models,
        args.cluster_predefined_c,
        args.class_weight_balanced,
        args.weight_initialization_type,
        args.num_iteration,
        args.num_consensus,
        args.tol,
        args.n_threads,
        args.verbose
    )

def parse_command_line():
    """
    Definition for the commandline parser
    Returns:

    """

    parser = argparse.ArgumentParser(
        prog='pyhydra',
        description='pyHYDRA for semi-supervised clustering and supervised classification.')

    subparser = parser.add_subparsers(
        title='''Task to perform per needs:''',
        description='''What kind of task do you want to use with pyHYDRA?
            (clustering, classification_roi, classification_voxel).''',
        dest='task',
        help='''****** Tasks proposed by pyHYDRA ******''')

    subparser.required = True

########################################################################################################################

    ## Add arguments for pyHYDRA classification
    classification_parser_roi = subparser.add_parser(
        'classify_roi',
        help='Perform binary classification for ROI features.')

    classification_parser_roi.add_argument(
        'feature_tsv',
        help="Path to the tsv containing extracted feature, following the BIDS convention. The tsv contains the following first columns:"
             "i) the first column is the participant_id. "
             "ii) the second column should be the session_id. "
             "iii) the third column should be the diagnosis. "
             "Following columns are the extracted feature per column",
        default=None
    )

    classification_parser_roi.add_argument(
        'output_dir',
        help='Path to store the classification results.',
        default=None
    )

    classification_parser_roi.add_argument(
        'cv_repetition',
        help='Number of repetitions for the chosen cross-validation (CV).',
        default=None, type=int
    )

    classification_parser_roi.add_argument(
        '-cs', '--cv_strategy',
        help='Chosen CV strategy, default is hold_out. ',
        type=str, default='hold_out',
        choices=['k_fold', 'hold_out'],
    )

    classification_parser_roi.add_argument(
        '-cwb', '--class_weight_balanced',
        help='If group samples are balanced, default is True. ',
        default=False, action="store_true"
    )

    classification_parser_roi.add_argument(
        '-nt', '--n_threads',
        help='Number of cores used, default is 4',
        type=int, default=4
    )

    classification_parser_roi.add_argument(
        '-v', '--verbose',
        help='Increase output verbosity',
        default=False, action="store_true"
    )

    classification_parser_roi.set_defaults(func=classification_roi_func)

########################################################################################################################

    ## Add arguments for pyHYDRA classification
    classification_parser_voxel = subparser.add_parser(
        'classify_voxel',
        help='Perform binary classification  for voxel-wise features.')

    classification_parser_voxel.add_argument(
        'feature_tsv',
        help="Path to the tsv containing extracted feature, following the BIDS convention. The tsv contains the following first columns:"
             "i) the first column is the participant_id. "
             "ii) the second column should be the session_id. "
             "iii) the third column should be the diagnosis. "
             "iv) the third column should be the path. "
             "Following columns are the extracted feature per column",
        default=None
    )

    classification_parser_voxel.add_argument(
        'output_dir',
        help='Path to store the classification results.',
        default=None
    )

    classification_parser_voxel.add_argument(
        'cv_repetition',
        help='Number of repetitions for the chosen cross-validation (CV).',
        default=None, type=int
    )

    classification_parser_voxel.add_argument(
        '-cs', '--cv_strategy',
        help='Chosen CV strategy, default is hold_out. ',
        type=str, default='hold_out',
        choices=['k_fold', 'hold_out'],
    )

    classification_parser_voxel.add_argument(
        '-cwb', '--class_weight_balanced',
        help='If group samples are balanced, default is True. ',
        default=False, action="store_true"
    )

    classification_parser_voxel.add_argument(
        '-nt', '--n_threads',
        help='Number of cores used, default is 4',
        type=int, default=4
    )

    classification_parser_voxel.add_argument(
        '-v', '--verbose',
        help='Increase output verbosity',
        default=False, action="store_true"
    )

    classification_parser_voxel.set_defaults(func=classification_voxel_func)

########################################################################################################################
    ## Add arguments for pyHYDRA clustering
    clustering_parser = subparser.add_parser(
        'cluster',
        help='Perform semi-supervised clustering via HYDRA.')

    clustering_parser.add_argument(
        'feature_tsv',
        help="Path to the tsv containing extracted feature, following the BIDS convention. The tsv contains the following first columns:"
             "i) the first column is the participant_id. "
             "ii) the second column should be the session_id. "
             "iii) the third column should be the diagnosis. "
             "Following columns are the extracted feature per column",
        default=None
    )

    clustering_parser.add_argument(
        'output_dir',
        help='Path to store the classification results.',
        default=None
    )

    clustering_parser.add_argument(
        'k_min',
        help='Number of cluster (k) minimum value.',
        default=None, type=int
    )

    clustering_parser.add_argument(
        'k_max',
        help='Number of cluster (k) maximum value.',
        default=None, type=int
    )

    clustering_parser.add_argument(
        'cv_repetition',
        help='Number of repetitions for the chosen cross-validation (CV).',
        default=None, type=int
    )

    clustering_parser.add_argument(
        '--covariate_tsv',
        help="Path to the tsv containing covariates, following the BIDS convention. The first 3 columns is the same as feature_tsv",
        default=None,
        type=str
    )

    clustering_parser.add_argument(
        '-cs', '--cv_strategy',
        help='Chosen CV strategy, default is hold_out. ',
        type=str, default='hold_out',
        choices=['k_fold', 'hold_out'],
    )

    clustering_parser.add_argument(
        '-sm', '--save_models',
        help='If save modles during all repetitions of CV. ',
        default=False, action="store_true"
    )

    clustering_parser.add_argument(
        '--cluster_predefined_c',
        type=float,
        default=0.25,
        help="Predefined hyperparameter C of SVM. Default is 0.25. "
             "Better choice may be guided by pyHYDRA global classification with nested CV for optimal C searching. "
    )

    clustering_parser.add_argument(
        '-cwb', '--class_weight_balanced',
        help='If group samples are balanced, default is True. ',
        default=False, action="store_true"
    )

    clustering_parser.add_argument(
        '-wit', '--weight_initialization_type',
        help='Strategy for initializing the weighted sample matrix of the polytope. ',
        type=str, default='DPP',
        choices=['DPP', 'random_assign'],
    )

    clustering_parser.add_argument(
        '--num_iteration',
        help='Number of iteration to converge each SVM.',
        default=50, type=int
    )

    clustering_parser.add_argument(
        '--num_consensus',
        help='Number of iteration for inner consensus clusetering.',
        default=20, type=int
    )

    clustering_parser.add_argument(
        '--tol',
        help='Clustering stopping criterion, until the polytope becomes stable',
        default=1e-8, type=float
    )

    clustering_parser.add_argument(
        '-nt', '--n_threads',
        help='Number of cores used, default is 4',
        type=int, default=4
    )

    clustering_parser.add_argument(
        '-v', '--verbose',
        help='Increase output verbosity',
        default=False, action="store_true"
    )


    clustering_parser.set_defaults(func=clustering_func)

    return parser












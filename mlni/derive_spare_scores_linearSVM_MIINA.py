from mlni.base import RB_Input, VB_Input
import os
import numpy as np

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


def derive_spare_score(feature_tsv, weight_txt):
    source_data = RB_Input(feature_tsv, standardization_method="minmax")
    kernel = source_data.get_kernel()
    y = source_data.get_y()

    w = np.loadtxt(weight_txt)

    # w = revert_mask(weights, source_mask, source_orig_shape).flatten()
    b = np.loadtxt(path.join(options.output_dir_source, 'classifier', 'fold_' + str(fi), 'intersect.txt'))

    target_image = target_data.get_x()
    target_label = target_data.get_y()

    y_hat = np.dot(w, target_image.transpose()) + b
    y_binary = (y_hat < 0) * 1.0

    evaluation = evaluate_prediction(target_label, y_binary)

    del evaluation['confusion_matrix']
    if fi == 0:
        res_df = pd.DataFrame(evaluation, index=[fi])
        res_final = pd.DataFrame(columns=list(res_df.columns))
    else:
        res_df = pd.DataFrame(evaluation, index=[fi])

    res_final = res_final.append(res_df)

result_dir = path.join(options.output_dir_target, 'test')
if not path.exists(result_dir):
    os.makedirs(result_dir)
res_final.to_csv(path.join(options.output_dir_target, 'test', 'results.tsv'), sep='\t', index=False)

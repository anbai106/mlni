from adml_regression_mlp import regression_roi

feature_tsv="/cbica/home/wenju/Reproducibile_paper/BrainAge/output/training_tsv/train_muse.tsv"
output_dir = "/home/hao/test"
cv_repetition=5
regression_roi(feature_tsv, output_dir, cv_repetition)
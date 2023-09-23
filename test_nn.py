from mlni.adml_classification_nn import classification_roi
feature_tsv = '/cbica/home/wenju/Reproducibile_paper/BrainEye/output/MLNI/data/training/mental_behavioural_disorder_diagnosis_C1024brain.tsv'
output_dir = '/home/hao/test/mlni'
classification_roi(feature_tsv, output_dir, 1, epochs=100, n_threads=8, gpu=True)
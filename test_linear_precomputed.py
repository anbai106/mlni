from mlni.adml_classification import classification_roi
import os

feature_tsv= "/run/user/1000/gvfs/sftp:host=cubic-login,user=wenju/cbica/home/wenju/Dataset/sopNMF_atlas/All_sites/Classification/AD/data/PSC32/AD_CN_allsites.tsv"
output_dir = "/run/user/1000/gvfs/sftp:host=cubic-login,user=wenju/cbica/home/wenju/Dataset/sopNMF_atlas/All_sites/Classification/AD/results/PSC32/SpareScore"
cv_repetition = 5
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(os.path.join(output_dir, 'classification', 'mean_results.tsv')):
    classification_roi(feature_tsv, output_dir, cv_repetition)
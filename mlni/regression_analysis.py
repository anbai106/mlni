# import os
# import pandas as pd
# from mlni.adml_regression import regression_roi
# from mlni.adml_regression_rbf import regression_roi as regression_roi_rbf
# from mlni.adml_regression_nn import regression_roi as regression_roi_nn
# from mlni.adml_regression_lasso import regression_roi as regression_roi_lasso
# from mlni.adml_regression_mlp import regression_roi as regression_roi_mlp
# import torch

# def run_all_regressors(feature_tsv_path, output_dir, cv_repetition):
#     """
#     Run all regressor types and collect their results
#     """
#     results = []
#     gpu_available = torch.cuda.is_available()
    
#     #'cv_strategy': 'k_fold'
#     # Define regressor configurations with k-fold support where possible
#     regressors = [
#         ('linear_svr', regression_roi, {'cv_strategy': 'k_fold'}),  
#         ('rbf_svr', regression_roi_rbf, {'cv_strategy': 'k_fold'}),  
#         ('neural_net', regression_roi_nn, {
#             'gpu': gpu_available
#         }),  # Neural Net only supports hold-out
#         ('lasso', regression_roi_lasso, {}),  
#         ('mlp', regression_roi_mlp, {}) 
#     ]
    
#     for regressor_name, regressor_func, extra_params in regressors:
#         print(f"\nRunning {regressor_name}...")
        
#         # Create regressor-specific output directory
#         regressor_output_dir = os.path.join(output_dir, regressor_name)
#         os.makedirs(regressor_output_dir, exist_ok=True)
        
#         # Run the regression
#         try:
#             # Check if k-fold strategy is specified and supported
#             if extra_params.get('cv_strategy') == 'k_fold':
#                 regressor_func(feature_tsv_path, regressor_output_dir, cv_repetition, cv_strategy='k_fold', **{k:v for k,v in extra_params.items() if k != 'cv_strategy'})
#             else:
#                 regressor_func(feature_tsv_path, regressor_output_dir, cv_repetition, **extra_params)
            
#             # Read results from mean_results.tsv
#             results_path = os.path.join(regressor_output_dir, 'regression', 'mean_results.tsv')
#             if os.path.exists(results_path):
#                 df = pd.read_csv(results_path, sep='\t')
                
#                 # Extract metrics
#                 result = {
#                     'regressor': regressor_name,
#                     'mae': df['mae_mean'].iloc[0] if 'mae_mean' in df.columns else df['mae'].iloc[0],
#                     'r': df['pearson_r'].iloc[0] if 'pearson_r' in df.columns else None
#                 }
#                 results.append(result)
            
#         except Exception as e:
#             print(f"Error running {regressor_name}: {str(e)}")
    
#     # Create summary DataFrame and save
#     summary_df = pd.DataFrame(results)
#     summary_path = os.path.join(output_dir, 'regression_summary.tsv')
#     summary_df.to_csv(summary_path, sep='\t', index=False)
    
#     return summary_df

# def run_regression_analysis(final_datasets_directory, output_base_directory, cv_repetition):
#     """
#     Run regression analysis on all datasets in the directory
#     """
#     os.makedirs(output_base_directory, exist_ok=True)
#     all_results = []
    
#     # Get list of TSV files
#     tsv_files = [f for f in os.listdir(final_datasets_directory) if f.endswith(".tsv")]
    
#     for tsv_file in tsv_files:
#         print(f"\nProcessing file: {tsv_file}")
        
#         # Setup paths
#         feature_tsv_path = os.path.join(final_datasets_directory, tsv_file)
#         dataset_name = os.path.splitext(tsv_file)[0]
#         output_dir = os.path.join(output_base_directory, dataset_name)
        
#         # Run all regressors
#         results_df = run_all_regressors(feature_tsv_path, output_dir, cv_repetition)
#         results_df['dataset'] = dataset_name
#         all_results.append(results_df)
    
#     # Combine all results
#     final_results = pd.concat(all_results, ignore_index=True)
#     final_results.to_csv(os.path.join(output_base_directory, 'all_results_summary.tsv'), sep='\t', index=False)
    
#     print("\nAll datasets processed successfully.")
#     return final_results
import re
from collections import defaultdict
import numpy as np
import pandas as pd

def extract_metrics(file_names, metrics, exclude_cond = lambda x: False):
        

    data = defaultdict(lambda: defaultdict(list))

    # Regular expressions to match lines
    exp_pattern = re.compile(r"EXPERIMENT: (.+)_([\d\.]+)")
    metric_pattern = re.compile(r"test_(\w+): ([\d\.]+)")
    minus_metric_pattern = re.compile(r"test_(\w+): -([\d\.]+)")
    add_metric_pattern = re.compile(r"Best performing model: Test Top 20 DE MSE: ([\d\.]+)")

    for file_name in file_names:
        with open(file_name, 'r') as file:
            current_experiment = None
            for line in file:
                # Check for experiment line
                exp_match = exp_pattern.match(line)
                if exp_match:
                    experiment_name = exp_match.group(1)
                    current_experiment = experiment_name
                    continue

                # Check for metric line
                if current_experiment:
                    if exclude_cond(current_experiment):
                        continue
                    metric_match = metric_pattern.match(line)
                    if metric_match:
                        metric_name, metric_value = metric_match.groups()
                        if metric_name in metrics:
                            data[current_experiment][metric_name].append(float(metric_value))
                    minus_metric_match = minus_metric_pattern.match(line)
                    if minus_metric_match:
                        metric_name, metric_value = minus_metric_match.groups()
                        if metric_name in metrics:
                            data[current_experiment][metric_name].append(-float(metric_value))
                    add_metric_match = add_metric_pattern.match(line)
                    if add_metric_match:
                        metric_name = "mse_top20_de"
                        metric_value = add_metric_match.group(1)
                        if metric_name in metrics:
                            data[current_experiment][metric_name].append(float(metric_value))

    # Calculate mean and std for each metric of each model
    results = {'model': []}
    for model in data.keys():
        results['model'].append(model)
        for metric_name, values in data[model].items():
            if metric_name not in results.keys():
                results[metric_name] = []
                results[f"{metric_name}_std"] = []
            if len(values) > 0:
                results[metric_name].append(np.mean(values))
                results[f"{metric_name}_std"].append(np.std(values))
            else:
                results[metric_name].append(0)
                results[f"{metric_name}_std"].append(0)
    results = pd.DataFrame(results, index = None)
    return results

def exclude_cond_func_final(x):
    if 'unperturb' in x:
        return True
    if 'ctrl_mean' in x and not ('norman' in x):
        return True
    if 'CPA' in x and not ('norman' in x):
        return True
    if 'ctrl_seed' in x or 'GENE2VEC' in x:
        return True
    if 'g2v' in x and not ('g2v_Y_app_R_bp_Y_pw_R'  in x):
        return True
    return False



    


txt_files = ["/l/users/ding.bai/all_backup/pert_new/mywork/res/output/run_unperturb_ctrl.txt",
             "/l/users/ding.bai/all_backup/pert_new/GEARS/res/output/base_norman.txt", 
             #
             "/home/ding.bai/ding-geneformer-downstream/GEARS/res/output/geneformer_64_05-07/geneformer_64_05-07.txt",
             "/home/ding.bai/ding-geneformer-downstream/GEARS/res/output/geneformer_64_04-26/geneformer_64_04-26.txt",
             "/home/ding.bai/ding-geneformer-downstream/GEARS/res/output/geneformer_64_04-10/geneformer_64_04-10.txt",
             "/home/ding.bai/ding-geneformer-downstream/GEARS/res/output/geneformer_64_04-25/geneformer_64_04-25.txt",
             "/home/ding.bai/ding-scfmv1-downstream/GEARS/res/output/ddp-hs1024-1b-02-05_2024-08-22_21-51-30/scfm_gears_hs1024-1b-02-05.txt",
             "/home/ding.bai/ding-scfmv1-downstream/GEARS/res/output/ddp-hs1024-1b-02-05_2024-10-09_18-06-32/scfm_gears_hs1024-1b-02-05.txt"]

metrics = ["mse_top20_de",
            "combo_seen0_mse_top20_de_non_dropout",
           "combo_seen1_mse_top20_de_non_dropout",
           "combo_seen2_mse_top20_de_non_dropout",
           "unseen_single_mse_top20_de_non_dropout"]

results = extract_metrics(txt_files, metrics = metrics, exclude_cond=exclude_cond_func_final)
results.to_csv("csvs/norman_split_and_ctrl_mse_top20_de.csv")

metrics = ["combo_seen0_pearson_delta_top20_de_non_dropout",
            "combo_seen1_pearson_delta_top20_de_non_dropout",
            "combo_seen2_pearson_delta_top20_de_non_dropout",
           "unseen_single_pearson_delta_top20_de_non_dropout"]

results = extract_metrics(txt_files, metrics = metrics, exclude_cond=exclude_cond_func_final)
results.to_csv("csvs/norman_split_and_ctrl_pearson_delta.csv")
            
metrics = [ "combo_seen0_frac_opposite_direction_top20_non_dropout",
            "combo_seen1_frac_opposite_direction_top20_non_dropout",
            "combo_seen2_frac_opposite_direction_top20_non_dropout",
            "unseen_single_frac_opposite_direction_top20_non_dropout"]

results = extract_metrics(txt_files, metrics = metrics, exclude_cond=exclude_cond_func_final)
results.to_csv("csvs/norman_split_and_ctrl_pod.csv")


metrics = ["combo_seen0_mse_top200_hvg",
            "combo_seen1_mse_top200_hvg",
            "combo_seen2_mse_top200_hvg",
           "unseen_single_mse_top200_hvg"]

results = extract_metrics(txt_files, metrics = metrics, exclude_cond=exclude_cond_func_final)
results.to_csv("csvs/norman_split_and_ctrl_mse_top200_hvg.csv")



metrics = ["combo_seen0_mse_top200_de",
            "combo_seen1_mse_top200_de",
            "combo_seen2_mse_top200_de",
           "unseen_single_mse_top200_de"]

results = extract_metrics(txt_files, metrics = metrics, exclude_cond=exclude_cond_func_final)
results.to_csv("csvs/norman_split_and_ctrl_mse_top200_de.csv")


metrics = ["combo_seen0_frac_in_range_45_55_non_dropout",
            "combo_seen1_frac_in_range_45_55_non_dropout",
            "combo_seen2_frac_in_range_45_55_non_dropout",
           "unseen_single_frac_in_range_45_55_non_dropout"]

results = extract_metrics(txt_files, metrics = metrics, exclude_cond=exclude_cond_func_final)
results.to_csv("csvs/norman_split_and_ctrl_frac_in_range_45_55_non_dropout.csv")



metrics = ["combo_seen0_frac_correct_direction_20_nonzero",
            "combo_seen1_frac_correct_direction_20_nonzero",
            "combo_seen2_frac_correct_direction_20_nonzero",
           "unseen_single_frac_correct_direction_20_nonzero"]

results = extract_metrics(txt_files, metrics = metrics, exclude_cond=exclude_cond_func_final)
results.to_csv("csvs/norman_split_and_ctrl_frac_correct_direction_20_nonzero.csv")


metrics = ["combo_seen0_frac_sigma_below_1_non_dropout",
            "combo_seen1_frac_sigma_below_1_non_dropout",
            "combo_seen2_frac_sigma_below_1_non_dropout",
            "unseen_single_frac_sigma_below_1_non_dropout"]

results = extract_metrics(txt_files, metrics = metrics, exclude_cond=exclude_cond_func_final)
results.to_csv("csvs/norman_split_and_ctrl_frac_sigma_below_1_non_dropout.csv")

metrics = ["combo_seen0_frac_sigma_below_2_non_dropout",
            "combo_seen1_frac_sigma_below_2_non_dropout",
            "combo_seen2_frac_sigma_below_2_non_dropout",
            "unseen_single_frac_sigma_below_2_non_dropout"]

results = extract_metrics(txt_files, metrics = metrics, exclude_cond=exclude_cond_func_final)
results.to_csv("csvs/norman_split_and_ctrl_frac_sigma_below_2_non_dropout.csv")




metrics = ["combo_seen0_frac_sigma_below_1",
            "combo_seen1_frac_sigma_below_1",
            "combo_seen2_frac_sigma_below_1",
            "unseen_single_frac_sigma_below_1"]

results = extract_metrics(txt_files, metrics = metrics, exclude_cond=exclude_cond_func_final)
results.to_csv("csvs/norman_split_and_ctrl_frac_sigma_below_1.csv")

metrics = ["combo_seen0_frac_sigma_below_2",
            "combo_seen1_frac_sigma_below_2",
            "combo_seen2_frac_sigma_below_2",
            "unseen_single_frac_sigma_below_2"]

results = extract_metrics(txt_files, metrics = metrics, exclude_cond=exclude_cond_func_final)
results.to_csv("csvs/norman_split_and_ctrl_frac_sigma_below_2.csv")
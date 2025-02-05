import scanpy as sc
from scipy import sparse
from scipy import stats
import numpy as np
import os
from collections import defaultdict
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from copy import deepcopy
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
import yaml
from matplotlib_venn import venn3

# Set Arial globally in Matplotlib configuration
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['lines.linewidth'] = 0.5
#set +o noclobber
#python GI_analysis.py > txt/GI_analysis.txt 2>&1

REPEAT = 2
ad = sc.read_h5ad('/l/users/ding.bai/all_backup/pert_new/mywork/data/norman/perturb_processed.h5ad')

ctrl = ad[ad.obs.condition == 'ctrl'].X.toarray().mean(0) #N,
print('ctrl:', ctrl.shape)

DE_nums = [50,]
for DE_num in DE_nums:

    conditions = ad.obs.condition.unique()
    pert_genes = []
    for condition in conditions:
        if condition == 'ctrl':
            continue
        cond1 = condition.split('+')[0]
        cond2 = condition.split('+')[1]
        if cond1 != 'ctrl': 
            if cond1 not in pert_genes:
                pert_genes.append(cond1)
        if cond2 != 'ctrl':
            if cond2 not in pert_genes:
                pert_genes.append(cond2)

    num_genes = len(ad.var.gene_name)
    if DE_num == 20:
        DE_dict_raw = ad.uns['top_non_dropout_de_20']
    else:
        DE_dict_raw = ad.uns['rank_genes_groups_cov_all']
        for k, v in DE_dict_raw.items():
            DE_dict_raw[k] = v[:DE_num]
    DE_dict = defaultdict(list)
    symbol_to_idx = {symbol: i for i, symbol in enumerate(ad.var.gene_name.tolist())}
    ens_to_idx = {ens: i for i, ens in enumerate(ad.var_names.tolist())}
    for key, item in DE_dict_raw.items():
        symbols = key.split('_')[1].split('+')
        new_key = []
        for symbol in symbols:
            if symbol != 'ctrl':
                new_key.append(symbol_to_idx[symbol])
        new_key.sort()
        new_key = tuple(new_key)
        if new_key in DE_dict.keys():
            continue

        new_item = []
        for ens in item:
            new_item.append(ens_to_idx[ens])
        DE_dict[new_key] = np.array(new_item, dtype=int)
    print('new DE dict: ', len(DE_dict.keys()), DE_dict[new_key])

    
    ad = sc.read_h5ad('/l/users/ding.bai/all_backup/pert_new/mywork/data/norman/perturb_processed.h5ad')
    with open('/l/users/ding.bai/all_backup/pert_new/GEARS/data_all/gene2go_all.pkl', 'rb') as f:
        g2g = pickle.load(f)
    with open('/l/users/ding.bai/all_backup/pert_new/GEARS/data_all/essential_all_data_pert_genes.pkl', 'rb') as f:
        essential_pert = pickle.load(f)
    g2g = {i: g2g[i] for i in essential_pert if i in g2g}

    ctrl = ad[ad.obs.condition == 'ctrl'].X.toarray().mean(0) #N,
    print('ctrl:', ctrl.shape)

    essential_pert = list(np.unique(list(g2g.keys())))
    conditions = ad.obs.condition.unique()
    pert_genes = []
    for condition in conditions:
        if condition == 'ctrl':
            continue
        cond1 = condition.split('+')[0]
        cond2 = condition.split('+')[1]
        if cond1 != 'ctrl': 
            if cond1 not in pert_genes:
                pert_genes.append(cond1)
        if cond2 != 'ctrl':
            if cond2 not in pert_genes:
                pert_genes.append(cond2)
    for pert_gene in pert_genes:
        if pert_gene not in essential_pert:
            essential_pert.append(pert_gene)
    essential_pert = np.array(essential_pert)
    print('essential_pert len: ', len(essential_pert))
    print('essential_pert[-10:]: ', essential_pert[:10], essential_pert[-10:])

    num_genes = len(ad.var.gene_name)

    DE_dict = defaultdict(list)
    symbol_to_idx = {symbol: i for i, symbol in enumerate(ad.var.gene_name.tolist())}
    ens_to_idx = {ens: i for i, ens in enumerate(ad.var_names.tolist())}
    for key, item in DE_dict_raw.items():
        symbols = key.split('_')[1].split('+')
        new_key = []
        for symbol in symbols:
            if symbol != 'ctrl':
                new_key.append(symbol_to_idx[symbol])
        new_key.sort()
        new_key = tuple(new_key)
        if new_key in DE_dict.keys():
            continue

        new_item = []
        for ens in item:
            new_item.append(ens_to_idx[ens])
        DE_dict[new_key] = np.array(new_item, dtype=int)
    print('new DE dict: ', len(DE_dict.keys()), DE_dict[new_key])

    if not os.path.exists('csvs/norman_go_20.csv'):
        go_graph = pd.read_csv('/l/users/ding.bai/all_backup/pert_new/mywork/data/norman/go.csv')
        gene_degrees = np.arange(num_genes)
        new_graph = pd.DataFrame({'source': [], 'target': [], 'importance': []})
        for symbol, idx in tqdm(symbol_to_idx.items()):
            symbol_graph = go_graph[go_graph['source'] == symbol]
            weights = symbol_graph['importance']
            weights = np.array(weights)
            if len(weights) <= 20:
                min_weight = 0
            else:
                weights.sort()
                min_weight = weights[-20]
            new_graph = pd.concat([new_graph, symbol_graph[symbol_graph['importance'] >= min_weight]])
        new_graph.to_csv('csvs/norman_go_20.csv')

    go_graph = pd.read_csv('csvs/norman_go_20.csv')
    gene_degrees = np.arange(num_genes)
    for symbol, idx in tqdm(symbol_to_idx.items()):
        weights = go_graph[go_graph['source'] == symbol]['importance']
        if len(weights) > 0:
            gene_degrees[idx] = np.array(weights).sum()
        else:
            gene_degrees[idx] = 0
    print('go_graph max, min degree: ', gene_degrees.max(), gene_degrees.min()) 

    pert_graph_dict = {}
    for pert in DE_dict.keys():
        pert_nodes = []
        for pi in pert:
            pi_symbol = ad.var.gene_name.tolist()[pi]
            for target in go_graph[go_graph['source'] == pi_symbol]['target'].tolist():
                pert_nodes.append(symbol_to_idx[target])
            for source in go_graph[go_graph['target'] == pi_symbol]['source'].tolist():
                pert_nodes.append(symbol_to_idx[source])
        pert_nodes = np.unique(pert_nodes).astype(int)
        pert_graph_dict[pert] = pert_nodes

    
    pred_dir_dict = {'scLong': '/home/ding.bai/ding-scfmv1-downstream/GEARS/result_process/preds/scfm_gears_downstream_hs1024-1b-02-05',
                    'GEARS': '/l/users/ding.bai/all_backup/pert_new/GEARS/preds/GEARS_seed_1_data_norman'}
    methods = list(pred_dir_dict.keys())

    method_pert_mean_dicts_dict = {}
    perts_test = []
    perts_0_1 = []
    perts_0_2 = []
    perts_1_2 = []
    perts_2_2 = []
    for method, pred_dir in pred_dir_dict.items():
        pert_mean_dicts = []

        for r in range(REPEAT):
            repeat_pred_dir = f'{pred_dir}_{r}'
            sp_files = os.listdir(repeat_pred_dir)
            pred_arrays = []
            for sp_file in sp_files:
                sparse_matrix = sparse.load_npz(f'{repeat_pred_dir}/{sp_file}')
                dense_matrix = sparse_matrix.toarray().reshape(-1, num_genes, 3) #B, N, 3
                pred_arrays.append(dense_matrix)
            pred_array = np.concatenate(pred_arrays, axis = 0) #All, N, 3
            print('pred_arrays.shape', pred_array.shape)
            del pred_arrays
            pert_rows = defaultdict(list)
            
            for i in range(pred_array.shape[0]):
                if method == 'scLong':
                    pert = tuple(np.where(pred_array[i, :, 0] > 0.5)[0])
                    if len(pert) == 0:
                        continue
                    if len(pert) > 2:
                        raise ValueError(f'len(pert) > 2! {pert}')
                    pert_rows[pert].append(pred_array[i, :, 1:])
                elif method == 'GEARS':
                    pert_old = tuple(pred_array[i, :2, 0])
                    if len(pert_old) == 0:
                        continue
                    if len(pert_old) > 2:
                        raise ValueError(f'len(pert) > 2! {pert_old}')
                    pert_new = []
                    for pi in pert_old:
                        if int(pi) >= 1:
                            gi = essential_pert[int(pi) - 1]
                            pert_new.append(symbol_to_idx[gi])
                    pert_new.sort()
                    pert_rows[tuple(pert_new)].append(pred_array[i, :, 1:])
            del pred_array

            pert_mean_dict = {}
            for pert, rows in pert_rows.items():
                mean_row = np.mean(np.array(rows), axis=0) #pert_num_cells, N, 2 -> N, 2
                differ_row = np.zeros_like(mean_row)
                differ_row[:, 0] = mean_row[:, 1] - mean_row[:, 0] #y_true - y_pred
                differ_row[:, 1] = mean_row[:, 1] - ctrl  #y_true - ctrl
                pert_mean_dict[pert] = differ_row # N, 2 
                #print('pert:', pert, np.mean(differ_row[:, 0] ** 2), np.mean(differ_row[:, 1] ** 2))
                #print('pert:', pert, np.mean(differ_row[:, 0] ** 2)/np.mean(differ_row[:, 1] ** 2))
            pert_mean_dicts.append(pert_mean_dict)

            if method == 'GEARS' and r == 0:
                perts_test = list(pert_mean_dict.keys())
                unseen_genes = []
                for pert in perts_test:
                    if len(pert) == 1:
                        perts_0_1.append(pert)
                        unseen_genes.append(pert[0])
                for pert in perts_test:
                    if len(pert) == 2:
                        if len(np.intersect1d(pert, unseen_genes)) == 2:
                            perts_0_2.append(pert)
                        elif len(np.intersect1d(pert, unseen_genes)) == 1:
                            perts_1_2.append(pert)
                        elif len(np.intersect1d(pert, unseen_genes)) == 0:
                            perts_2_2.append(pert)
                print(f'Scenarios: 0/1, 0/2, 1/2, 2/2: {len(perts_0_1)}, {len(perts_0_2)}, {len(perts_1_2)}, {len(perts_2_2)}')
        method_pert_mean_dicts_dict[method] = pert_mean_dicts


    scenarios = {'seen_0_1': perts_0_1, 'seen_0_2': perts_0_2, 'seen_1_2': perts_1_2, 'seen_2_2': perts_2_2, 'all2': perts_0_2 + perts_1_2 + perts_2_2}

    def get_pert_vec_errors(pert, DE):
        pert_0 = ad.var.gene_name.tolist()[pert[0]]
        if len(pert) == 2:
            pert_1 = ad.var.gene_name.tolist()[pert[1]]
        else:
            pert_1 = 'ctrl'
        pert_names = [f'{pert_0}+{pert_1}', f'{pert_1}+{pert_0}']
        mat = []
        for pert_name in pert_names:
            mat.append(ad[ad.obs.condition == pert_name].X.toarray()[:, DE])
        mat = np.concatenate(mat, axis = 0)
        return mat.mean(0) - ctrl[DE] #, mat.std(0)

    ################
    # create GI figures for seen_2_2. for only our method
    ################
    lut = {
    'seen 0/2': 'deeppink',
    'seen 1/2': 'darkblue',
    'seen 2/2': 'darkorange',
    }
    #colors_list = [lut['seen 0/2'] for _ in perts_0_2] + [lut['seen 1/2'] for _ in perts_1_2] + [lut['seen 2/2'] for _ in perts_2_2]
    colors_list = ['darkorange'] * len(scenarios['all2'])
    for name, scenario in scenarios.items():
        if name != 'all2':
            continue
        print(f'>> Sceanario: {name};')
        magnitude_vecs = {'Ground Truth': [],
                        'GEARS': [],
                        'scLong': []}
        for pert in scenario:
            DE = DE_dict[pert]
            genes = np.array([ad.var.gene_name.tolist()[i] for i in DE])
            pert_0 = ad.var.gene_name.tolist()[pert[0]]
            pert_1 = ad.var.gene_name.tolist()[pert[1]]
            pert_name = f'{pert_0}+{pert_1}'

            true_change_vec_multi= get_pert_vec_errors(pert, DE)
            true_change_vec_0= get_pert_vec_errors((pert[0], ), DE)
            true_change_vec_1= get_pert_vec_errors((pert[1], ), DE)

            X_true = np.array((true_change_vec_0, true_change_vec_1, np.ones(DE_num))).T #DE, 3
            reg_vec_true = np.linalg.inv((X_true.T @ X_true)) @ X_true.T @ true_change_vec_multi
            assert reg_vec_true.shape[0] == 3
            magnitude_vecs['Ground Truth'].append(np.sqrt(reg_vec_true[0]**2 + reg_vec_true[1]**2))
            
            for method in methods:
                ours_change_vec = 0
                for r in range(REPEAT):
                    ours_change_vec += method_pert_mean_dicts_dict[method][r][pert][DE, 1] - method_pert_mean_dicts_dict[method][r][pert][DE, 0]
                ours_change_vec /= REPEAT
                reg_vec = np.linalg.inv((X_true.T @ X_true)) @ X_true.T @ ours_change_vec
                magnitude_vecs[method].append(np.sqrt(reg_vec[0]**2 + reg_vec[1]**2))

        perts_dict = {'synergy': {'Ground Truth': [], 'GEARS': [], 'scLong': []}, 
                      'suppressor': {'Ground Truth': [], 'GEARS': [], 'scLong': []}}
        
        for k, v in magnitude_vecs.items():
            v = np.array(v)
            magnitude_vecs[k] = v
            argv = np.argsort(v)
            perts_dict['synergy'][k] = list(argv[-15:])
            perts_dict['suppressor'][k] = list(argv[:15])
        
        with open(f'yaml/synergy_suppressor_de_{DE_num}.yaml', 'w') as file:
            yaml.dump(perts_dict, file, default_flow_style=False, sort_keys=False, indent=4)

        fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.8))

        # 设置不同颜色
        colors = ['#CC99FF', '#FFCC99', '#66B2FF']

        # 绘制第一个子图 (synergy)
        venn3(
            [set(perts_dict['synergy']['scLong']), 
            set(perts_dict['synergy']['GEARS']), 
            set(perts_dict['synergy']['Ground Truth'])],
            set_labels=('scLong', 'GEARS', 'Ground Truth'),
            set_colors=colors,
            ax=axes[0]
        )
        axes[0].set_title('Synergy')

        # 绘制第二个子图 (suppressor)
        venn3(
            [set(perts_dict['suppressor']['scLong']), 
            set(perts_dict['suppressor']['GEARS']), 
            set(perts_dict['suppressor']['Ground Truth'])],
            set_labels=('scLong', 'GEARS', 'Ground Truth'),
            set_colors=colors,
            ax=axes[1]
        )
        axes[1].set_title('Suppressor')

        plt.tight_layout()
        plt.savefig(f'figs/synergy_suppresor_venn3_de_{DE_num}.svg', format='svg')

        plt.close()

        # 创建两个子图
        fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.8))

        # 设置相同的 x 和 y 轴范围（保证为方形）
        x_limits = [0, 3.8]
        y_limits = x_limits  # 保证 x 和 y 范围相同

        # 绘制第一张散点图 (Ground Truth vs GEARS)
        axes[0].scatter(magnitude_vecs['Ground Truth'], magnitude_vecs['GEARS'], c='#FFCC99', marker = 'x', s=plt.rcParams['lines.markersize'] ** 2/2)
        axes[0].plot(x_limits, y_limits, 'r--')  # 45度红色虚线
        axes[0].set_xlabel('Ground Truth', fontsize = 9)
        axes[0].set_ylabel('GEARS', fontsize = 9)
        axes[0].set_xticks(ticks=1 * np.arange(4))
        axes[0].set_yticks(ticks=1 * np.arange(4))

        axes[0].spines['right'].set_visible(False)
        axes[0].spines['top'].set_visible(False)

        # 计算 Pearson 相关系数并在右下角显示
        pearson_corr, _ = pearsonr(magnitude_vecs['Ground Truth'], magnitude_vecs['GEARS'])
        axes[0].text(0.95, 0.05, f'Pearson: {pearson_corr:.2f}', 
                    horizontalalignment='right', verticalalignment='bottom',
                    transform=axes[0].transAxes, fontsize =7)

        # 设置相同的x和y轴范围
        axes[0].set_xlim(x_limits)
        axes[0].set_ylim(y_limits)
        axes[0].set_aspect('equal', adjustable='box')  # 设置方形图

        # 绘制第二张散点图 (Ground Truth vs ours)
        axes[1].scatter(magnitude_vecs['Ground Truth'], magnitude_vecs['scLong'], c='#CC99FF', marker = 'x',s=plt.rcParams['lines.markersize'] ** 2/2)
        axes[1].plot(x_limits, y_limits, 'r--')  # 45度红色虚线
        axes[1].set_xlabel('Ground Truth', fontsize = 9)
        axes[1].set_ylabel('scLong', fontsize = 9)
        axes[1].set_xticks(ticks=1 * np.arange(4))
        axes[1].set_yticks(ticks=1 * np.arange(4))

        axes[1].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)

        # 计算 Pearson 相关系数并在右下角显示
        pearson_corr, _ = pearsonr(magnitude_vecs['Ground Truth'], magnitude_vecs['scLong'])
        axes[1].text(0.95, 0.05, f'Pearson: {pearson_corr:.2f}', 
                    horizontalalignment='right', verticalalignment='bottom',
                    transform=axes[1].transAxes, fontsize = 7)

        # 设置相同的x和y轴范围
        axes[1].set_xlim(x_limits)
        axes[1].set_ylim(y_limits)
        axes[1].set_aspect('equal', adjustable='box')  # 设置方形图

        plt.title('Magnitude')
        # 保存图像
        plt.tight_layout()
        plt.savefig(f'figs/magnitude_scatter_plots_de_{DE_num}.svg', format='svg')



                

import re
from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy

import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import yaml

# 设置 matplotlib 使用 TrueType 字体 (Type 42) 而不是 Type 3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 设置全局字体，可以选择其他常见的字体，比如 Arial
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或者 'Arial'


color_dict = {'GEARS': '#FFCC99',
              'Geneformer': '#99CC99',
              'LongSC': '#CC99FF'}

with open('norman_mse_pod_pearson.yml', 'r', encoding='utf-8') as file:
    value_dict = yaml.load(file, Loader=yaml.FullLoader)


# Plotting     
models = ['GEARS', 'Geneformer', 'LongSC']

fig, axs = plt.subplots(1,3)
metric_to_plot = list(value_dict.keys())
indices = np.arange(4)
fig.set_size_inches(12.3, 3.3)
scenario_names = ['Seen 0/2', 'Seen 1/2', 'Seen 2/2', 'Seen 0/1']
for i, metric in enumerate(metric_to_plot):
    ax = axs[i]
    width = 1 / (len(models) + 1)
    for model_idx, model in enumerate(models):
        values = np.array(value_dict[metric][model])[[0, 2, 4, 6]]
        if i == 1:
            values = 1-values
        errors = np.array(value_dict[metric][model])[[1, 3, 5, 7]]
        label = model
        ax.bar(indices + model_idx * width, values, yerr=errors, label=label, width=width, color= color_dict[model]) #, color=colors[model_idx % len(colors)])
    
    if i == 1:
        ax.set_ylim(0,0.15)
    elif i == 2:
        ax.set_ylim(0,1)

    ax.set_xlabel('Scenario', fontsize = 13)
    ax.set_ylabel(f'{metric}', fontsize = 13)
    #plt.title(f'Histogram of {column} by Model')
    ax.set_xticks(ticks=indices + width * (len(models) - 1) /2)
    ax.set_xticklabels(scenario_names, fontsize=11)
    if i == 1:
        ax.legend(loc = 'upper left', fontsize=8.5,frameon=False)
    else:
        ax.legend(loc = 'upper right', fontsize=8.5,frameon=False)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig.subplots_adjust(hspace=0, wspace=0.1)


plt.title('Genetic perturbtion effects prediction')

fig.tight_layout()
fig.savefig(f'figs/norman_split_1_mse_pod_pearson.pdf', dpi = 300)
fig = None
plt.close()

'''
for i, metric in enumerate(metric_to_plot):
    if i < 2:
        continue
    width = 1 / (len(models) + 1)
    fig = plt.figure(figsize=(4, 3))
    ax = brokenaxes(ylims=((0, 0.05), (.45, 1)), hspace=.05)
    for model_idx, model in enumerate(models):
        values = np.array(value_dict[metric][model])[[0, 2, 4, 6]]
        errors = np.array(value_dict[metric][model])[[1, 3, 5, 7]]
        label = model
        ax.bar(indices + model_idx * width, values, yerr=errors, label=label, width=width, color= color_dict[model])
    ax.set_xlabel('Scenario', fontsize = 9)
    ax.set_ylabel(f'{metric}', fontsize = 9)
    #plt.title(f'Histogram of {column} by Model')
    #ax.set_xticks(ticks=np.array((0,2)) + 1)
    #ax.set_xticklabels(scenario_names, fontsize=9)
    #ax.legend(loc = 'upper right', fontsize=9)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)

    fig.tight_layout()
    fig.savefig(f'figs/norman_split_1_pearson.pdf', dpi = 300)
    fig = None'''

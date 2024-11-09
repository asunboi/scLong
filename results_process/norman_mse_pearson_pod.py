import re
from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy

import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

# 设置 matplotlib 使用 TrueType 字体 (Type 42) 而不是 Type 3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 设置全局字体，可以选择其他常见的字体，比如 Arial
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或者 'Arial'


color_dict = {'GEARS': '#FFCC99',
              'Geneformer': '#99CC99',
              'LongSC': '#CC99FF'}

value_dict = {'MSE of the top 20 DE genes': 
                {'GEARS': [0.21884186200000003,0.010856425099045082,0.22105025,0.006529477619114716,0.130274668,0.006765251882646793,0.191461622,0.005225890877201316],
                 'Geneformer': [0.18528493249999997,0.01591112596831911,0.23708378749999998,0.013132003336299023,0.1957637875,0.015098965990118958,0.16920258500000002,0.00396859953543124],
                 'LongSC': [0.17019435000000003,0.0037425625085583647,0.20414560999999998,0.005494611930955635,0.123279192,0.011722206605772136,0.16726231,0.0019722212270601486]
                 },
                 'Percentage of non-zero\ngenes with wrong direction': 
              {'GEARS': [0.9311803405572757,0.0025000000000000356,0.8931790863185368,0.0030121567803414943,0.9212047275050372,0.007520767828777525,0.8879335561485053,0.003901713529147037],
                 'Geneformer': [0.9418666150670795,0.0029462782549439376,0.8931523763303618,0.005326742745138329,0.8929808180582174,0.005756266300285938,0.9120339795837363,0.0011474923381844795],
                 'LongSC': [0.9493227554179566,0.0032894736842105643,0.9150166389667245,0.0007898670853186962,0.9247616590495848,0.005733696987566983,0.9179555357905647,0.005343716433941992]
                 },
            'Pearson of expression changes\nof the top 20 DE genes ': 
              {'GEARS': [0.8615294511325319,0.00983198610899352,0.8191908940935507,0.01221995350243661,0.8551728915820289,0.014845483136017811,0.5611978784087428,0.01333858337738759],
                 'Geneformer': [0.8599235769346047,0.004789189248844506,0.7998738122246304,0.00837225050816954,0.8056603630180477,0.034267342971927815,0.575546061929573,0.012806097736365357], 
                 'LongSC': [0.8935746489962148,0.004593198752148586,0.834333088421406,0.0005212094538510348,0.8845535109799564,0.0076168742588116345,0.625385517271012,0.005463243694292741]
                 },
              }


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

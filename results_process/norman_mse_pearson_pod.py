import re
from collections import defaultdict
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import yaml




# Set Arial globally in Matplotlib configuration
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['lines.linewidth'] = 0.5


color_dict = {'GEARS': '#FFCC99',
              'Geneformer': '#99CC99',
              'scGPT': '#FF9999',
              'scFoundation': '#D1B26F',
              'scLong': '#CC99FF'}

with open('norman_mse_pod_pearson.yml', 'r', encoding='utf-8') as file:
    value_dict = yaml.load(file, Loader=yaml.FullLoader)

z = np.random.normal()

# Plotting     
models = ['GEARS', 'Geneformer', 'scGPT', 'scFoundation', 'scLong']

fig, axs = plt.subplots(1,2)
metric_to_plot = list(value_dict.keys())
indices = np.arange(4)
fig.set_size_inches(7.2, 1.8)
scenario_names = ['Seen 0/2', 'Seen 1/2', 'Seen 2/2', 'Seen 0/1']
for i, metric in enumerate(metric_to_plot):
    ax = axs[i]
    width = 1 / (len(models) + 1)
    for model_idx, model in enumerate(models):
        values = np.array(value_dict[metric][model])[[0, 2, 4, 6]]
        errors = np.array(value_dict[metric][model])[[1, 3, 5, 7]]
        label = model
        ax.bar(indices + model_idx * width, values, yerr=errors, label=label, width=width, color= color_dict[model]) #, color=colors[model_idx % len(colors)])
    if i == 0:
        ax.set_ylim(0.1,0.26)
    if i == 1:
        ax.set_ylim(0.5,0.92)

    #ax.set_xlabel('Scenario', fontsize = 11.5)
    ax.set_ylabel(f'{metric}')
    if i == 0:
        ax.set_yticks(ticks = np.arange(2, 6) * 0.05)
    if i == 1:
        ax.set_yticks(ticks = np.arange(5, 10) * 0.1)
    ax.set_xticks(ticks=indices + width * (len(models) - 1) /2)
    ax.set_xticklabels(scenario_names)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig.subplots_adjust(hspace=0, wspace=0.1)

fig.tight_layout()
fig.savefig(f'figs/norman_split_1_mse_pearson.svg', format='svg')
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
    fig.savefig(f'figs/norman_split_1_pearson.svg', dpi = 300)
    fig = None'''

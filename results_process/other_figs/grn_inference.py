import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import yaml


plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['lines.linewidth'] = 0.5

color_dict = {'DeepSEM': '#FFCC99',
              'Geneformer': '#99CC99',
              'scLong': '#CC99FF'}

with open('grn_inference_dict.yml', 'r', encoding='utf-8') as file:
    grn_inference_dict = yaml.load(file, Loader=yaml.FullLoader)
data1 = np.array(grn_inference_dict['data1']).reshape((-1, 1))
data2 = np.array(grn_inference_dict['data2']).reshape((-1, 1))
error1 = np.array(grn_inference_dict['error1']).reshape((-1, 1))
error2 = np.array(grn_inference_dict['error2']).reshape((-1, 1))



models = ['DeepSEM', 'Geneformer',  'scLong']
colors = [color_dict[model] for model in models]
 
width = 1 / (len(models) + 1)

fig, axes = plt.subplots(1,2)
fig.set_size_inches(7.2, 2.4)

for i in range(2):
    ax = axes[i]
    indices = np.array([0, 2, 4])
    values = [data2, data1][i]
    errors = [error2, error1][i]
    for model_idx, model in enumerate(models):
        ax.bar([(indices * width)[model_idx]], values[model_idx], yerr=errors[model_idx], width=width, color = colors[model_idx]) #, color=colors[model_idx % len(colors)])
    if i == 0:
        ax.set_ylim((1, 1.4))
    elif i == 1:
        ax.set_ylim((1.1, 1.25))
        ax.set_yticks(ticks = [1.10, 1.15, 1.20, 1.25])
    ax.set_ylabel(['AUPR', 'EPR'][i], fontsize = 9)
    #plt.title(f'Histogram of {column} by Model')
    ax.set_xticks(ticks=indices * width)
    ax.set_xticklabels(models, fontsize=9)

    #ax.legend(loc = 'upper right', fontsize=10,frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #fig.subplots_adjust(hspace=0, wspace=0.1)
#plt.title('Gene regulartory network inference', fontsize = 9)
fig.savefig(f'figs/grn_inference_bar.svg',format="svg")
fig = None
plt.close()
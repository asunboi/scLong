import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import pickle

plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['lines.linewidth'] = 0.5


title = "Gene Profile Prediction"
dataset = "DeepCE"


color_dict = {'DeepCE': '#FFCC99',
              'Geneformer': '#99CC99',
              'scGPT': '#FF9999',
              'scFoundation': '#D1B26F',
              'UCE': '#99FFCC',
              'scLong': '#CC99FF'}

with open('gene_profile_dict.pkl', 'rb') as file:
    gene_profile_dict = pickle.load(file)

data1 = np.array(gene_profile_dict['data1'])
std_err1 = np.array(gene_profile_dict['std_err1'])

data2 = np.array(gene_profile_dict['data2'])
std_err2 = np.array(gene_profile_dict['std_err2'])

data3 = np.array(gene_profile_dict['data3'])
std_err3 = np.array(gene_profile_dict['std_err3'])

data4 = np.array(gene_profile_dict['data4'])
std_err4 = np.array(gene_profile_dict['std_err4'])

data5 = np.array(gene_profile_dict['data5'])
std_err5 = np.array(gene_profile_dict['std_err5'])


y1 = np.mean(data1, axis=1)
y2 = np.mean(data2, axis=1)
y3 = np.mean(data3, axis=1)
y4 = np.mean(data4, axis=1)
y5 = np.mean(data5, axis=1)

#ys = [y1, y2, y3, y4, y5]
ys = [y5, y1, y2, y3, y4]




#std_errs = [std_err1, std_err2, std_err3, std_err4, std_err5]
std_errs = [std_err5, std_err1, std_err2, std_err3, std_err4]

models = ['DeepCE', 'Geneformer', 'scGPT', 'scFoundation', 'UCE', 'scLong']


#######################
#, 'Spearman', 'Pearson'


fig = plt.figure(figsize=(7.2, 1.8))  # 设置整体图形大小
gs = fig.add_gridspec(1, 8)
axs = [fig.add_subplot(gs[0, 0:2]),
       fig.add_subplot(gs[0, 2:5]),
       fig.add_subplot(gs[0, 5:8])]

#scenario_names_sets = [["Spearman", "Pearson"], ['Pos-P@100', 'Neg-P@100'], ['Root mean squared error']]
scenario_names_sets = [['Root mean squared error'], ["Spearman", "Pearson"], ['Pos-P@100', 'Neg-P@100'], ]
idxes = [[0,1],[1, 3], [3, 5]]

for ax_i, ax in enumerate(axs):
    width = 1 / (len(models) + 1)
    if ax_i != 0:
        indices = np.arange(2)
    else:
        indices = np.arange(1)

    start = idxes[ax_i][0]
    end = idxes[ax_i][1]

    for model_idx, model in enumerate(models):
        values = np.array([y[model_idx] for y in ys[start: end]])
        errors = np.array([std_err[model_idx] for std_err in std_errs[start: end]])
        label = model
        ax.bar(indices + model_idx * width, values, yerr=errors, label=label, width=width, color = color_dict[model]) #, color=colors[model_idx % len(colors)])
        #for idx in indices:
        #    ax.text(idx + model_idx * width-width/3, values[idx] + errors[idx] +0.002, f"{values[idx]:.4f}", color='black', size=7)

    scenario_names = scenario_names_sets[ax_i]
    if ax_i == 1:
        ax.set_ylim((0.4, 0.48))
        ax.set_yticks(ticks = np.arange(20, 25)*0.02)
    elif ax_i == 2:
        ax.set_ylim((0.2, 0.32))
        ax.set_yticks(ticks = np.arange(10, 17)*0.02)
    else:
        ax.set_ylim((1.7, 1.78))
        ax.set_yticks(ticks = np.arange(85, 90)*0.02)

    ax.set_xticks(ticks=indices + width * (len(models) - 1) /2)
    ax.set_xticklabels(scenario_names, fontsize=9)

    #ax.legend(loc = 'upper left', fontsize=9,frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #fig.subplots_adjust(hspace=0, wspace=0.1)

#ax.set_title('Gene profile prediction', fontsize = 18)
fig.tight_layout()
fig.savefig(f'figs/gene_profile_all.svg', format='svg')
fig = None
plt.close()





'''

#######################
#, 'POS-P@100', 'NEG-P@100'


fig, ax = plt.subplots()
indices = np.arange(2)
fig.set_size_inches(5, 4)
scenario_names = ['POS-P@100', 'NEG-P@100']

width = 1 / (len(models) + 1)
for model_idx, model in enumerate(models):
    values = np.array([y[model_idx] for y in ys[2:4]])
    errors = np.array([std_err[model_idx] for std_err in std_errs[2:4]])
    label = model
    ax.bar(indices + model_idx * width, values, yerr=errors, label=label, width=width, color = color_dict[model]) #, color=colors[model_idx % len(colors)])
    for idx in indices:
        ax.text(idx + model_idx * width-width/3, values[idx] + errors[idx] +0.002, f"{values[idx]:.4f}", color='black', size=7)

ax.set_ylim((0.2, 0.32))
ax.set_xlabel('Metric', fontsize = 13)
ax.set_ylabel(f'P@100 score', fontsize = 13)
#plt.title(f'Histogram of {column} by Model')
ax.set_xticks(ticks=indices + width * (len(models) - 1) /2)
ax.set_xticklabels(scenario_names, fontsize=12)

ax.legend(loc = 'upper left', fontsize=9,frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#fig.subplots_adjust(hspace=0, wspace=0.1)

#ax.set_title('Gene profile prediction', fontsize = 18)
fig.tight_layout()
fig.savefig(f'gene_profile_pos_neg.svg', format='svg')
fig = None
plt.close()


#######################
#, 'RMSE'


fig, ax = plt.subplots()
indices = np.arange(1)
fig.set_size_inches(3, 4)
scenario_names = ['RMSE']

width = 1 / (len(models) + 1)
for model_idx, model in enumerate(models):
    values = np.array([y[model_idx] for y in ys[4:]])
    errors = np.array([std_err[model_idx] for std_err in std_errs[4:]])
    label = model
    ax.bar(indices + model_idx * width, values, yerr=errors, label=label, width=width, color = color_dict[model]) #, color=colors[model_idx % len(colors)])
    for idx in indices:
        ax.text(idx + model_idx * width-width/3, values[idx] + errors[idx] +0.002, f"{values[idx]:.4f}", color='black', size=7)

ax.set_ylim((1.7, 1.8))
ax.set_xlabel('Metric', fontsize = 13)
ax.set_ylabel(f'Root mean squared error ', fontsize = 13)
#plt.title(f'Histogram of {column} by Model')
ax.set_xticks(ticks=indices + width * (len(models) - 1) /2)
ax.set_xticklabels(scenario_names, fontsize=12)

ax.legend(loc = 'upper left', fontsize=9, frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#fig.subplots_adjust(hspace=0, wspace=0.1)

#ax.set_title('Gene profile prediction', fontsize = 18)
fig.tight_layout()
fig.savefig(f'gene_profile_rmse.pdf', dpi = 300)
fig = None
plt.close()'''
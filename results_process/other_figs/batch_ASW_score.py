import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import scib
import scanpy as sc
import pickle


plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 7
plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['lines.linewidth'] = 0.5

color_dict = {
    'Raw': '#6FD1B2',
    'HVG': '#FF99CC',
    'scVI': '#FFCC99',
              'Geneformer': '#99CC99',
              'scGPT': '#FF9999',
              'scFoundation': '#D1B26F',
              'UCE': '#99FFCC',
              'scLong': '#CC99FF'}

with open('batch_integration_dict.pkl', 'rb') as file:
    batch_integration_dict = pickle.load(file)

print(batch_integration_dict)

#batchASW
data1 = batch_integration_dict['data1']


models = ['Raw', 'HVG', 'scVI', 'Geneformer',  'scGPT', 'scFoundation', 'UCE', 'scLong']
colors = [color_dict[model] for model in models]
 
width = 1 / (len(models) + 1)

fig, ax = plt.subplots(1,1)
fig.set_size_inches(7.5, 3)

indices = np.arange(len(models))
values = data1.reshape((-1,))
for model_idx, model in enumerate(models):
    ax.bar([(indices * width)[model_idx]], values[model_idx], width=2*width/3, color = colors[model_idx]) #, color=colors[model_idx % len(colors)])
ax.set_ylim((0.6, 1))
ax.set_yticks(ticks = [0.6,0.7,0.8,0.9,1.0], labels = ['0.6','0.7','0.8','0.9','1.0'], fontsize=9)
ax.set_ylabel('batch ASW', fontsize=11)
#plt.title(f'Histogram of {column} by Model')
ax.set_xticks(ticks=indices * width)
ax.set_xticklabels(models, fontsize=9)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig.savefig(f'figs/batch_ASW.svg',format="svg")
fig = None
plt.close()
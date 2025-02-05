import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import yaml


# 设置 matplotlib 使用 TrueType 字体 (Type 42) 而不是 Type 3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 设置全局字体，可以选择其他常见的字体，比如 Arial
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或者 'Arial'

color_dict = {'DeepDDS': '#FFCC99',
              'Geneformer': '#99CC99',
              'LongSC': '#CC99FF'}

with open('drug_comb_dict.yml', 'r', encoding='utf-8') as file:
    drug_comb_dict = yaml.load(file, Loader=yaml.FullLoader)
data1 = np.array(drug_comb_dict['data1']).reshape((-1, 1))
errors = np.array(drug_comb_dict['errors']).reshape((-1, 1))


bar_width=0.3
models = ['DeepDDS', 'Geneformer',  'LongSC']
colors = [color_dict[model] for model in models]
 
fig, ax = plt.subplots()
indices = np.array([0, 2, 4])
fig.set_size_inches(4, 3)

width = 1 / (len(models) + 1)
values = data1
for model_idx, model in enumerate(models):
    ax.bar([(indices * width)[model_idx]], values[model_idx], yerr = errors[model_idx], width=width, color = colors[model_idx]) #, color=colors[model_idx % len(colors)])
    ax.text((indices * width)[model_idx]-width/3, values[model_idx] + errors[model_idx] +0.002, f"{values[model_idx][0]:.3f}", color='black', size=10)
ax.set_ylim((0.55, 0.68))
ax.set_yticks(ticks = [0.55, 0.6, 0.65])
ax.set_xlabel('Method', fontsize = 14)
ax.set_ylabel(f'AUROC', fontsize = 14)
#plt.title(f'Histogram of {column} by Model')
ax.set_xticks(ticks=indices * width)
ax.set_xticklabels(models, fontsize=12)

#ax.legend(loc = 'upper right', fontsize=10,frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#fig.subplots_adjust(hspace=0, wspace=0.1)

ax.set_title('Drug combination prediction', fontsize = 16)
fig.tight_layout()
fig.savefig(f'drug_combination_prediction.pdf',format="pdf", bbox_inches="tight",dpi = 300)
fig = None
plt.close()

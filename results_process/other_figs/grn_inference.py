import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np

# 设置 matplotlib 使用 TrueType 字体 (Type 42) 而不是 Type 3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 设置全局字体，可以选择其他常见的字体，比如 Arial
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或者 'Arial'

color_dict = {'DeepSEM': '#FFCC99',
              'Geneformer': '#99CC99',
              'LongSC': '#CC99FF'}


data1 = np.array([
    [1.15],
    [1.18],
    [1.21]
])
data2 = np.array([
    [1.11],
    [1.12],
    [1.35]
])

palette = pyplot.get_cmap('Set1')
models = ['DeepSEM', 'Geneformer',  'LongSC']
colors = [color_dict[model] for model in models]
 
width = 1 / (len(models) + 1)

fig, axes = plt.subplots(1,2)
fig.set_size_inches(8, 3)

for i in range(2):
    ax = axes[i]
    indices = np.array([0, 2, 4])
    values = [data1, data2][i]
    for model_idx, model in enumerate(models):
        ax.bar([(indices * width)[model_idx]], values[model_idx], width=width, color = colors[model_idx]) #, color=colors[model_idx % len(colors)])
        ax.text((indices * width)[model_idx]-width/3, 
                    values[model_idx]+0.003, 
                    f"{values[model_idx][0]:.3f}", 
                    color='black', size=10)
    ax.set_ylim((1, 1.4))
    ax.set_xlabel('Method', fontsize = 14)
    ax.set_ylabel(['EPR', 'AUPR'][i], fontsize = 14)
    #plt.title(f'Histogram of {column} by Model')
    ax.set_xticks(ticks=indices * width)
    ax.set_xticklabels(models, fontsize=12)

    #ax.legend(loc = 'upper right', fontsize=10,frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #fig.subplots_adjust(hspace=0, wspace=0.1)
plt.title('GRN inference', fontsize = 16)
fig.tight_layout()
fig.savefig(f'grn_inference_bar.pdf',format="pdf", bbox_inches="tight",dpi = 300)
fig = None
plt.close()
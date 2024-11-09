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

color_dict = {'DeepCDR': '#FFCC99',
              'Geneformer': '#99CC99',
              'LongSC': '#CC99FF'}

data1 = np.array([
    [0.8371],
    [0.8516],
    [0.8732]
])

ys = [data1]

palette = pyplot.get_cmap('Set1')
models = ['DeepCDR', 'Geneformer',  'LongSC']
colors = [color_dict[model] for model in models]
 
fig, ax = plt.subplots()
indices = np.array([0, 2, 4])
fig.set_size_inches(4, 3)

width = 1 / (len(models) + 1)
values = data1
for model_idx, model in enumerate(models):
    ax.bar([(indices * width)[model_idx]], values[model_idx], width=width, color = colors[model_idx]) #, color=colors[model_idx % len(colors)])
    ax.text((indices * width)[model_idx]-width/3, 
                 values[model_idx]+0.002, 
                 f"{values[model_idx][0]:.3f}", 
                  color='black', size=10)
ax.set_ylim((0.8, 0.9))
ax.set_xlabel('Method', fontsize = 14)
ax.set_ylabel(f'Pearson score', fontsize = 14)
#plt.title(f'Histogram of {column} by Model')
ax.set_xticks(ticks=indices * width)
ax.set_xticklabels(models, fontsize=12)

#ax.legend(loc = 'upper right', fontsize=10,frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#fig.subplots_adjust(hspace=0, wspace=0.1)

ax.set_title('Drug response prediction', fontsize = 16)
fig.tight_layout()
fig.savefig(f'drug_response_prediction.pdf',format="pdf", bbox_inches="tight",dpi = 300)
fig = None
plt.close()
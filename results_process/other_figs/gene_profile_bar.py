import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np

# 设置 matplotlib 使用 TrueType 字体 (Type 42) 而不是 Type 3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 设置全局字体，可以选择其他常见的字体，比如 Arial
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或者 'Arial'


title = "Gene Profile Prediction"
dataset = "DeepCE"


color_dict = {'DeepCE': '#FFCC99',
              'Geneformer': '#99CC99',
              'LongSC': '#CC99FF'}

data1 = np.array([
    [0.4439],
    [0.4496],
    [0.4538]
])
data2 = np.array([
    [0.4423],
    [0.4499],
    [0.4536]
])
data3 = np.array([
    [0.2557],
    [0.2506],
    [0.2578]
])
data4 = np.array([
    [0.295],
    [0.2913],
    [0.2961]
])
my_y_ticks = [np.linspace(0.4, 0.5, num=3),np.linspace(0.4, 0.5, num=3),np.linspace(0.2, 0.3, num=3),np.linspace(0.25, 0.35, num=3)]
ymin = [0.4,0.4,0.2,0.25]

y1 = np.mean(data1, axis=1)
y2 = np.mean(data2, axis=1)
y3 = np.mean(data3, axis=1)
y4 = np.mean(data4, axis=1)

ys = [y1, y2]

std_err1 = np.array([0.0309,0.00851,0.0082])
std_err2 = np.array([0.0316,0.00827,0.0085])
std_err3 = np.array([0.0266,0.0315,0.008])
std_err4 = np.array([0.02,0.0209,0.0087])

std_errs = [std_err1, std_err2]

print(y1.shape,std_err1.shape)
palette = pyplot.get_cmap('Set1')
tasks = ["Spearman", "Pearson", 'POS-P@100', 'NEG-P@100']
fig, ax = plt.subplots(1,len(tasks),figsize=(12, 3)) 
x=np.array([0])
error_params = dict(elinewidth=1,ecolor='black',capsize=0)
bar_width=0.3
models = ['DeepCE', 'Geneformer',  'LongSC']

colors = [
    (245/255.,198/255.,196/255.), 
    (194/255.,191/255.,2/255.),
    (87/255.,170/255.,62/255.),
    (178/255.,113/255.,171/255.)
]

fig, ax = plt.subplots()
indices = np.arange(2)
fig.set_size_inches(4, 5)
scenario_names = ["Spearman", "Pearson"] #, 'POS-P@100', 'NEG-P@100']

width = 1 / (len(models) + 1)
for model_idx, model in enumerate(models):
    values = np.array([y[model_idx] for y in ys])
    errors = np.array([std_err[model_idx] for std_err in std_errs])
    label = model
    ax.bar(indices + model_idx * width, values, yerr=errors, label=label, width=width, color = color_dict[model]) #, color=colors[model_idx % len(colors)])

ax.set_ylim((0.4, 0.5))
ax.set_xlabel('Metric', fontsize = 16)
ax.set_ylabel(f'Correlation score', fontsize = 16)
#plt.title(f'Histogram of {column} by Model')
ax.set_xticks(ticks=indices + width * (len(models) - 1) /2)
ax.set_xticklabels(scenario_names, fontsize=13)

ax.legend(loc = 'upper right', fontsize=10,frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#fig.subplots_adjust(hspace=0, wspace=0.1)

ax.set_title('Gene profile prediction', fontsize = 18)
fig.tight_layout()
fig.savefig(f'gene_profile.pdf', dpi = 300)
fig = None
plt.close()
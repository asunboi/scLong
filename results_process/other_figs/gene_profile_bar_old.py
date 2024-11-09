import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np

title = "Gene Profile Prediction"
dataset = "DeepCE"

data1 = np.array([
    [0.4439],
    [0.4417],
    [0.4538]
])
data2 = np.array([
    [0.4423],
    [0.4379],
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

std_err1 = np.array([0.0309,0.0301,0.0082])
std_err2 = np.array([0.0316,0.0218,0.0085])
std_err3 = np.array([0.0266,0.0315,0.008])
std_err4 = np.array([0.02,0.0209,0.0087])

print(y1.shape,std_err1.shape)
palette = pyplot.get_cmap('Set1')
tasks = ["Spearman", "Pearson", 'POS-P@100', 'NEG-P@100']
fig, ax = plt.subplots(1,2,figsize=(4, 3)) 
x=np.array([0])
error_params = dict(elinewidth=1,ecolor='black',capsize=0)
bar_width=0.3
label = ['DeepCE', 'Gene2Vec',  'LongSC']
tick_label = ['DeepCE', 'Gene2Vec',  'LongSC']

colors = ['#FFCC99','#99CC99','#CC99FF']
for i in range(len(tasks)):
    if i > 1:
        break
    ##### add the numeric results in every bar #####
    avg_list = [y1, y2, y3, y4]
    mean_list = [std_err1, std_err2, std_err3, std_err4]

    bar_x=[bar_width*j for j in range(len(label))]
    bar_y=[avg_list[i][j] for j in range(len(label))]
    bar_err=[mean_list[i][j] for j in range(len(label))]
    ax[i].bar(bar_x, bar_y,
            bar_width*2/3,
            color=colors,
            yerr=bar_err,
            error_kw=error_params)
    j=0
    for j in range(len(label)):

        ax[i].text(bar_width*j-0.08, 
                 avg_list[i][j]+mean_list[i][j]+0.005, 
                 f"{avg_list[i][j]:.3f}", 
                  color='black', size=8)
        
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ticks_loc=[x[0]+bar_width*i for i in range(len(tick_label))]
    ax[i].set_xticks(ticks_loc)
    ax[i].set_xticklabels(tick_label, fontsize=8)
    ax[i].set_yticks(my_y_ticks[i])
    ax[i].set_ylim(ymin=ymin[i])
    #ax[i].legend(loc='lower right', prop={'size':10})
    #plt.ylabel('MSE', size=10)

    if i == 2: 
        ax[i].set_title(f'{title}', size=12)
    ax[i].set_ylabel(f'{tasks[i]}', labelpad=-10, y=1.02, rotation=0, fontsize=8)
fig.tight_layout()
plt.savefig(f'{title}.pdf', format="pdf", transparent=True, bbox_inches="tight", pad_inches=0.1)
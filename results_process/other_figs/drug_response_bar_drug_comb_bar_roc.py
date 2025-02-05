import matplotlib.pyplot as plt
import numpy as np
import re
import yaml


plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 7
plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['lines.linewidth'] = 0.5


fig = plt.figure()  # 设置整体图形大小
gs = fig.add_gridspec(1, 14)
axs = [fig.add_subplot(gs[0, 0:5]),
       fig.add_subplot(gs[0, 6:9]),
       fig.add_subplot(gs[0, 10:14])]

fig.set_size_inches(8.4, 2.4)


with open('drug_and_drug_comb_dict.yml', 'r', encoding='utf-8') as file:
    drug_and_drug_comb_dict = yaml.load(file, Loader=yaml.FullLoader)

for ax_i, ax in enumerate(axs):
    if ax_i == 0:
        color_dict = {'DeepCDR': '#FFCC99',
                    'Geneformer': '#99CC99',
                    'scFoundation': '#D1B26F',
                    'scLong': '#CC99FF'}

        data1 = np.array(drug_and_drug_comb_dict['drug']['data1']).reshape((-1, 1))
        errors = np.array(drug_and_drug_comb_dict['drug']['errors']).reshape((-1, 1))



        models = ['DeepCDR', 'Geneformer', 'scFoundation', 'scLong']
        colors = [color_dict[model] for model in models]
        indices = np.array([0, 2, 4, 6])

        width = 1 / (len(models) + 1)
        values = data1
        for model_idx, model in enumerate(models):
            ax.bar([(indices * width)[model_idx]], values[model_idx], yerr=errors[model_idx], width=width, color = colors[model_idx]) #, color=colors[model_idx % len(colors)])
        ax.set_ylim((0.82, 0.88))
        ax.set_ylabel(f'Pearson correlation', fontsize = 7)
        ax.set_yticks(ticks=0.02 * np.arange(41, 45))
        ax.set_xticks(ticks=indices * width)
        ax.set_xticklabels(models, fontsize=7)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        #

        ax.set_title('Single drug response prediction', fontsize = 9)
        
    elif ax_i == 1:
                
        data1 = np.array(drug_and_drug_comb_dict['drug_comb']['data1']).reshape((-1, 1))
        errors = np.array(drug_and_drug_comb_dict['drug_comb']['errors']).reshape((-1, 1))

        color_dict = {'DeepDDS': '#FFCC99',
              'Geneformer': '#99CC99',
              'scLong': '#CC99FF'}


        models = ['DeepDDS', 'Geneformer',  'scLong']
        colors = [color_dict[model] for model in models]
        
        indices = np.array([0, 2, 4])

        width = 1 / (len(models) + 1)
        values = data1
        for model_idx, model in enumerate(models):
            ax.bar([(indices * width)[model_idx]], values[model_idx], yerr = errors[model_idx], width=width, color = colors[model_idx]) 
        ax.set_ylim((0.56, 0.66))
        ax.set_yticks(ticks = 0.02 * np.arange(28, 34))
        ax.set_ylabel(f'AUROC', fontsize = 7)
        ax.set_xticks(ticks=indices * width)
        ax.set_xticklabels(models, fontsize = 7)

        
        ax.set_title('Drug combination response prediction', fontsize = 9)

        #ax.legend(loc = 'upper right', fontsize=10,frameon=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


    else:
        
        # 读取txt文件
        with open('auc_drug_comb.txt', 'r') as file:
            data = file.read()
        data = data.replace('\n', ' ').replace('\t', ' ')

        # 使用正则表达式匹配所有括号中的数组
        matches = re.findall(r'\[(.*?)\]', data)

        # 将每个匹配的字符串转换为浮点数数组
        arrays = []
        for match in matches:
            # 使用split去除多余空格，提取小数并转换为浮点数
            array = [float(num) for num in match.split( )]
            arrays.append(np.array(array))

        # 输出提取的所有数组
        for idx, array in enumerate(arrays):
            print(f"Array {idx + 1}: {array.shape}")

        color_dict = {'DeepDDS': '#FFCC99',
                    'Geneformer': '#99CC99',
                    'scLong': '#CC99FF'}

        arrays_dict = {'DeepDDS': arrays[0:3],
                    'Geneformer': arrays[3:6],
                    'scLong': arrays[6:9]}
        
        ax.plot((0,1), (0,1), 'r--')  # 45度红色虚线


        # 遍历字典中的每个 method 和对应的 array
        for method, data in arrays_dict.items():
            fpr = data[0]  # 第一行为 FPR
            tpr = data[1]  # 第二行为 TPR
            # 绘制 ROC 曲线
            ax.plot(fpr, tpr, label=method, color = color_dict[method])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # 设置 x 和 y 轴标签
        ax.set_xlabel('False positive rate', fontsize = 7)
        ax.set_ylabel('True positive rate', fontsize = 7)

        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))

        ax.set_xticks(np.arange(0, 6)*0.2)
        ax.set_yticks(np.arange(0, 6)*0.2)

        # 设置标题

        # 添加图例
        ax.legend(loc = 'lower right', fontsize=7,frameon=False)
        ax.set_aspect('equal', adjustable='box')

fig.subplots_adjust(hspace=0, wspace=0.1)
#fig.tight_layout()
fig.savefig(f'figs/drug_and_drug_comb_res_all.svg', format='svg')
fig = None
plt.close()
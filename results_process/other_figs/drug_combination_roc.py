import re

import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np

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


# 设置 matplotlib 使用 TrueType 字体 (Type 42) 而不是 Type 3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 设置全局字体，可以选择其他常见的字体，比如 Arial
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或者 'Arial'

color_dict = {'DeepDDS': '#FFCC99',
              'Geneformer': '#99CC99',
              'LongSC': '#CC99FF'}

arrays_dict = {'DeepDDS': arrays[0:3],
               'Geneformer': arrays[3:6],
               'LongSC': arrays[6:9]}

# 创建 ROC 曲线图
fig, ax = plt.subplots()
fig.set_size_inches(4, 3)

# 遍历字典中的每个 method 和对应的 array
for method, data in arrays_dict.items():
    fpr = data[0]  # 第一行为 FPR
    tpr = data[1]  # 第二行为 TPR
    # 绘制 ROC 曲线
    ax.plot(fpr, tpr, label=method)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 设置 x 和 y 轴标签
ax.set_xlabel('False positive rate', fontsize = 14)
ax.set_ylabel('True positive rate', fontsize = 14)

# 设置标题
ax.set_title('Drug combination: ROC curve', fontsize = 16)

# 添加图例
ax.legend(loc = 'lower right', fontsize=12,frameon=False)

# 保存图片
fig.savefig('drug_combination_roc_curve.pdf',format="pdf", bbox_inches="tight",dpi = 300)


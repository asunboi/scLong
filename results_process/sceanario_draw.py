import scanpy as sc
from scipy import sparse
from scipy import stats
import numpy as np
import os
from collections import defaultdict
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from copy import deepcopy
import matplotlib.patches as mpatches

REPEAT = 2
# 设置 matplotlib 使用 TrueType 字体 (Type 42) 而不是 Type 3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 设置全局字体，可以选择其他常见的字体，比如 Arial
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或者 'Arial'

colors = {'seen 0/2': 'deeppink', 'seen 1/2': 'darkblue', 'seen 2/2': 'darkorange'} 
# Create a second legend for categories
category_patches = [mpatches.Patch(color=color, label=cat) for cat, color in colors.items()]
second_legend = plt.legend(handles=category_patches, loc='upper right', bbox_to_anchor=(0.006, -0.08), title = 'Scenarios', fontsize = 6, title_fontsize=6)
#plt.setp(second_legend.get_title(),fontsize=16)
plt.tight_layout()
plt.savefig(f'figs/sceanario_title_all2.pdf', dpi = 120)
plt.close()
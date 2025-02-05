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

plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['lines.linewidth'] = 0.5

colors = {'seen 0/2': 'deeppink', 'seen 1/2': 'darkblue', 'seen 2/2': 'darkorange'} 
# Create a second legend for categories
category_patches = [mpatches.Patch(color=color, label=cat) for cat, color in colors.items()]
second_legend = plt.legend(handles=category_patches, loc='upper right', bbox_to_anchor=(0.006, -0.08), title = 'Scenarios', fontsize = 7, title_fontsize=7)
#plt.setp(second_legend.get_title(),fontsize=16)
plt.tight_layout()
plt.savefig(f'figs/sceanario_title_all2.svg', format='svg')
plt.close()
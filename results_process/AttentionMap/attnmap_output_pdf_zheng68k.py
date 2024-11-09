# -*- coding: utf-8 -*-

#set +o noclobber

#python attnmap_output_pdf_zheng68k.py > res/attnmap_output_pdf_zheng68k.txt 2>&1

'''
{'CD8+ Cytotoxic T': 20307, 
 'CD8+/CD45RA+ Naive Cytotoxic': 16361, 
 'CD4+/CD45RO+ Memory': 3031, 
 'CD19+ B': 5579, 
 'CD4+/CD25 T Reg': 6116, 
 'CD56+ NK' : 8522, 
 'CD4+ T Helper2': 92, 
 'CD4+/CD45RA+/CD25- Naive T': 1857, 
 'CD34+': 188, 
 'Dendritic': 1946, 
 'CD14+ Monocyte': 1944}

celltypes = ['CD8+ Cytotoxic T', 'CD8+/CD45RA+ Naive Cytotoxic', 'CD4+/CD45RO+ Memory', 'CD19+ B', ]
celltypes = ['CD4+/CD25 T Reg', 'CD56+ NK', 'CD4+ T Helper2', 'CD4+/CD45RA+/CD25- Naive T', ]
celltypes = ['CD34+', 'Dendritic', 'CD14+ Monocyte']'''




celltype_makergenes = {'CD8+ Cytotoxic T': ['CD8A', 'CD8B', 'CD3D', 'CD3E', 'CD3G'], 
 'CD8+/CD45RA+ Naive Cytotoxic': ['CD8A', 'CD8B'], 
 'CD4+/CD45RO+ Memory': ['CD4', 'FOXP3', 'IL2RA', 'IL7R', 'CD3D', 'CD3E', 'CD3G'], 
 'CD19+ B': ['CD19', 'MS4A1', 'CD79A'], 
 'CD4+/CD25 T Reg': ['CD4', 'FOXP3', 'IL2RA', 'IL7R', 'CD3D', 'CD3E', 'CD3G'], 
 'CD56+ NK' : ['NCAM1', 'FCGR3A'], 
 'CD4+ T Helper2': ['CD4', 'FOXP3', 'IL2RA', 'IL7R', 'CD3D', 'CD3E', 'CD3G'], 
 'CD4+/CD45RA+/CD25- Naive T': ['CD4', 'FOXP3', 'IL2RA', 'IL7R', 'CD3D', 'CD3E', 'CD3G'], 
 'CD34+': ['CD34', 'THY1', 'ENG', 'KIT', 'PROM1'], 
 'Dendritic': ['IL3RA', 'CD1C', 'BATF3', 'THBD', 'CD209'], 
 'CD14+ Monocyte': ['CD14', 'FCGR1A', 'CD68', 'S100A12']}


celltype_scina_signature_genes = { 
 'CD14+ Monocyte': ['AIF1', 'CST3', 'FCN1', 'FTH1', 'FTL', 'GPX1', 'LST1', 'LYZ', 'S100A8', 'S100A9', 'TYMP'],
 'CD19+ B': ['CD37', 'CD74', 'CD79A', 'CD79B', 'HLA-DPA1', 'HLA-DRA'],
 'CD56+ NK': ['CLIC3', 'CST7', 'FGFBP2', 'GNLY', 'GZMA', 'GZMB', 'HOPX', 'IFITM2', 'KLRB1', 'NKG7', 'PRF1']}


import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pickle5 as pickle
import scipy.sparse as sparse
import sys
sys.path.append('/home/ding.bai/ding-scfmv1-downstream')
from performer_pytorch_cont.ding_models import DualEncoderSCFM
from GEARS.gears.model import reindex_tensor_universal
import scanpy as sc
import numpy as np
from tqdm import tqdm
import pandas as pd

# 设置 matplotlib 使用 TrueType 字体 (Type 42) 而不是 Type 3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 设置全局字体，可以选择其他常见的字体，比如 Arial
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或者 'Arial'



# 选择行和列之和最大的 m 个索引
m = 50  # 我们要选择前 m 个 genes

scfm_genes_list_path = '/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_genes_27k.txt'
with open('/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_genes_27k.txt', 'r') as f:
    ens_id_list = [line.strip('\n') for line in f.readlines()]

ens_id2symbol = {}
with open('/home/ding.bai/gene_id_translate/human_symbol_to_ens.txt', 'r') as f:
    for line in f.readlines():
        symbol = line.split(',')[0]
        ens_id = line.split(',')[1].strip('\n')
        if (':' not in symbol) and ('nan' not in symbol):
            ens_id2symbol[ens_id] = symbol


# Convert new indices to a list of index positions from the original tensor
scfm_genes_pad = [ens_id2symbol.get(ens_id, f'unknown{i}') for i, ens_id in enumerate(ens_id_list)] + ['PAD']
scfm_mapping = {idx: i for i, idx in enumerate(scfm_genes_pad)}
scfm_gene_num = len(scfm_genes_pad)


celltypes = list(celltype_makergenes.keys())
for celltype in celltypes:
    print(f'now processing celltype: {celltype}')
    celltype = celltype.replace('/', '-')
    mk_genes = celltype_makergenes.get(celltype, [])
    mk_genes = [g for g in mk_genes if g in scfm_genes_pad]
    mk_genes_idxes = [scfm_mapping[g] for g in mk_genes]
    print(f"remaining mk_genes: {mk_genes}")

    sig_genes = celltype_scina_signature_genes.get(celltype, [])
    sig_genes = [g for g in sig_genes if g in scfm_genes_pad]
    sig_genes_idxes = [scfm_mapping[g] for g in sig_genes]
    print(f"remaining sig_genes: {sig_genes}")


    attn_mat = torch.from_numpy(np.load(f"zheng68k_mats/{celltype}_attn_mat.npy")).to(torch.float32)
    print(f"attn_mat: {attn_mat.shape}")

    # 计算每行和每列的和
    row_sums = attn_mat.sum(dim=1)  # 每行的和
    col_sums = attn_mat.sum(dim=0)  # 每列的和

    # 行和列之和
    row_col_sums = row_sums + col_sums
    _, top_m_indices = torch.topk(row_col_sums, m)

    top_m_indices_new = np.unique([idx for idx in top_m_indices if ens_id_list[idx] in ens_id2symbol] + mk_genes_idxes + sig_genes_idxes)
    top_m_names = [ens_id2symbol[ens_id_list[idx]] for idx in top_m_indices_new]  # 对应的名称

    # 获取 m*m 的小方阵
    small_matrix = attn_mat[top_m_indices_new][:, top_m_indices_new]
    print(f"attn_map_shape: {small_matrix.shape}")

    #font_size = 1 
    #fig_size = (1.25 * font_size * m * 0.2, font_size * m * 0.2)  # 图像大小按比例调整
    fig_size = (6, 5)

    # 将矩阵转换为 DataFrame 以便与名称对应
    df = pd.DataFrame(small_matrix.numpy(), index=top_m_names, columns=top_m_names)

    #sns.set_theme(font_scale=font_size)
    # 使用 clustermap，仅在 y 方向上进行聚类
    sns_clustermap = sns.clustermap(
        df,
        cmap="coolwarm",
        annot=False,  # 不显示数字
        row_cluster=True,  # 聚类行
        col_cluster=False,  # 禁用列的聚类
        xticklabels=True,  # 显示 x 方向的名称
        yticklabels=True,  # 显示 y 方向的名称
        cbar_pos=None,
        dendrogram_ratio=(0.15, 0.2),
        figsize=fig_size,# 设置图像大小
    )

    # 获取行聚类后的名称顺序
    new_order = sns_clustermap.dendrogram_row.reordered_ind
    reordered_names = [top_m_names[i] for i in new_order]

    # 手动重新排列列顺序，使其与聚类后的行顺序相同
    sns_clustermap.data2d = df.iloc[new_order, new_order]

    # 重新绘制 clustermap
    sns_clustermap.ax_heatmap.clear()  # 清除原有的 heatmap
    sns.heatmap(
        sns_clustermap.data2d,
        cmap="coolwarm",
        annot=False,  # 不显示数字
        ax=sns_clustermap.ax_heatmap,
        cbar=True,
        cbar_kws={'location': 'right', 'ticks': [], 'label': 'Attention weights'},
        xticklabels=reordered_names,
        yticklabels=reordered_names
    )

    #
    cbar = sns_clustermap.ax_heatmap.collections[0].colorbar
    cbar.ax.set_position([0.82, 0.2, 0.03, 0.6])

    
    # 设置 x 和 y 标签的颜色
    x_labels = sns_clustermap.ax_heatmap.get_xticklabels()
    y_labels = sns_clustermap.ax_heatmap.get_yticklabels()

    for label in enumerate(x_labels):
        if label[1].get_text() in mk_genes:
            label[1].set_color('red')
        elif label[1].get_text() in sig_genes:
            label[1].set_color('purple')
        else:
            label[1].set_color('black')

    for label in enumerate(y_labels):
        if label[1].get_text() in mk_genes:
            label[1].set_color('red')
        elif label[1].get_text() in sig_genes:
            label[1].set_color('purple')
        else:
            label[1].set_color('black')

    # 重新绘制标签
    sns_clustermap.ax_heatmap.set_xticklabels(x_labels, size=4)
    sns_clustermap.ax_heatmap.set_yticklabels(y_labels, size=4)
    
    #plt.rcParams['axes.titley'] = 1.2
    #plt.rcParams['axes.titlepad'] = 10
    sns_clustermap.ax_heatmap.set_ylabel('Genes', labelpad=0, size=12)
    sns_clustermap.ax_heatmap.set_xlabel('Genes', labelpad=0, size=10)

    celltype = celltype.replace('+ ', '+')
    plt.title(f'Cell type: {celltype}')

    # 保存为 PDF 格式，图像足够大
    celltype = celltype.replace('/', '-')
    sns_clustermap.savefig(f"celltype_figs_new/celltype_{celltype}_{len(top_m_names)}.pdf", format='pdf', bbox_inches='tight', dpi=300)

    plt.close()

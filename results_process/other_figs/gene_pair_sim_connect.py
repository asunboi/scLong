import re

import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import torch
import pickle5 as pickle
import scipy.sparse as sparse
import sys
sys.path.append('/home/ding.bai/ding-scfmv1-downstream')
from performer_pytorch_cont.ding_models import DualEncoderSCFM
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr

# 设置 matplotlib 使用 TrueType 字体 (Type 42) 而不是 Type 3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 设置全局字体，可以选择其他常见的字体，比如 Arial
plt.rcParams['font.family'] = 'DejaVu Sans'  # 或者 'Arial'

#set +o noclobber
#python gene_pair_sim_connect.py > res/gene_pair_sim_connect_0.txt 2>&1


scfm_genes_list_path = '/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_genes_27k.txt'
with open("/l/users/ding.bai/Geneformer/checkpoints_ScFM/gocont_4096_48m_pretrain_1b_mix.pkl", 'rb') as f:  
    scfm_hyper_params =pickle.load(f)
print(scfm_hyper_params)
scfm_hyper_params['gene2vec_file'] = "/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_gene2vec_27k.npy"
scfm_ckpt_path = "/l/users/ding.bai/Geneformer/checkpoints_ScFM/gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth"

model = DualEncoderSCFM(**scfm_hyper_params).to('cpu')

ckpt = torch.load(scfm_ckpt_path, map_location = 'cpu')
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False

all_params = 0
all_params_grad = 0
param_num_dict = {"emb": 0, 
                "mini_encoder": 0, 
                "large_encoder": 0,
                "decoder": 0}
print("\nIncluding layers without parameters:")
for name, module in model.named_modules():
    if len(list(module.parameters(recurse=False))) > 0:
        # Only include modules with parameters
        total_params = sum(p.numel() for p in module.parameters(recurse=False))
        total_params_grad = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        all_params += total_params
        all_params_grad += total_params_grad
        for key in list(param_num_dict.keys()):
            if key in name:
                param_num_dict[key] += total_params_grad
                break

print(f"All model | Number of Params: {all_params}")
print(f"All model | Number of Params_grad: {all_params_grad}")
print(f"Modules param_grad number dict: {param_num_dict}")

model.eval()

#with open('/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_genes_27k.txt', 'r') as f:
#    ens_id_list = [line.strip('\n') for line in f.readlines()]

gene_emb = model.pos_emb.emb.weight.to('cpu')[:-1, :]
print(gene_emb.shape)

gene2vec = np.load(scfm_hyper_params['gene2vec_file'])
print(((gene_emb.numpy() - gene2vec)**2).mean())

G_go_old = model.go_conv.G_go.to('cpu')
G_go_weight_old = model.go_conv.G_go_weight.to('cpu')

G_go = [[], []]
G_go_weight = []
for idx, weight in enumerate(G_go_weight_old):
    node_0 = G_go_old[0, idx]
    node_1 = G_go_old[1, idx]
    if node_1 >= node_0:
        continue
    G_go[0].append(node_0)
    G_go[1].append(node_1)
    G_go_weight.append(weight)
G_go = torch.tensor(G_go).to(torch.long)
G_go_weight = torch.tensor(G_go_weight)


print(G_go.shape)
print(G_go_weight.shape)



K = 1000  # 假设选择前 K 条边

#topk_indices = np.random.choice(len(G_go_weight), size = (K,), replace = False)

# 1. 选出前 K 大的边
#topk_weights, topk_indices = torch.topk(G_go_weight, K)
#topk_weights  = G_go_weight[topk_indices]

# 2. 针对前 K 条边的节点，计算 Cosine similarity
#u_nodes = G_go[0, topk_indices]  # 第一个节点
#v_nodes = G_go[1, topk_indices]  # 第二个节点
u_nodes= G_go[0, :]
v_nodes= G_go[1, :]
topk_weights = G_go_weight

# 计算 u 和 v 节点在 gene_emb 中的 cosine similarity
cos_similarities = F.cosine_similarity(gene_emb[u_nodes], gene_emb[v_nodes])

# 3. 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(topk_weights.numpy(), cos_similarities.numpy(), color='b', label='Data points')

# 拟合一条直线
coefficients = np.polyfit(topk_weights.numpy(), cos_similarities.numpy(), 1)
poly = np.poly1d(coefficients)
plt.plot(topk_weights.numpy(), poly(topk_weights.numpy()), color='r', label=f'Fit: y={coefficients[0]:.3f}x + {coefficients[1]:.3f}')

# 设置标签
plt.xlabel('Edge Weights')
plt.ylabel('Cosine Similarity')
plt.title('Scatter Plot of Edge Weights vs Cosine Similarity')
plt.legend()

# 4. 计算 Pearson 相关系数
pearson_r, _ = pearsonr(topk_weights.numpy(), cos_similarities.numpy())
print(f"Pearson correlation coefficient: {pearson_r:.3f}")

# 输出拟合直线斜率
slope = coefficients[0]
print(f"Slope of the fit line: {slope:.3f}")

plt.savefig('figs/gene_pair_sim_connect.pdf',format="pdf", bbox_inches="tight",dpi = 300)

# 显示图像
plt.close()


######################
## Original G2V
######################

# 计算 u 和 v 节点在 gene_emb 中的 cosine similarity
cos_similarities = F.cosine_similarity(gene_emb[u_nodes], gene_emb[v_nodes])

# 3. 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(topk_weights.numpy(), cos_similarities.numpy(), color='b', label='Data points')

# 拟合一条直线
coefficients = np.polyfit(topk_weights.numpy(), cos_similarities.numpy(), 1)
poly = np.poly1d(coefficients)
plt.plot(topk_weights.numpy(), poly(topk_weights.numpy()), color='r', label=f'Fit: y={coefficients[0]:.3f}x + {coefficients[1]:.3f}')

# 设置标签
plt.xlabel('Edge Weights')
plt.ylabel('Cosine Similarity')
plt.title('Scatter Plot of Edge Weights vs Cosine Similarity')
plt.legend()

# 4. 计算 Pearson 相关系数
pearson_r, _ = pearsonr(topk_weights.numpy(), cos_similarities.numpy())
print(f"Pearson correlation coefficient: {pearson_r:.3f}")

# 输出拟合直线斜率
slope = coefficients[0]
print(f"Slope of the fit line: {slope:.3f}")

plt.savefig('figs/gene_pair_sim_connect.pdf',format="pdf", bbox_inches="tight",dpi = 300)

# 显示图像
plt.close()


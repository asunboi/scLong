# -*- coding: utf-8 -*-

#set +o noclobber

#CUDA_VISIBLE_DEVICES=1 python attnmap_output_pretrain.py > attnmap_output_pretrain.txt 2>&1

import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pickle5 as pickle
import scipy.sparse as sparse
import sys
sys.path.append('/home/ding.bai/ding-scfmv1-downstream')
from performer_pytorch_cont.ding_models import DualEncoderSCFM

scfm_genes_list_path = '/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_genes_27k.txt'
with open("/l/users/ding.bai/Geneformer/checkpoints_ScFM/gocont_4096_48m_pretrain_1b_mix.pkl", 'rb') as f:  
    scfm_hyper_params =pickle.load(f)
print(scfm_hyper_params)
scfm_hyper_params['gene2vec_file'] = "/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_gene2vec_27k.npy"
scfm_ckpt_path = "/l/users/ding.bai/Geneformer/checkpoints_ScFM/gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth"

model = DualEncoderSCFM(**scfm_hyper_params).to('cuda')

ckpt = torch.load(scfm_ckpt_path, map_location = 'cuda')
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
        #print(f"Layer: {name} | Module: {module.__class__.__name__} | Number of Params: {total_params}")
        #print(f"Layer: {name} | Module: {module.__class__.__name__} | Number of Params_grad: {total_params_grad}")
        for key in list(param_num_dict.keys()):
            if key in name:
                param_num_dict[key] += total_params_grad
                break

print(f"All model | Number of Params: {all_params}")
print(f"All model | Number of Params_grad: {all_params_grad}")
print(f"Modules param_grad number dict: {param_num_dict}")

L = scfm_hyper_params['top_seq_len']
print(f'top_seq_len: {L}')
model.eval()


start = 10000
cell_num = 100
batch_size = 1
# 选择行和列之和最大的 m 个索引
m = 200  # 我们要选择前 m 个


data_dir = f'/l/users/ding.bai/scfm-1b-data/start_{start}_cell_num_{cell_num}'

full_data = torch.zeros(cell_num, scfm_hyper_params['max_seq_len'])
for idx in range(start, start + cell_num):
    full_seq = sparse.load_npz(f"{data_dir}/cell{idx}.npz").toarray()
    full_seq = torch.from_numpy(full_seq).squeeze().to(torch.float32)
    full_seq = torch.cat((full_seq, torch.tensor([0.]))).to('cuda')
    full_data[idx - start, :] = full_seq

full_data = full_data.to('cuda')
print(f"full_data.shape: {full_data.shape}")

with open('/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_genes_27k.txt', 'r') as f:
    ens_id_list = [line.strip('\n') for line in f.readlines()]

ens_id2symbol = {}
with open('/home/ding.bai/gene_id_translate/human_symbol_to_ens.txt', 'r') as f:
    for line in f.readlines():
        symbol = line.split(',')[0]
        ens_id = line.split(',')[1].strip('\n')
        if (':' not in symbol) and ('nan' not in symbol):
            ens_id2symbol[ens_id] = symbol



for i in range(0, full_data.size(0), batch_size):
    print(f">> tensor: {i}")
    batch = full_data[i:i+batch_size, :].squeeze()  # 取出每个 batch
    _, top_indices = torch.topk(batch, L)  # 选择前 L 个最大值的索引
    top_L_names = [ens_id_list[idx] for idx in top_indices]  # 对应的名称

    exp_out, output = model(batch.reshape((1, -1)), output_attentions = True) #batch_size, seq_len, seq_len or 1, seq_len, seq_len?
    print(exp_out.shape)
    output = output['large_enc_attentions'].squeeze()
    print(output.shape) # L, L

    # 计算每行和每列的和
    row_sums = output.sum(dim=1)  # 每行的和
    col_sums = output.sum(dim=0)  # 每列的和

    # 行和列之和
    row_col_sums = row_sums + col_sums

    _, top_m_indices = torch.topk(row_col_sums, m)

    top_m_indices_new = [idx for idx in top_m_indices if top_L_names[idx] in ens_id2symbol]
    top_m_names = [ens_id2symbol[top_L_names[idx]] for idx in top_m_indices_new]  # 对应的名称

    # 获取 m*m 的小方阵
    small_matrix = output[top_m_indices_new][:, top_m_indices_new]
    print(f"attn_map_shape: {small_matrix.shape}")

    clustermap = sns.clustermap(small_matrix.cpu().numpy(), cmap="coolwarm", annot=False, xticklabels=top_m_names, yticklabels=top_m_names)
    clustermap.savefig(f"/home/ding.bai/ding-scfmv1-downstream/AttentionMap/pretrain_figs/cell_{i}_large_enc_attn_{len(top_m_names)}x{len(top_m_names)}.png")
    break
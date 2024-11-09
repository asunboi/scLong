# -*- coding: utf-8 -*-

#set +o noclobber

#CUDA_VISIBLE_DEVICES=0 python attnmap_output_npy_zheng68k.py > res/attnmap_output_npy_zheng68k_0.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python attnmap_output_npy_zheng68k.py > res/attnmap_output_npy_zheng68k_1.txt 2>&1
#CUDA_VISIBLE_DEVICES=2 python attnmap_output_npy_zheng68k.py > res/attnmap_output_npy_zheng68k_2.txt 2>&1
#CUDA_VISIBLE_DEVICES=3 python attnmap_output_npy_zheng68k.py > res/attnmap_output_npy_zheng68k_3.txt 2>&1

celltypes = [0,1,2]
celltypes = [3,4,5]
celltypes = [6,7,8]
celltypes = [9,10]


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

cell_num = 92
batch_size = 1
# 选择行和列之和最大的 m 个索引
m = 500  # 我们要选择前 m 个 genes

ad = sc.read_h5ad('/home/ding.bai/ding-scfmv1-downstream/cell_type_annotation/Zheng68K_raw_65943_cells_20386_genes.h5ad')
celltypes = ad.obs['celltype'].unique()[celltypes].tolist()

print(f"celltypes: {celltypes}")

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

with open('/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_genes_27k.txt', 'r') as f:
    ens_id_list = [line.strip('\n') for line in f.readlines()]

ens_id2symbol = {}
with open('/home/ding.bai/gene_id_translate/human_symbol_to_ens.txt', 'r') as f:
    for line in f.readlines():
        symbol = line.split(',')[0]
        ens_id = line.split(',')[1].strip('\n')
        if (':' not in symbol) and ('nan' not in symbol):
            ens_id2symbol[ens_id] = symbol


input_genes = ad.var.gene_name.tolist()
input_mapping = {idx: i for i, idx in enumerate(input_genes)}
input_gene_num = len(input_genes)
# Convert new indices to a list of index positions from the original tensor
scfm_genes_pad = [ens_id2symbol.get(ens_id, f'unknown{i}') for i, ens_id in enumerate(ens_id_list)] + ['PAD']
scfm_mapping = {idx: i for i, idx in enumerate(scfm_genes_pad)}
print(f'intersect genes num: {len(np.intersect1d(scfm_genes_pad, input_genes))}')
scfm_index_positions = [input_mapping.get(idx, input_gene_num) for idx in scfm_genes_pad]



for celltype in celltypes:
    print(f'>> Now processing celltype: {celltype}')
    full_data = ad[ad.obs['celltype'] == celltype]
    random_numbers = np.random.choice(full_data.shape[0], cell_num, replace=False)
    # 对选取的数进行从小到大的排序
    sorted_numbers = np.sort(random_numbers)
    full_data = torch.from_numpy(full_data.X[sorted_numbers,:].toarray()).squeeze().to(torch.float32).to('cuda')
    full_data = reindex_tensor_universal(full_data, scfm_index_positions, dim = 1, filler = '0')
    print(f'full_data.shape: {full_data.shape}')

    attn_mat = 0
    
    for i in tqdm(range(0, full_data.size(0), batch_size)):
        batch = full_data[i:i+batch_size, :].squeeze()  # 取出每个 batch
        _, top_indices = torch.topk(batch, L)  # 选择前 L 个最大值的索引
        top_L_names = [scfm_genes_pad[idx] for idx in top_indices]  # 对应的名称
        top_L_mapping = {name: i for i, name in enumerate(top_L_names)}
        top_L_scfm_index_positions = [top_L_mapping.get(name, L) for name in scfm_genes_pad]

        exp_out, output = model(batch.reshape((1, -1)), output_attentions = True) #batch_size, seq_len, seq_len or 1, seq_len, seq_len?
        output = output['large_enc_attentions'].squeeze() #4096, 4096

        output = reindex_tensor_universal(output, top_L_scfm_index_positions, dim = 0, filler = '0') #27875, 4096
        if i == 0:
            print(f"1st output.shape: {output.shape}")
        output = reindex_tensor_universal(output, top_L_scfm_index_positions, dim = 1, filler = '0') #27875, 27875
        if i == 0:
            print(f"2nd output.shape: {output.shape}")

        attn_mat += output

    attn_mat = attn_mat/cell_num
    print(f"attn_mat.max: {attn_mat.max()}\n attn_mat.min: {attn_mat.min()}\n attn_mat.mean: {attn_mat.mean()}")
    attn_mat = attn_mat.cpu().numpy()
    celltype = celltype.replace('/', '-')
    np.save(f'zheng68k_mats/{celltype}_attn_mat.npy', attn_mat)
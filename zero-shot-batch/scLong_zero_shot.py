import os
from typing import List, Optional, Tuple, Dict, Union
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import seaborn as sns
import scanpy as sc

import numpy as np

import torch
import pickle5 as pickle
import scipy.sparse as sparse
import sys
from tqdm import tqdm
import argparse
from copy import deepcopy
from new_utils import *
import torch.functional as F
from sklearn.decomposition import PCA


sys.path.append('/home/ding.bai/ding-scfmv1-downstream')
from performer_pytorch_cont.ding_models import DualEncoderSCFM

OUTPUT_DIR = '/l/users/ding.bai/zero-shot/res'

#set +o noclobber
#CUDA_VISIBLE_DEVICES=0 python scLong_zero_shot.py > res/scLong_zero_shot_test.txt 2>&1

parser = argparse.ArgumentParser()
parser.add_argument('--gene2vec_path', type=str, default='/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_gene2vec_27k.npy')
parser.add_argument('--scfm_hyper_params_path', type=str, default='/l/users/ding.bai/Geneformer/checkpoints_ScFM/gocont_4096_48m_pretrain_1b_mix.pkl')
parser.add_argument('--scfm_ckpt_path', type=str, default='/l/users/ding.bai/Geneformer/checkpoints_ScFM/gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth')
parser.add_argument('--output_key', type=str, default='merged_decodings')
parser.add_argument('--n_cells', type=str, default='all') # 1e3, 7500, 1e4, all


args = parser.parse_args()

gene2vec_path = args.gene2vec_path
scfm_hyper_params_path = args.scfm_hyper_params_path
scfm_ckpt_path = args.scfm_ckpt_path
batch_size = 4
output_key = args.output_key

if args.n_cells == 'all':
    n_cells = 1e10
else:
    n_cells = int(args.n_cells)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_sys(device)


def reindex_tensor_universal(tensor, index_positions, dim, filler = '0', device = 'cpu'):
    """
    Reindex a tensor along a specified dimension using new indices.

    :param tensor: Input tensor
    :param original_indices: List of original indices
    :param new_indices: List of new indices
    :param dim: Dimension along which to reindex
    :filler: Reindex filler, '0' or 'mean'. 
    :return: Reindexed tensor
    """

    # Convert list to tensor
    index_tensor = torch.tensor(index_positions, device=device)
    # Adjust shapes for the gather operation
    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = len(index_positions)
    index_shape = [1] * len(tensor_shape)
    index_shape[dim] = len(index_positions)
    index_tensor = index_tensor.view(*index_shape).expand(*tensor_shape)

    if filler == '0':
        padder_shape = deepcopy(tensor_shape)
        padder_shape[dim] = 1
        padder = torch.zeros(padder_shape, dtype = tensor.dtype, device=device)
    elif filler == 'mean':
        padder = tensor.mean(dim = dim, keepdim = True)
    else:
        raise ValueError("filler should be 0 or mean!")
    expanded_tensor = torch.cat((tensor, padder), dim = dim)

    # Reindex tensor
    reindexed = torch.gather(expanded_tensor, dim, index_tensor)

    return reindexed

print_sys(f'scLong output_key: {output_key}')



with open(scfm_hyper_params_path, 'rb') as f:  
    scfm_hyper_params =pickle.load(f)
scfm_hyper_params['gene2vec_file'] = gene2vec_path
print_sys(scfm_hyper_params)
scfm_ckpt_path = scfm_ckpt_path

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
print_sys("\nIncluding layers without parameters:")
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

print_sys(f"All model | Number of Params: {all_params}")
print_sys(f"All model | Number of Params_grad: {all_params_grad}")
print_sys(f"Modules param_grad number dict: {param_num_dict}")


with open('/home/ding.bai/gene_id_translate/human_symbol_to_ens.txt', 'r') as f:
    symbol2ens = {line.split(',')[0]: line.split(',')[1].rstrip('\n') for line in f.readlines() if line.split(',')[1] != 'unknown'}



L = scfm_hyper_params['top_seq_len']
print_sys(f'top_seq_len: {L}')
model.eval()



adata = sc.read_h5ad('/l/users/ding.bai/zero-shot/data/datasets/pancreas_scib.h5ad')
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.filter_genes(adata, min_cells=10)

n_cells = int(np.min((adata.n_obs, n_cells)))
        
# if adata_ too big, take a subset
if adata.n_obs > n_cells:
    print_sys(f"adata_ has {adata.n_obs} cells. Taking a subset of {n_cells} cells.")
    sc.pp.subsample(adata, n_obs = n_cells, copy = False)


input_genes = adata.var.index.tolist()
input_genes = [symbol2ens.get(symbol, f'unknown{i}') for i, symbol in enumerate(input_genes)]

input_mapping = {idx: i for i, idx in enumerate(input_genes)}
input_gene_num = len(input_genes)
# Convert new indices to a list of index positions from the original tensor
with open('/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_genes_27k.txt', 'r') as f:
    scfm_genes = [line.rstrip('\n') for line in f.readlines()]
    print_sys(f"target and scfm genes intersect: {len(np.intersect1d(input_genes, scfm_genes))}")
scfm_genes_pad = scfm_genes + ['PAD']
scfm_seq_len = len(scfm_genes_pad)
scfm_mapping = {idx: i for i, idx in enumerate(scfm_genes_pad)}


scfm_index_positions = [input_mapping.get(idx, input_gene_num) for idx in scfm_genes_pad]
input_index_positions = [scfm_mapping.get(idx, scfm_seq_len) for idx in input_genes]


cell_embeddings = []
embedding_key = "X_scLong"

for i in tqdm(range(0, n_cells, batch_size)):
    with torch.no_grad():
        x = torch.tensor(adata.X[i:i+batch_size, :]).to(torch.float32).to(device)  # (batch_size, seq_len)
        x_scfm = reindex_tensor_universal(x, scfm_index_positions, dim = 1, filler = '0', device=device) #b, seq_len
        cell_emb = model.get_cell_emb(x_scfm)
        cell_embeddings.append(cell_emb.cpu().numpy())

cell_embeddings = np.concatenate(cell_embeddings, axis = 0) # (n_cells, seq_len)


adata.obsm[embedding_key] = cell_embeddings
    
print_sys(f'cell_embeddings: {cell_embeddings.shape}')


res = evaluate(adata_ = adata, batch_key = 'batch', label_key = ['celltype'], embedding_key = "X_scLong", 
         res_path = os.path.join(OUTPUT_DIR, f"scLong_batch_cell_emb_mode.csv"))

print_sys(res)


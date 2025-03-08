# -*- coding: utf-8 -*-

#set +o noclobber

#CUDA_VISIBLE_DEVICES=0 python embed.py --target_data_path > res/embed.txt 2>&1

import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pickle5 as pickle
import scipy.sparse as sparse
import sys
sys.path.append('/home/ding.bai/ding-scfmv1-downstream')
from performer_pytorch_cont.ding_models import DualEncoderSCFM
import scanpy as sc
import numpy as np
from tqdm import tqdm
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--target_data_path', type=str, default='')
parser.add_argument('--gene2vec_path', type=str, default='selected_gene2vec_27k.npy')
parser.add_argument('--scfm_hyper_params_path', type=str, default='gocont_4096_48m_pretrain_1b_mix.pkl')
parser.add_argument('--scfm_ckpt_path', type=str, default='gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth')
parser.add_argument('--scfm_genes_list_path', type=str, default='selected_genes_27k.txt')
parser.add_argument('--target_embed_path', type=str, default='')


args = parser.parse_args()

target_data_path = args.target_data_path
target_embed_path = args.target_embed_path
gene2vec_path = args.gene2vec_path
scfm_hyper_params_path = args.scfm_hyper_params_path
scfm_ckpt_path = args.scfm_ckpt_path
scfm_genes_list_path = args.scfm_genes_list_path
batch_size = 4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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




with open(scfm_hyper_params_path, 'rb') as f:  
    scfm_hyper_params =pickle.load(f)
scfm_hyper_params['gene2vec_file'] = gene2vec_path
print(scfm_hyper_params)
scfm_ckpt_path = scfm_ckpt_path

model = DualEncoderSCFM(**scfm_hyper_params).to(device)

ckpt = torch.load(scfm_ckpt_path, map_location = device)
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

L = scfm_hyper_params['top_seq_len']
print(f'top_seq_len: {L}')
model.eval()



adata = sc.read_h5ad(target_data_path)
n_cells = adata.n_obs

########################################################
### Make sure adata gene index are ENSEMBL IDs.
### Expressions needs to be log1p normalzed.
########################################################

input_genes = adata.var.index.tolist() 

input_mapping = {idx: i for i, idx in enumerate(input_genes)}
input_gene_num = len(input_genes)
# Convert new indices to a list of index positions from the original tensor
with open(scfm_genes_list_path, 'r') as f:
    scfm_genes = [line.rstrip('\n') for line in f.readlines()]
    print(f"target and scfm genes intersect: {len(np.intersect1d(input_genes, scfm_genes))}")
scfm_genes_pad = scfm_genes + ['PAD']
scfm_seq_len = len(scfm_genes_pad)
scfm_mapping = {idx: i for i, idx in enumerate(scfm_genes_pad)}


scfm_index_positions = [input_mapping.get(idx, input_gene_num) for idx in scfm_genes_pad]
input_index_positions = [scfm_mapping.get(idx, scfm_seq_len) for idx in input_genes]

output = []
for i in tqdm(range(0, n_cells, batch_size)):
    with torch.no_grad():
        x = torch.tensor(adata.X[i:i+batch_size, :]).to(torch.float32).to(device)  # (batch_size, n_genes)

        x_scfm = reindex_tensor_universal(x, scfm_index_positions, dim = 1, filler = '0', device=device) #batch_size, seq_len
        
        output_encodings = model(x_scfm, return_encodings = True)['merged_decodings'] #batch_size, seq_len, emb_dim
    
        output_reindex = reindex_tensor_universal(output_encodings, input_index_positions, dim = 1, filler = 'mean', device=device) #batch_size, n_genes, emb_dim

        output.append(output_reindex.cpu().numpy()) # (batch_size, n_genes, emb_dim)


output = np.concatenate(output, axis = 0) # (n_cells, n_genes, emb_dim)
print(f'output shape: {output.shape}')

np.save(target_embed_path, output)

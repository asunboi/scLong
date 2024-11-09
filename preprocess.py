import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import h5py
panglao = sc.read_h5ad('./panglao_10000.h5ad')

ref = []
print('reading ref...')
with open('ensembl.txt', 'r') as file:
    for line in file:
        columns = line.strip().split('\t')
        ref.append(columns[1])
ref = ref[1:] # remove header
print(f'ref size: {len(ref)}')

for i in range(200,201):
    print(f'reading dataset_{i}...')
    data = ad.read_h5ad(f'./dataset_{i}.h5ad')
    counts = sparse.lil_matrix((data.X.shape[0],panglao.X.shape[1]),dtype=np.float32)

    # Open the h5ad file using h5py
    with h5py.File(f'./dataset_{i}.h5ad', 'r') as f:
        var_group = f['var']
        print(list(var_group))
        if 'feature_id' in list(var_group):
            var_id = var_group['feature_id']
        if 'gene_ids' in list(var_group):
            var_id = var_group['gene_ids']
        if 'gene_id' in list(var_group):
            var_id = var_group['gene_id']
        if 'ensembl_ids' in list(var_group):
            var_id = var_group['ensembl_ids']
        if 'ensembl_id' in list(var_group):
            var_id = var_group['ensembl_id']
        if '_index' in list(var_group):
            var_id = var_group['_index']
        var_id = [name.decode() for name in var_id]
        obj = var_id
    print(f'obj size: {len(obj)}')
    print('writing obj...')
    str = '\t'
    f=open("obj.txt","w")
    f.write(str.join(obj))
    f.close()

    print('merging matrix...')
    for i in range(len(ref)):
        if ref[i] in obj:
            loc = obj.index(ref[i])
            counts[:,i] = data.X[:,loc]

    counts = counts.tocsr()
    new = ad.AnnData(X=counts)
    new.var_names = ref
    new.obs_names = data.obs_names
    new.obs = data.obs
    new.uns = panglao.uns
    print(f'matrix shape: {new.X.shape}')
    print('filtering cells...')
    sc.pp.filter_cells(new, min_genes=200)
    sc.pp.normalize_total(new, target_sum=1e4)
    print(f'filtered matrix shape: {new.X.shape}')
    #sc.pp.log1p(new, base=2)
    new.write(f'./preprocess/dataset_{i}.h5ad')
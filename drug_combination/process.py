import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Test intersect genes
ckpt='v20228'
ind=True
gene_list_scfm=[]
with open('/data3/ruiyi/scfm/selected_genes_27k.txt') as f:
    for l in f.readlines():
        gene_list_scfm.append(str(l.strip()))
print(len(gene_list_scfm))

gene_list_deepdds=[]
if ind:
    with open('/data1/ruiyi/deepdds/data/independent_set/independent_cell_features_954.csv') as f:
        for l in f.readlines():
            l=l.strip().split(',')
            #print(l)
            break
else:
    with open('/data1/ruiyi/deepdds/data/new_cell_features_954.csv') as f:
        for l in f.readlines():
            l=l.strip().split(',')
            #print(l)
            break
gene_list_deepdds=l[1:]
print(len(gene_list_deepdds))

cnt=0
for gene in gene_list_deepdds:
    if gene in gene_list_scfm:cnt+=1
print(cnt)

if ind:
    deepdds_scfm_output=np.load('/data1/ruiyi/deepdds/deepdds_ind_scfm{}_output.npy'.format(ckpt))
else:
    deepdds_scfm_output=np.load('/data1/ruiyi/deepdds/deepdds_scfm{}_output.npy'.format(ckpt))
print(deepdds_scfm_output.shape)

deepdds_scfm_output_processed=np.zeros((deepdds_scfm_output.shape[0],len(gene_list_deepdds),deepdds_scfm_output.shape[-1]))
gene_deepdds2idx={v:k for k,v in enumerate(gene_list_deepdds)}
for k,v in enumerate(gene_list_scfm):
    if v in gene_deepdds2idx:
        idx=gene_deepdds2idx[v]
        deepdds_scfm_output_processed[:,idx,:]=deepdds_scfm_output[:,k,:]
print(deepdds_scfm_output_processed.shape)    

if ind:
    np.save('/data1/ruiyi/deepdds/deepdds_ind_scfm{}_output_processd.npy'.format(ckpt), deepdds_scfm_output_processed)
else:
    np.save('/data1/ruiyi/deepdds/deepdds_scfm{}_output_processd.npy'.format(ckpt), deepdds_scfm_output_processed)

cell_list_deepdds=[]
if ind:
    with open('/data1/ruiyi/deepdds/data/independent_set/independent_cell_features_954.csv') as f:
        for l in f.readlines():
            l=l.strip().split(',')
            cell_list_deepdds.append(l[0])
else:
    with open('/data1/ruiyi/deepdds/data/new_cell_features_954.csv') as f:
        for l in f.readlines():
            l=l.strip().split(',')
            cell_list_deepdds.append(l[0])
cell_list_deepdds=cell_list_deepdds[2:]
#cell_list_deepdds,len(cell_list_deepdds)
print(len(cell_list_deepdds))

cell2embed={cell_list_deepdds[i].split('_')[0]:deepdds_scfm_output_processed[i] for i in range(len(cell_list_deepdds))}
if ind:
    np.save('/data1/ruiyi/deepdds/deepddsindcell2scfm{}embed.npy'.format(ckpt), cell2embed)
else:
    np.save('/data1/ruiyi/deepdds/deepddscell2scfm{}embed.npy'.format(ckpt), cell2embed)

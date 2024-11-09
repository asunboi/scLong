import pandas as pd
import numpy as np
import math
import os
import torch
from sklearn.preprocessing import normalize
# Load the CSV file
#os.environ["MKL_NUM_THREADS"] = "64"

df_np=np.load('/data3/ruiyi/deepsem/deepsem_geneformer_embed_reordered.npy')
df_np=df_np.reshape((df_np.shape[0],-1))
df_np=normalize(df_np,norm='l1')
print(df_np.shape)
print(df_np[0])
print(sum(df_np[0]),sum(sum(df_np)))

for df_i in df_np:
    m = np.median(df_i[df_i > 0])
    if not math.isnan(m):

    # Assign the median to the zero elements 
        df_i[df_i == 0] = m

print(df_np[0])

df_torch=torch.HalfTensor(df_np).cuda()
print(df_torch.mean())

adj=torch.matmul(df_torch,torch.t(df_torch))
#adj=np.dot(df_np.T,df_np)
print(adj.shape)
for i in range(adj.shape[0]):
    adj[i,i]=0
adj_norm=normalize(adj.detach().cpu().numpy(),norm='l1')
np.save('/data3/ruiyi/deepsem/adj_deepsem_geneformer.npy',adj_norm)


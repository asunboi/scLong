import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease
from tqdm import tqdm


# Load the CSV file
df_expr = pd.read_csv('/data3/ruiyi/deepsem/ExpressionData.csv')
df_expr = df_expr.transpose()
new_header = df_expr.iloc[0]

# Step 2: Re-assign these names to the columns
df_expr.columns = new_header

# Step 3: Drop the first row from the DataFrame
df_expr = df_expr.drop(df_expr.index[0])

# Step 4: Reset the DataFrame index
df_expr.reset_index(drop=True, inplace=True)
#print(df_expr.head)

dfkey2idx={v:k for k,v in enumerate(df_expr.keys())}

scfm_rep = np.load('/data3/ruiyi/deepsem/deepsem_scfm10m_nopre_output.npy')

#scfm_rep=scfm_rep.T
print(scfm_rep.shape)
print(scfm_rep[0])

scfm_deepsem_rep=np.zeros((len(df_expr.keys()),scfm_rep.shape[0],scfm_rep.shape[-1]))

gene_list_scfm=[]

with open('/data3/ruiyi/scfm/selected_genes_27k.txt') as f:
    for s in f.readlines():
        gene_list_scfm.append(s.strip())

gene_scfm_idx={v:k for k,v in enumerate(gene_list_scfm)}

data = EnsemblRelease(109)

data.download()
data.index()

pos=0
neg=0
cnt=0
for k in tqdm(df_expr.keys()):
    try:
        gene_id=data.gene_ids_of_gene_name(k)[0]
        if gene_id in gene_scfm_idx:
            scfm_idx=gene_scfm_idx[gene_id]
            data_scfm_rep=scfm_rep[:,scfm_idx,:]
            pos+=1
            scfm_deepsem_rep[dfkey2idx[k],:,:]=data_scfm_rep
        else:
            neg+=1
    except:
        cnt+=1

print(pos,neg,cnt,scfm_deepsem_rep.shape)

np.save('/data3/ruiyi/deepsem/deepsem_scfm10m_nopre_reordered.npy',scfm_deepsem_rep)

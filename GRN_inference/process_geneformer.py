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

#scfm_rep = np.load('/data3/ruiyi/deepsem/deepsem_scfmv20226_output.npy')
df_geneformer=pd.read_csv('/data3/ruiyi/geneformer/output/deepsem.csv',index_col=0)
geneformer_rep=df_geneformer.values

print(geneformer_rep.shape)
#exit()
#print(scfm_rep[0])

geneformer_deepsem_rep=np.zeros((len(df_expr.keys()),geneformer_rep.shape[-1]))

gene_list_geneformer=list(df_geneformer.index)

gene_geneformer_idx={v:k for k,v in enumerate(gene_list_geneformer)}

# Replace 'XX' with the release number you want to use
data = EnsemblRelease(109)

data.download()
data.index()

pos=0
neg=0
cnt=0
for k in tqdm(df_expr.keys()):
    try:
        gene_id=data.gene_ids_of_gene_name(k)[0]
        if gene_id in gene_geneformer_idx:
            geneformer_idx=gene_geneformer_idx[gene_id]
            data_geneformer_rep=geneformer_rep[:,geneformer_idx]
            pos+=1
            geneformer_deepsem_rep[dfkey2idx[k],:,:]=data_geneformer_rep
        else:
            neg+=1
    except:
        cnt+=1

print(pos,neg,cnt,geneformer_deepsem_rep.shape)

np.save('/data3/ruiyi/deepsem/deepsem_geneformer_embed_reordered.npy',geneformer_deepsem_rep)
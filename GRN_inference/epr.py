import pandas as pd
import numpy as np
epr_list=[]

for ep in range(10,15):
    output = pd.read_csv('/data3/ruiyi/deepsem/hesc_scfm1227_matrix_out/GRN_inference_result_ep{}.tsv'.format(str(ep)),sep='\t')
    output=output.rename(columns={"TF": "Gene1", "Target": "Gene2"})
    #output = pd.read_csv('hesc_out_ep50/GRN_inference_result.tsv',sep='\t')
    output['EdgeWeight'] = abs(output['EdgeWeight'])
    output = output.sort_values('EdgeWeight',ascending=False)
    label = pd.read_csv('/data3/ruiyi/deepsem/label_hesc.csv')
    TFs = set(label['Gene1'])
    Genes = set(label['Gene1'])| set(label['Gene2'])
    output = output[output['Gene1'].apply(lambda x: x in TFs)]
    output = output[output['Gene2'].apply(lambda x: x in Genes)]
    label_set = set(label['Gene1']+'|'+label['Gene2'])
    output= output.iloc[:len(label_set)]
    epr=len(set(output['Gene1']+'|' +output['Gene2']) & label_set) / (len(label_set)**2/(len(TFs)*len(Genes)-len(TFs)))
    print(epr)
    epr_list.append(epr)

print('std:',np.std(epr_list))

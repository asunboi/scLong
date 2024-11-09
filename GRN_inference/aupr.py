from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd

ap_list=[]
for ep in range(10,15):
    output = pd.read_csv('/data3/ruiyi/deepsem/hesc_scfm1227_matrix_out/GRN_inference_result_ep{}.tsv'.format(str(ep)),sep='\t')
    #output = pd.read_csv('hesc_out_ep50/GRN_inference_result.tsv',sep='\t')
    output=output.rename(columns={"TF": "Gene1", "Target": "Gene2"})
    #print(output.head)
    output['EdgeWeight'] = abs(output['EdgeWeight'])
    output = output.sort_values('EdgeWeight',ascending=False)
    label = pd.read_csv('/data3/ruiyi/deepsem/label_hesc.csv')
    TFs = set(label['Gene1'])
    Genes = set(label['Gene1'])| set(label['Gene2'])
    output = output[output['Gene1'].apply(lambda x: x in TFs)]
    output = output[output['Gene2'].apply(lambda x: x in Genes)]
    label_set = set(label['Gene1']+label['Gene2'])
    preds,labels,randoms = [] ,[],[]
    res_d = {}
    l = []
    p= []
    for item in (output.to_dict('records')):
            res_d[item['Gene1']+item['Gene2']] = item['EdgeWeight']
    for item in (set(label['Gene1'])):
            for item2 in  set(label['Gene1'])| set(label['Gene2']):
                if item+item2 in label_set:
                    l.append(1)
                else:
                    l.append(0)
                if item+ item2 in res_d:
                    p.append(res_d[item+item2])
                else:
                    p.append(-1)
    ap_score=average_precision_score(l,p)/np.mean(l)
    print(ap_score)
    ap_list.append(ap_score)

print('std',np.std(ap_list))

import os
from itertools import islice
import sys
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch
from creat_data_DC import creat_data
from tqdm import tqdm

import pandas as pd

ind=True
ckpt='geneformer2'
ckpt_scfm = 'v20228'
#ckpt = ckpt_scfm
scfm=True
geneformer = True
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None, smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if ind:
            self.cell2embed=np.load('/data1/ruiyi/deepdds/deepddsindcell2scfm{}embed.npy'.format(ckpt_scfm),allow_pickle=True).item()
            print(len(self.cell2embed))
            if geneformer:
                #self.processed_paths[0]=''
                print(self.processed_paths)
                celllist = list(self.cell2embed.keys())
                df = pd.read_csv('/data3/ruiyi/geneformer/output/deepdds_ind.csv', index_col=0, skiprows=1)

                # Convert the DataFrame to a dictionary where keys are row names and values are NumPy arrays
                self.cell2geneformer =  {celllist[k]: np.array(df.loc[row]) for k,row in enumerate(df.index)}
                print(len(self.cell2geneformer),len(self.cell2embed))
                #exit()
                print(self.cell2geneformer['SW948'].shape)
            #print(self.cell2embed['SW948'].shape)
            #exit()
        else:
            self.cell2embed=np.load('/data1/ruiyi/deepdds/deepddscell2scfm{}embed.npy'.format(ckpt_scfm),allow_pickle=True).item()
            if geneformer:
                print(self.processed_paths)
                celllist = list(self.cell2embed.keys())
                df = pd.read_csv('/data3/ruiyi/geneformer/output/deepdds.csv', index_col=0, skiprows=1)

                # Convert the DataFrame to a dictionary where keys are row names and values are NumPy arrays
                self.cell2geneformer =  {celllist[k]: np.array(df.loc[row]) for k,row in enumerate(df.index)}
                print(len(self.cell2geneformer),len(self.cell2embed))

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
            #print('data',self.data)
            #print('slices',self.slices)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            #print(xd, xt, xt_featrue, y, smile_graph)
            self.process(xd, xt, xt_featrue, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                print(cellId)
                return row[1:]
        return False
    
    def get_cell_scfm_feature(self, cellId, cell_features):
        return self.cell2embed.get(cellId)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, xt_featrue, y, smile_graph):
        print(len(xd),len(xt),len(y))
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print('number of data', data_len)
        for i in tqdm(range(data_len)):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.Tensor([labels]))
            #print(target,xt_featrue)

            if scfm and not geneformer:
                #print(target,xt_featrue)
                print('test')
                exit()
                cell_scfm = self.get_cell_scfm_feature(target, xt_featrue)
                if cell_scfm is None:
                    print('fail: ',target)
                    continue
                cell = self.get_cell_feature(target, xt_featrue)
                new_cell = []
                for n in cell:
                    new_cell.append(float(n))
                cell_scfm=torch.FloatTensor([cell_scfm])
                cell=torch.FloatTensor([new_cell])
                cell_combined=torch.cat((cell_scfm,cell.unsqueeze(-1)),dim=-1)

                GCNData.cell = cell_combined
            elif geneformer:
                cell = self.get_cell_feature(target, xt_featrue)
                #print(self.cell2geneformer)
                cell_geneformer = self.cell2geneformer.get(target)
                if cell_geneformer is None:
                    print('fail: ',target)
                    continue
                #print(cell_geneformer)
                #exit()
                new_cell = []
                for n in cell:
                    new_cell.append(float(n))
                for n in cell_geneformer:
                    new_cell.append(n)
                GCNData.cell = torch.FloatTensor([new_cell])
            else:
                print('test2')
                cell = self.get_cell_feature(target, xt_featrue)
                new_cell = []
                for n in cell:
                    new_cell.append(float(n))
                GCNData.cell = torch.FloatTensor([new_cell])
            '''
            if cell == False : # 如果读取cell失败则中断程序
                print('cell', cell)
                sys.exit()
            '''
            
            
            # print('cell_feature', cell_feature)
            
            #print(GCNData.cell.shape)
            #exit()
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
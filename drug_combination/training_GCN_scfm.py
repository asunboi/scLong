import random
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch.utils.data as Data
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle

from torch.utils.data import TensorDataset, Dataset
from models.gat import GATNet
from models.gat_gcn_test import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
from torch.optim.lr_scheduler import StepLR
import pandas as pd

# CPU or GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

# modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
# datasets = [['davis', 'kiba'][int(sys.argv[1])]]
# model_st = modeling.__name__
modeling = GCNNet

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0001
LOG_INTERVAL = 1
NUM_EPOCHS = 100

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
datafile = 'new_labels_0_10'



if scfm and not geneformer:
    drug1_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=datafile + '_drug1_combined_{}'.format(ckpt))
    drug2_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=datafile + '_drug2_combined_{}'.format(ckpt))
    ind_datafile='independent_set/independent_input'
    ind_drug1_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=ind_datafile + '_drug1_combined_{}'.format(ckpt))
    ind_drug2_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=ind_datafile + '_drug2_combined_{}'.format(ckpt))
elif geneformer:
    name = ckpt
    drug1_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=datafile + '_drug1_combined_{}'.format(name))
    drug2_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=datafile + '_drug2_combined_{}'.format(name))
    ind_datafile='independent_set/independent_input'
    ind_drug1_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=ind_datafile + '_drug1_combined_{}'.format(name))
    ind_drug2_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=ind_datafile + '_drug2_combined_{}'.format(name))
else:
    drug1_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=datafile + '_drug1_bsl')
    drug2_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=datafile + '_drug2_bsl')
    ind_datafile='independent_set/independent_input'
    ind_drug1_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=ind_datafile + '_drug1_bsl')
    ind_drug2_data = TestbedDataset(root='/data1/ruiyi/deepdds/data', dataset=ind_datafile + '_drug2_bsl')


lenth = len(drug1_data)
pot = int(lenth/5)
print('lenth', lenth)
print('pot', pot)

few_shot=False
domain_gen=True
random_num = random.sample(range(0, lenth), lenth)
for i in range(5):
    if domain_gen:
        drug1_data_train = drug1_data
        drug1_data_test = ind_drug1_data
        
        drug2_data_train = drug2_data
        drug2_data_test = ind_drug2_data
    else:
        if few_shot:
            train_num = random_num[pot*i:pot*(i+1)]
            test_num = random_num[:pot*i] + random_num[pot*(i+1):]
        else:
            test_num = random_num[pot*i:pot*(i+1)]
            train_num = random_num[:pot*i] + random_num[pot*(i+1):]


        drug1_data_train = drug1_data[train_num]
        drug1_data_test = drug1_data[test_num]

        drug2_data_test = drug2_data[test_num]
        drug2_data_train = drug2_data[train_num]
    # print('type(drug1_data_train)', type(drug1_data_train))
    # print('drug1_data_train[0]', drug1_data_train[0])
    # print('len(drug1_data_train)', len(drug1_data_train))
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None, pin_memory=True)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None, pin_memory=True)



    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None, pin_memory=True)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None, pin_memory=True)

    model = modeling().to(device)
    for name, param in model.named_parameters():
        print(f"Parameter Name: {name}, Shape: {param.size()}")
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_params)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Create a StepLR scheduler
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)


    # model_file_name = 'data/result/GCNNet(DrugA_DrugB)' + str(i) + '--model_' + datafile +  '.model'
    # result_file_name = 'data/result/GCNNet(DrugA_DrugB)' + str(i) + '--result_' + datafile +  '.csv'
    # file_AUCs = 'data/result/GCNNet(DrugA_DrugB)' + str(i) + '--AUCs--' + datafile + '.txt'
    # AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    # with open(file_AUCs, 'w') as f:
    #     f.write(AUCs + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        scheduler.step()
        if epoch%5==0:
            T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
            # T is correct label
            # S is predict score
            # Y is predict label
            #print(T,S,Y)
            # compute preformence
            try:
                AUC = roc_auc_score(T, S)
            except:
                AUC=0
            fpr, tpr, thres = metrics.roc_curve(T, S)
            
            precision, recall, threshold = metrics.precision_recall_curve(T, S)
            PR_AUC = metrics.auc(recall, precision)
            BACC = balanced_accuracy_score(T, Y)
            tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
            TPR = tp / (tp + fn)
            PREC = precision_score(T, Y)
            ACC = accuracy_score(T, Y)
            KAPPA = cohen_kappa_score(T, Y)
            recall = recall_score(T, Y)

            # save data
            AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]
            # save_AUCs(AUCs, file_AUCs)
            ret = [rmse(T, S), mse(T, S), pearson(T, S), spearman(T, S), ci(T, S)]
            if best_auc < AUC:
                best_auc = AUC
                print('best_auc:',best_auc)
                print(fpr)
                print(tpr)
                print(thres)
            print(AUCs)
            print(ret)
            '''
                # torch.save(model.state_dict(), model_file_name)
            independent_num = []
            independent_num.append(test_num)
            independent_num.append(T)
            independent_num.append(Y)
            independent_num.append(S)
            txtDF = pd.DataFrame(data=independent_num)
            result_file_name='gcn_scfm_results.csv'
            txtDF.to_csv(result_file_name, index=False, header=False)
            '''
    if scfm:
        with open('fpr_tpr/fpr_tpr_scfm_run{}_{}.pkl'.format(str(i),ckpt),'wb') as fp:
            pickle.dump((fpr,tpr), fp)
    else:
        with open('fpr_tpr/fpr_tpr_bsl_run{}_{}.pkl'.format(str(i),ckpt),'wb') as fp:
            pickle.dump((fpr,tpr), fp)


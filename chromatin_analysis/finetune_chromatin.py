# -*- coding: utf-8 -*-

import os
import pdb
import gc
import argparse
import json
import random
import math
import random
import subprocess
from functools import reduce
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import pandas as pd
import copy
from scipy import sparse
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, auc, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch_cont.ding_models import DualEncoderSCFM, get_similarity_network, GeneSimNetwork
from Geneformer.geneformer.pretrainer import token_dictionary
from Geneformer.geneformer import DataCollatorForGeneClassification
from tqdm.notebook import tqdm
import scanpy as sc
import anndata as ad
from utils import *
import pickle
import time
import datetime
from datasets import load_from_disk

import os
os.environ['MASTER_ADDR'] = 'localhost'   # IP address of the master node
os.environ['WORLD_SIZE'] = '1'            # Total number of processes
os.environ['RANK'] = '0'                  # The rank of this process
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import wandb
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["WANDB_DISABLED"] = 'true'
os.environ["NCCL_SOCKET_IFNAME"]="^docker0,lo"
os.environ["NCCL_DEBUG_SUBSYS"]="ALL"

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=1, help='Local process rank.')
parser.add_argument("--gene_list_path", type=str,
                    default="/data/luoyingtao/scFM/checkpoints_ScFM/selected_genes_27k.txt")
parser.add_argument("--gene_info_table", type=str,
                    default="gene_info_table.csv")
parser.add_argument("--finetune_dataset", type=str,
                    default="panglao_SRA553822-SRS2119548.dataset")
parser.add_argument("--bivalent_TFs_path", type=str, default="bivalent_vs_no_methyl.pickle")
parser.add_argument("--gene_num", type=int, default=27874, help='Number of genes.')
parser.add_argument("--cell_num", type=int, default=49738024, help='Number of cells.')
parser.add_argument("--epoch", type=int, default=5, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=8, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--valid_step", type=int, default=10000, help='Number of training steps between twice validation.')
parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
parser.add_argument("--replace_prob", type=float, default=0.9,
                    help='Probability of replacing with [MASK] token for masking.')
parser.add_argument("-pos_embed", action='store_false', help='Using Gene2vec encoding or not.')
parser.add_argument("--ckpt_dir", type=str, default='/data/luoyingtao/scFM/checkpoints_ScFM/ckpts/',
                    help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='gocont_4096_38m_pretrain_1126', help='Pretrained model name.')
parser.add_argument("--go_data_path", type=str,
                    default='/data/luoyingtao/scFM/checkpoints_ScFM/human_ens_gene2go')  # Without _graph.csv
parser.add_argument("--gene2vec_data_path", type=str,
                    default='/data/luoyingtao/scFM/checkpoints_ScFM/selected_gene2vec_27k.npy')
parser.add_argument("-go_use_gene2vec", action='store_false')
parser.add_argument("-use_wandb", action='store_false')
parser.add_argument("--model_path", type=str, default='/data/luoyingtao/scFM/gocont_4096_38m_pretrain_1b_2023-11-26_19-01-03-003.pth',
                    help='Directory of checkpoint to load.')
parser.add_argument("--start_epoch", type=int, default=1, help='start_epoch')
parser.add_argument("--log_every", type=int, default=60, help='log_every')
parser.add_argument("--save_time_interval", type=int, default=1, help='model save time interval in hours.')
parser.add_argument("--val_time_interval", type=int, default=48, help='model val time interval in hours.')
parser.add_argument("--master_port", type=str, default='12355', help='port on the master node.')

args = parser.parse_args()
local_rank = args.local_rank
rank = int(os.environ["RANK"])
is_master = rank == 0
os.environ['MASTER_PORT'] = args.master_port       # A free port on the master node

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
VALIDATE_STEP = args.valid_step
MASK_PROB = args.mask_prob
REPLACE_PROB = args.replace_prob
RANDOM_TOKEN_PROB = 0.
MASK_TOKEN_ID = -10
PAD_TOKEN_ID = -10
MASK_IGNORE_TOKEN_IDS = [0]
POS_EMBED_USING = args.pos_embed
LOG_EVERY = args.log_every
PAD_4096 = True
finetune_method = 'lora'  # 'lora', 'full'
if finetune_method == 'lora':
    r=8
    lora_alpha=4


with open(args.gene_list_path, "r") as f:
    gene_list = [line.rstrip('\n') for line in f.readlines()]
    print('Reading gene_list finished!')
    print(len(gene_list))

if len(gene_list) != SEQ_LEN - 1:
    raise ValueError("Gene num not correct with selected genes!")

model_name = args.model_name
ckpt_dir = args.ckpt_dir
save_time_interval = args.save_time_interval * 60 * 60
val_time_interval = args.val_time_interval * 60 * 60

dist.init_process_group(backend='nccl', init_method='env://')
print('DDP init!')
# local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device('cuda:1')
torch.cuda.set_device(device)
world_size = torch.distributed.get_world_size()
seed_all(SEED + torch.distributed.get_rank())

print("local_rank: ", local_rank)


def preprocess_classifier_batch(cell_batch, max_len):
    if max_len == None:
        max_len = max([len(i) for i in cell_batch["input_ids"]])

    def pad_label_example(example):
        example["labels"] = np.pad(example["labels"],
                                   (0, max_len - len(example["input_ids"])),
                                   mode='constant', constant_values=-100)
        example["input_ids"] = np.pad(example["input_ids"],
                                      (0, max_len - len(example["input_ids"])),
                                      mode='constant', constant_values=token_dictionary.get("<pad>"))
        example["attention_mask"] = (example["input_ids"] != token_dictionary.get("<pad>")).astype(int)
        return example

    padded_batch = cell_batch.map(pad_label_example)
    return padded_batch


def vote(logit_pair):
    a, b = logit_pair
    if a > b:
        return 0
    elif b > a:
        return 1
    elif a == b:
        return "tie"


def py_softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


# get cross-validated mean and sd metrics
def get_cross_valid_metrics(all_tpr, all_roc_auc, all_tpr_wt):
    wts = [count / sum(all_tpr_wt) for count in all_tpr_wt]
    print(wts)
    all_weighted_tpr = [a * b for a, b in zip(all_tpr, wts)]
    mean_tpr = np.sum(all_weighted_tpr, axis=0)
    mean_tpr[-1] = 1.0
    all_weighted_roc_auc = [a * b for a, b in zip(all_roc_auc, wts)]
    roc_auc = np.sum(all_weighted_roc_auc)
    roc_auc_sd = math.sqrt(np.average((all_roc_auc - roc_auc) ** 2, weights=wts))
    return mean_tpr, roc_auc, roc_auc_sd


# Function to find the largest number smaller
# than or equal to N that is divisible by k
def find_largest_div(N, K):
    rem = N % K
    if (rem == 0):
        return N
    else:
        return N - rem
    

# forward batch size is batch size for model inference (e.g. 200)
def classifier_predict(model, evalset, forward_batch_size, mean_fpr):
    predict_logits = []
    predict_labels = []
    model.eval()

    test_dataset = Dataset(evalset, device=device)
    test_sampler = SequentialDistributedSampler(test_dataset, batch_size=BATCH_SIZE, world_size=world_size)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

    # ensure there is at least 2 examples in each batch to avoid incorrect tensor dims
    evalset_len = len(evalset)
    max_divisible = find_largest_div(evalset_len, forward_batch_size)
    if len(evalset) - max_divisible == 1:
        evalset_len = max_divisible

    max_evalset_len = max(evalset.select([i for i in range(evalset_len)])["length"])

    for i in range(0, evalset_len, forward_batch_size):
        max_range = min(i + forward_batch_size, evalset_len)
        batch_evalset = evalset.select([i for i in range(i, max_range)])
        padded_batch = preprocess_classifier_batch(batch_evalset, max_evalset_len)
        padded_batch.set_format(type="torch")

        input_data_batch = padded_batch["input_ids"]
        attn_msk_batch = padded_batch["attention_mask"]
        label_batch = padded_batch["labels"]
        with torch.no_grad():
            outputs = model(
                input_ids=input_data_batch.to("cuda"),
                attention_mask=attn_msk_batch.to("cuda"),
                labels=label_batch.to("cuda"),
            )
            predict_logits += [torch.squeeze(outputs.logits.to("cpu"))]
            predict_labels += [torch.squeeze(label_batch.to("cpu"))]

    logits_by_cell = torch.cat(predict_logits)
    all_logits = logits_by_cell.reshape(-1, logits_by_cell.shape[2])
    labels_by_cell = torch.cat(predict_labels)
    all_labels = torch.flatten(labels_by_cell)
    logit_label_paired = [item for item in list(zip(all_logits.tolist(), all_labels.tolist())) if item[1] != -100]
    y_pred = [vote(item[0]) for item in logit_label_paired]
    y_true = [item[1] for item in logit_label_paired]
    logits_list = [item[0] for item in logit_label_paired]
    # probability of class 1
    y_score = [py_softmax(item)[1] for item in logits_list]
    conf_mat = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # plot roc_curve for this split
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.show()
    # interpolate to graph
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    return fpr, tpr, interp_tpr, conf_mat


class Dataset(Dataset):
    def __init__(self, data, device):
        super().__init__()
        self.data = data
        self.device = device
        padding_size = SEQ_LEN-2048
        if PAD_4096:
            self.padded_data = F.pad(data['input_ids'], (0, padding_size), 'constant', 0)
            self.padded_label = F.pad(data['labels'], (0, padding_size), 'constant', 0)

    def __getitem__(self, index):
        if PAD_4096:
            data = self.padded_data[index].to(torch.float32).to(self.device)
            label = self.padded_label[index].to(self.device)
        else:
            data = torch.tensor(self.data['input_ids'][index], dtype=torch.float32).to(self.device)
            label = torch.tensor(self.data['labels'][index]).to(self.device)

        return data, label

    def __len__(self):
        return len(self.data['length'])


class CustomBCELoss(nn.Module):
    def __init__(self, device, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.device = device
        self.loss = nn.BCEWithLogitsLoss(reduction='none').to(device)

    def forward(self, input, target):
        input = input.squeeze(-1)
        target = target.to(torch.float32)
        mask = (target != self.ignore_index).float()  # Create a mask to ignore specific indices
        loss = self.loss(input, target)
        masked_loss = loss * mask  # Apply the mask to the loss
        return masked_loss.mean()

        
def retain_non_masked_values(tensor1, tensor2, mask_token=MASK_TOKEN_ID):
    mask = tensor2 != mask_token
    result1 = tensor1[mask]
    result2 = tensor2[mask]
    return result1, result2


cell_num = args.cell_num
data = np.array([f'cell{i}.npz' for i in range(cell_num)])

# table of corresponding Ensembl IDs, gene names, and gene types (e.g. coding, miRNA, etc.)
gene_info = pd.read_csv(args.gene_info_table, index_col=0)

# create dictionaries for corresponding attributes
gene_id_type_dict = dict(zip(gene_info["ensembl_id"], gene_info["gene_type"]))
gene_name_id_dict = dict(zip(gene_info["gene_name"], gene_info["ensembl_id"]))
gene_id_name_dict = {v: k for k, v in gene_name_id_dict.items()}


# function for preparing targets and labels
def prep_inputs(genegroup1, genegroup2, id_type):
    if id_type == "gene_name":
        targets1 = [gene_name_id_dict[gene] for gene in genegroup1 if gene_name_id_dict.get(gene) in token_dictionary]
        targets2 = [gene_name_id_dict[gene] for gene in genegroup2 if gene_name_id_dict.get(gene) in token_dictionary]
    elif id_type == "ensembl_id":
        targets1 = [gene for gene in genegroup1 if gene in token_dictionary]
        targets2 = [gene for gene in genegroup2 if gene in token_dictionary]

    targets1_id = [token_dictionary[gene] for gene in targets1]
    targets2_id = [token_dictionary[gene] for gene in targets2]

    targets = np.array(targets1_id + targets2_id)
    labels = np.array([0] * len(targets1_id) + [1] * len(targets2_id))
    nsplits = min(5, min(len(targets1_id), len(targets2_id)) - 1)
    assert nsplits > 2
    print(f"# targets1: {len(targets1_id)}\n# targets2: {len(targets2_id)}\n# splits: {nsplits}")
    return targets, labels, nsplits


# preparing targets and labels for dosage sensitive vs insensitive TFs
with open(args.bivalent_TFs_path , 'rb') as f:
  dosage_tfs = pickle.load(f)
sensitive = dosage_tfs['bivalent'].dropna()
insensitive = dosage_tfs['no_methylation'].dropna()
targets, labels, nsplits = prep_inputs(sensitive, insensitive, "ensembl_id")

# load training dataset
train_dataset = load_from_disk(args.finetune_dataset)
shuffled_train_dataset = train_dataset.shuffle(seed=42)
subsampled_train_dataset = shuffled_train_dataset.select([i for i in range(50_000)])

# Preprocess Gene ontology weighted graph
edge_list = get_similarity_network(gene_list, data_path=args.go_data_path, num_similar_genes_go_graph=20)
sim_network = GeneSimNetwork(edge_list, gene_list)
G_go = sim_network.edge_index
G_go_weight = sim_network.edge_weight
print(G_go.shape, G_go_weight.shape)


# cross-validate gene classifier
def cross_validate(raw_data, targets, raw_labels, nsplits, subsample_size, output_dir, num_proc):
    # check if output directory already written to
    # ensure not overwriting previously saved model
    last_save_time = time.time()
    last_val_time = time.time()

    model_dir_test = os.path.join(output_dir, "ksplit0/models/pytorch_model.bin")
    if os.path.isfile(model_dir_test) == True:
        raise Exception("Model already saved to this directory.")

    # initiate eval metrics to return
    num_classes = len(set(raw_labels))
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    all_roc_auc = []
    all_tpr_wt = []
    label_dicts = []
    confusion = np.zeros((num_classes, num_classes))

    # set up cross-validation splits
    skf = StratifiedKFold(n_splits=nsplits, random_state=0, shuffle=True)
    # train and evaluate
    iteration_num = 0
    for train_index, eval_index in tqdm(skf.split(targets, raw_labels)):
        if len(raw_labels) > 500:
            print("early stopping activated due to large # of training examples")
            nsplits = 3
            if iteration_num == 3:
                break
        print(f"****** Crossval split: {iteration_num}/{nsplits - 1} ******\n")
        # generate cross-validation splits
        targets_train, targets_eval = targets[train_index], targets[eval_index]
        labels_train, labels_eval = raw_labels[train_index], raw_labels[eval_index]
        label_dict_train = dict(zip(targets_train, labels_train))
        label_dict_eval = dict(zip(targets_eval, labels_eval))
        label_dicts += (iteration_num, targets_train, targets_eval, labels_train, labels_eval)

        # function to filter by whether contains train or eval labels
        def if_contains_train_label(example):
            a = label_dict_train.keys()
            b = example['input_ids']
            return not set(a).isdisjoint(b)

        def if_contains_eval_label(example):
            a = label_dict_eval.keys()
            b = example['input_ids']
            return not set(a).isdisjoint(b)

        # filter dataset for examples containing classes for this split
        print(f"Filtering training data")
        trainset = raw_data.filter(if_contains_train_label, num_proc=num_proc)
        print(f"Filtered {round((1 - len(trainset) / len(raw_data)) * 100)}%; {len(trainset)} remain\n")
        print(f"Filtering evalation data")
        evalset = raw_data.filter(if_contains_eval_label, num_proc=num_proc)
        print(f"Filtered {round((1 - len(evalset) / len(raw_data)) * 100)}%; {len(evalset)} remain\n")

        # minimize to smaller training sample
        training_size = min(subsample_size, len(trainset))
        trainset_min = trainset.select([i for i in range(training_size)])
        eval_size = min(training_size, len(evalset))
        half_training_size = round(eval_size / 2)
        evalset_train_min = evalset.select([i for i in range(half_training_size)])
        evalset_oos_min = evalset.select([i for i in range(half_training_size, eval_size)])

        # label conversion functions
        def generate_train_labels(example):
            example["labels"] = [label_dict_train.get(token_id, -100) for token_id in example["input_ids"]]
            return example

        def generate_eval_labels(example):
            example["labels"] = [label_dict_eval.get(token_id, -100) for token_id in example["input_ids"]]
            return example

        # label datasets
        print(f"Labeling training data")
        trainset_labeled = trainset_min.map(generate_train_labels)
        print(f"Labeling evaluation data")
        evalset_train_labeled = evalset_train_min.map(generate_eval_labels)
        print(f"Labeling evaluation OOS data")
        evalset_oos_labeled = evalset_oos_min.map(generate_eval_labels)

        # create output directories
        ksplit_output_dir = os.path.join(output_dir, f"ksplit{iteration_num}")
        ksplit_model_dir = os.path.join(ksplit_output_dir, "models/")

        # ensure not overwriting previously saved model
        model_output_file = os.path.join(ksplit_model_dir, "pytorch_model.bin")
        if os.path.isfile(model_output_file) == True:
            raise Exception("Model already saved to this directory.")

        # make training and model output directories
        subprocess.call(f'mkdir {ksplit_output_dir}', shell=True)
        subprocess.call(f'mkdir {ksplit_model_dir}', shell=True)

        collator = DataCollatorForGeneClassification()
        collated_trainset = collator([trainset_labeled[i] for i in range(len(trainset_labeled))])
        collated_valset = collator([evalset_train_labeled[i] for i in range(len(evalset_train_labeled))])

        train_dataset = Dataset(collated_trainset, device=device)
        val_dataset = Dataset(collated_valset, device=device)
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = SequentialDistributedSampler(val_dataset, batch_size=BATCH_SIZE, world_size=world_size)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

        model_params = {
            "max_seq_len": SEQ_LEN,  # max length of sequence
            "top_seq_len": 4096,
            "base_dim": 200,  # dim of tokens
            "mini_enc_depth": 2,  # layers in mini encoder
            "mini_enc_heads": 8,  # num of heads in mini encoder
            "mini_enc_dim_head": 64,  # dim of heads in mini encoder
            "large_dim": 1280,  # dim of tokens in large encoder
            "large_enc_depth": 42,  # layers in large encoder
            "large_enc_heads": 32,  # num of heads in large encoder
            "large_enc_dim_head": 64,  # dim of heads in large encoder
            "dec_depth": 2,  # layers in decoder
            "dec_heads": 8,  # num of heads decoder
            "dec_dim_head": 64,  # dim of heads decoder
            "mask_token_thres": -1,  # values smaller than -1 is a mask
            "g2v_position_emb": POS_EMBED_USING,
            "G_go": G_go,
            "G_go_weight": G_go_weight,
            "go_num_layers": 1,
            "device": 'cuda',
            "go_use_gene2vec": args.go_use_gene2vec,
            "gene2vec_file": args.gene2vec_data_path
        }
        model = DualEncoderSCFM(**model_params)

        if is_master:
            with open(ckpt_dir + model_name + '.pkl', 'wb') as handle:
                pickle.dump(model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
                    # print(f"Layer: {name} | Module: {module.__class__.__name__} | Number of Params: {total_params}")
                    # print(f"Layer: {name} | Module: {module.__class__.__name__} | Number of Params_grad: {total_params_grad}")
                    for key in list(param_num_dict.keys()):
                        if key in name:
                            param_num_dict[key] += total_params_grad
                            break

            print(f"All model | Number of Params: {all_params}")
            print(f"All model | Number of Params_grad: {all_params_grad}")
            print(f"Modules param number dict: {param_num_dict}")

        path = args.model_path
        if path != 'None':
            ckpt = torch.load(path, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
            if finetune_method == 'linprob':
                model.exp_to_out = nn.Linear(model_params["base_dim"], 1)
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.exp_to_out.parameters():
                    param.requires_grad = True
            elif finetune_method == 'lora':
                from peft import inject_adapter_in_model, LoraConfig
                target_modules = ["to_q", "to_v", "to_k", "to_out", "w1", "w2"]
                lora_config = LoraConfig(lora_alpha=lora_alpha, lora_dropout=0.1, r=r, bias="none", target_modules=target_modules)
                model = inject_adapter_in_model(lora_config, model)
                
                # # Transfer weights for the two linear layers in TwoLayerMLPEmb
                # pretrained_token_emb = copy.deepcopy(model.token_emb)
                # model.token_emb.fc1 = lora.Linear(1, 50, r=r, lora_alpha=lora_alpha)
                # model.token_emb.fc1.weight.data = pretrained_token_emb.fc1.weight.data.clone()
                # model.token_emb.fc1.bias.data = pretrained_token_emb.fc1.bias.data.clone()
                # model.token_emb.fc2 = lora.Linear(50, model_params['base_dim'], r=r, lora_alpha=lora_alpha)
                # model.token_emb.fc2.weight.data = pretrained_token_emb.fc2.weight.data.clone()
                # model.token_emb.fc2.bias.data = pretrained_token_emb.fc2.bias.data.clone()

                # pretrained_mask_emb = copy.deepcopy(model.mask_emb)
                # model.mask_emb = lora.Embedding(1, model_params['base_dim'], r=r, lora_alpha=lora_alpha, 
                #                                 max_norm=True, dtype = torch.float32)
                # model.mask_emb.weight.data = pretrained_mask_emb.weight.data.clone()

                # pretrained_base_to_large = copy.deepcopy(model.base_to_large)
                # model.base_to_large = lora.Linear(model_params['base_dim'], model_params['large_dim'], r=r, lora_alpha=lora_alpha)
                # model.base_to_large.weight.data = pretrained_base_to_large.weight.data.clone()
                # model.base_to_large.bias.data = pretrained_base_to_large.bias.data.clone()

                # pretrained_large_to_base = copy.deepcopy(model.large_to_base)
                # model.large_to_base = lora.Linear(model_params['large_dim'], model_params['base_dim'], r=r, lora_alpha=lora_alpha)
                # model.large_to_base.weight.data = pretrained_large_to_base.weight.data.clone()
                # model.large_to_base.bias.data = pretrained_large_to_base.bias.data.clone()

                # pretrained_exp_to_out = copy.deepcopy(model.exp_to_out)
                # model.exp_to_out = lora.Linear(model_params['base_dim'], 1, r=r, lora_alpha=lora_alpha)
                # model.exp_to_out.weight.data = pretrained_exp_to_out.weight.data.clone()
                # model.exp_to_out.bias.data = pretrained_exp_to_out.bias.data.clone()

                # lora.mark_only_lora_as_trainable(model, bias='lora_only')
            else:
                for param in model.parameters():
                    param.requires_grad = True
        

        model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        # optimizer
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        # if path != 'None':
        #     ckpt = torch.load(path, map_location='cpu')
        #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # learning rate scheduler
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=LEARNING_RATE,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9
        )
        # if path != 'None':
        #     ckpt = torch.load(path, map_location='cpu')
        #     scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        # loss_fn = nn.MSELoss(reduction='mean').to(local_rank)
        # loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean').to(local_rank)
        loss_fn = CustomBCELoss(device)

        train_log_step = 0
        valid_log_step = 0

        assert EPOCHS != 0
        dist.barrier()
        for i in range(args.start_epoch + 1, args.start_epoch + EPOCHS + 1):
            train_loader.sampler.set_epoch(i)
            model.train()
            dist.barrier()
            running_loss = 0.0
            cum_acc = 0.0
            for index, (data, labels) in enumerate(train_loader):
                index += 1
                if index % GRADIENT_ACCUMULATION != 0:
                    with model.no_sync():
                        logits = model(data)
                        loss = loss_fn(logits, labels) / GRADIENT_ACCUMULATION
                        loss.backward()
                if index % GRADIENT_ACCUMULATION == 0:
                    logits = model(data)
                    loss = loss_fn(logits, labels) / GRADIENT_ACCUMULATION
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
                    optimizer.step()
                    optimizer.zero_grad()
                    if is_master:
                        current_time = time.time()
                        if current_time - last_save_time >= save_time_interval:
                            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                            save_ckpt(time_stamp, model, optimizer, scheduler, loss.item(), model_name, ckpt_dir)
                            last_save_time = current_time
                if index % LOG_EVERY == 0:
                    if is_master and args.use_wandb:
                        print(
                            {"train_loss": loss.item(), "learning_rate": scheduler.optimizer.param_groups[0]['lr'],
                             "train_log_step": train_log_step})
                        train_log_step += 1
                running_loss += loss.item()

                # for validation on training steps
                if index % VALIDATE_STEP == 0:
                    model.eval()
                    dist.barrier()
                    running_loss_val = 0.0
                    running_error = 0.0
                    predictions = []
                    truths = []
                    val_index_num = 0
                    with torch.no_grad():
                        for val_index, (data, labels) in enumerate(val_loader):
                            val_index_num += 1
                            logits = model(data)
                            loss_val = loss_fn(logits, labels)
                            running_loss_val += loss_val.item()
                            if index % LOG_EVERY == 0:
                                if is_master and args.use_wandb:
                                    print({"val_step_loss": loss_val.item(), "valid_log_step": valid_log_step})
                                    valid_log_step += 1
                                # if is_master:
                                #     current_time = time.time()
                                #     if current_time - last_save_time >= save_time_interval:
                                #         time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                                #         save_ckpt(time_stamp, model, optimizer, scheduler, loss.item(), model_name,
                                #                   ckpt_dir)
                                        # last_save_time = current_time
                        del data, labels, logits
                        # gather
                        loss_val_all = running_loss_val / val_index_num
                        loss_val_reduced = get_reduced(loss_val_all, local_rank, 0, world_size)

                    # if is_master:
                    #     save_ckpt(index, model, optimizer, scheduler, loss.item(), model_name, ckpt_dir)

                    model.train()

            epoch_loss = running_loss / index
            epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
            if is_master:
                print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}')
                if args.use_wandb:
                    print({"train_epoch_loss": epoch_loss, "epoch": i})
            dist.barrier()
            scheduler.step()

            if i % VALIDATE_EVERY == 0:
                model.eval()
                dist.barrier()
                running_loss = 0.0
                running_error = 0.0
                predictions = []
                truths = []
                with torch.no_grad():
                    for index, (data, labels) in enumerate(val_loader):
                        index += 1
                        logits = model(data)
                        loss = loss_fn(logits, labels)
                        running_loss += loss.item()
                        if index % LOG_EVERY == 0:
                            if is_master and args.use_wandb:
                                print({"val_loss": loss.item(), "valid_log_step": valid_log_step})
                                valid_log_step += 1
                            # if is_master:
                            #     current_time = time.time()
                            #     if current_time - last_save_time >= save_time_interval:
                            #         time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                            #         save_ckpt(time_stamp, model, optimizer, scheduler, loss.item(), model_name,
                            #                   ckpt_dir)
                            #         last_save_time = current_time
                    del data, labels, logits
                    # gather
                    val_loss = running_loss / index
                    val_loss = get_reduced(val_loss, local_rank, 0, world_size)
                if is_master:
                    print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f}')
                    if args.use_wandb:
                        print({"val_epoch_loss": val_loss, "epoch": i})

            if is_master:
                save_ckpt(f"epoch{i}", model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir)

        # evaluate model
        fpr, tpr, interp_tpr, conf_mat = classifier_predict(model, evalset_oos_labeled, 200, mean_fpr)

        # append to tpr and roc lists
        confusion = confusion + conf_mat
        all_tpr.append(interp_tpr)
        all_roc_auc.append(auc(fpr, tpr))
        # append number of eval examples by which to weight tpr in averaged graphs
        all_tpr_wt.append(len(tpr))

        iteration_num = iteration_num + 1

    # get overall metrics for cross-validation
    mean_tpr, roc_auc, roc_auc_sd = get_cross_valid_metrics(all_tpr, all_roc_auc, all_tpr_wt)
    return all_roc_auc, roc_auc, roc_auc_sd, mean_fpr, mean_tpr, confusion, label_dicts


# plot ROC curve
def plot_ROC(bundled_data, title):
    plt.figure()
    lw = 2
    for roc_auc, roc_auc_sd, mean_fpr, mean_tpr, sample, color in bundled_data:
        plt.plot(mean_fpr, mean_tpr, color=color,
                 lw=lw, label="{0} (AUC {1:0.2f} $\pm$ {2:0.2f})".format(sample, roc_auc, roc_auc_sd))
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


# plot confusion matrix
def plot_confusion_matrix(classes_list, conf_mat, title):
    display_labels = []
    i = 0
    for label in classes_list:
        display_labels += ["{0}\nn={1:.0f}".format(label, sum(conf_mat[:, i]))]
        i = i + 1
    display = ConfusionMatrixDisplay(confusion_matrix=preprocessing.normalize(conf_mat, norm="l1"),
                                     display_labels=display_labels)
    display.plot(cmap="Blues", values_format=".2g")
    plt.title(title)


# max input size
max_input_size = 2 ** 11  # 2048

# set training hyperparameters
# max learning rate
max_lr = 5e-5
# how many pretrained layers to freeze
freeze_layers = 4
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 24
# batch size for training and eval
batch_size = 12
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 500
# number of epochs
epochs = 1
# optimizer
optimizer = "adamw"

# set training arguments
subsample_size = 10_000
# define output directory path
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
training_output_dir = f"dosage_sensitivity/{datestamp}_ScFM_GeneClassifier_dosageTF_L{max_input_size}_B{batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_n{subsample_size}_F{freeze_layers}/"
# if not os.path.exists(training_output_dir):
#     os.makedirs(training_output_dir)

# ensure not overwriting previously saved model
ksplit_model_test = os.path.join(training_output_dir, "ksplit0/models/pytorch_model.bin")
if os.path.isfile(ksplit_model_test) == True:
    raise Exception("Model already saved to this directory.")

# make output directory
subprocess.call(f'mkdir {training_output_dir}', shell=True)

# cross-validate gene classifier
all_roc_auc, roc_auc, roc_auc_sd, mean_fpr, mean_tpr, confusion, label_dicts \
    = cross_validate(subsampled_train_dataset, targets, labels, nsplits, subsample_size, training_output_dir, 1)

# bundle data for plotting
bundled_data = []
bundled_data += [(roc_auc, roc_auc_sd, mean_fpr, mean_tpr, "ScFM", "red")]

# plot ROC curve
plot_ROC(bundled_data, 'Bivalent vs No Methylation Genes')

# plot confusion matrix
classes_list = ["Dosage Sensitive", "Dosage Insensitive"]
plot_confusion_matrix(classes_list, confusion, "ScFM")

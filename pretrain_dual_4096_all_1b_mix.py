# -*- coding: utf-8 -*-

import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch_cont.ding_models import DualEncoderSCFM, get_similarity_network, GeneSimNetwork
import scanpy as sc
import anndata as ad
from utils import *
import pickle
import time
import datetime
from torch import autocast
from torch.cuda.amp import GradScaler


import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--gene_list_path", type=str, 
                    default = "/lustre/home/ding.bai/scFound/check_datasets/selected/selected_genes_27k.txt")
parser.add_argument("--gene_num", type=int, default=27874, help='Number of genes.')
parser.add_argument("--cell_num", type=int, default=49738024, help='Number of cells.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
parser.add_argument("--valid_batch_size", type=int, default=5, help='Number of validation batch size.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--valid_step", type=int, default=1e3, help='Number of training steps between twice validation.')
parser.add_argument("--valid_rate", type=float, default=0.05, help='validation rate.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
parser.add_argument("--replace_prob", type=float, default=0.9, help='Probability of replacing with [MASK] token for masking.')
parser.add_argument("-pos_embed", action='store_false', help='Using Gene2vec encoding or not.')
parser.add_argument("--ckpt_dir", type=str, default='/lustre/scratch/shared-folders/bio_project/dingbai/ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='gocont_27k_16m_pretrain', help='Pretrained model name.')
parser.add_argument("--data_path", type=str, 
                    default='/lustre/scratch/shared-folders/bio_project/dingbai/all_cell_exp_no_dup/cell_files', 
                    help='Path of data for pretraining.')
parser.add_argument("--go_data_path", type=str, default='/lustre/scratch/shared-folders/bio_project/dingbai/human_ens_gene2go') #Without _graph.csv
parser.add_argument("--gene2vec_data_path", type=str, default='/lustre/scratch/shared-folders/bio_project/dingbai/selected_gene2vec_27k.npy')
parser.add_argument("-go_use_gene2vec", action='store_false')
parser.add_argument("-use_wandb", action='store_false')
parser.add_argument("-compile", action='store_true')
parser.add_argument("--model_path", type=str, default='None', 
                    help='Directory of checkpoint to load.')
parser.add_argument("--start_epoch", type=int, default=1, help='start_epoch')
parser.add_argument("--log_every", type=int, default=60, help='log_every')
parser.add_argument("--save_time_interval", type=int, default=24, help='model save time interval in hours.')
parser.add_argument("--val_time_interval", type=int, default=48, help='model val time interval in hours.')
parser.add_argument("--amp_dtype", type=str, default="bfloat16")
parser.add_argument("--api_key", type=str, default="None")
parser.add_argument("--load_opt_and_sched", action='store_true')

args = parser.parse_args()
local_rank = args.local_rank
rank = int(os.environ["RANK"])
is_master = rank == 0

if is_master and args.use_wandb:
    api_key = args.api_key
    if api_key != "None":
        wandb.login(key=api_key)
    wandb.init(
        # set the wandb project where this run will be logged
        project = "pretrain_1b_50m",
        name = args.model_name,
        entity = "biofm",
        # track hyperparameters and run metadata
        config={
        "architecture": "scBERT_continous+GO_GNN",
        "gene_num": args.gene_num,
        "cell_num": args.cell_num,
        "load model": args.model_path,
        "start epoch": args.start_epoch,
        "epochs to train": args.epoch,
        "valid_every": args.valid_every,
        "batch_size": args.batch_size,
        "go_use_gene2vec": args.go_use_gene2vec
        }
    )

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = args.valid_batch_size
VALIDATE_STEP = args.valid_step
VALID_RATE = args.valid_rate
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
MASK_PROB = args.mask_prob
REPLACE_PROB = args.replace_prob
RANDOM_TOKEN_PROB = 0.
MASK_TOKEN_ID = -10
PAD_TOKEN_ID = -10
MASK_IGNORE_TOKEN_IDS = [0]
POS_EMBED_USING = args.pos_embed
LOG_EVERY = args.log_every
COMPILE = args.compile
AMP_DTYPE = args.amp_dtype
cell_num = args.cell_num
load_opt_and_sched = args.load_opt_and_sched

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[AMP_DTYPE]

with open(args.gene_list_path, "r") as f:
    gene_list = [line.rstrip('\n') for line in f.readlines()]

if len(gene_list) != SEQ_LEN - 1:
    raise ValueError("Gene num not correct with selected genes!")

model_name = args.model_name
ckpt_dir = args.ckpt_dir
save_time_interval = args.save_time_interval * 60 * 60
val_time_interval = args.val_time_interval * 60 * 60
last_save_time = time.time()
last_val_time = time.time()

dist.init_process_group(backend='nccl', init_method='env://')
print('DDP init!')
local_rank = int(os.environ['LOCAL_RANK'])
device = f'cuda:{local_rank}'
torch.cuda.set_device(device)
world_size = torch.distributed.get_world_size()
seed_all(SEED + torch.distributed.get_rank())

print("local_rank: ", local_rank)

# get the random prob matrix and True means smaller than prob threshold
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# get the mask matrix which cannot be masked
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)      # num of mask of a single sequence in average
    num_tokens = mask.sum(dim=-1, keepdim=True)     # num of pure tokens of each sequence except special tokens
    mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0),mask.size(-1)).to(device)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())        # only 15% of pure tokens can be masked
    mask_excess = mask_excess[:, :max_masked]       # get difference between 15% of pure tokens and 15% of all tokens
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)     # rand (0-1) as prob, special token use -1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)      # get index of topk prob to mask
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)        # delete difference of mask not pure
    new_mask = torch.zeros((batch, seq_len + 1), device=device)     # get (batch, seq_len) shape zero matrix
    new_mask.scatter_(-1, sampled_indices, 1)       # set masks in zero matrix as 1
    return new_mask[:, 1:].bool()       # the final mask, True is mask

def data_mask(data,
    mask_prob = MASK_PROB,
    replace_prob = REPLACE_PROB,
    num_tokens = None,
    random_token_prob = RANDOM_TOKEN_PROB,
    mask_token_id = MASK_TOKEN_ID,
    pad_token_id = PAD_TOKEN_ID,
    mask_ignore_token_ids = MASK_IGNORE_TOKEN_IDS
):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)   # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)      # get the True/False mask matrix
    # get mask indices
    ## mask_indices = torch.nonzero(mask, as_tuple=True)   # get the index of mask(nonzero value of mask matrix)
    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
    masked_input = data.clone().detach()
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        assert num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
        random_token_prob = prob_mask_like(data, random_token_prob)       # get the mask matrix of random token replace
        random_tokens = torch.randint(0, num_tokens, data.shape, device=data.device)     # generate random token matrix with the same shape as input
        random_no_mask = mask_with_tokens(random_tokens, mask_ignore_token_ids)        # not masked matrix for the random token matrix
        random_token_prob &= ~random_no_mask        # get the pure mask matrix of random token replace
        random_indices = torch.nonzero(random_token_prob, as_tuple=True)        # index of random token replace
        # masked_data[random_indices] = random_tokens[random_indices]        # replace some tokens by random token
    # [mask] input
    replace_prob = prob_mask_like(data, replace_prob)     # get the mask matrix of token being masked
    masked_input = masked_input.masked_fill(mask * replace_prob, mask_token_id)        # get the data has been masked by mask_token
    # mask out any tokens to padding tokens that were not originally going to be masked
    labels = data.masked_fill(~mask, pad_token_id)        # the label of masked tokens
    return masked_input, labels

class SCDataset(Dataset):
    def __init__(self, data, device, data_dir):
        super().__init__()
        self.data = data
        self.device = device
        self.data_dir = data_dir

    def __getitem__(self, index):
        #rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = sparse.load_npz(self.data_dir + '/' + self.data[index]).toarray()
        full_seq = torch.from_numpy(full_seq).squeeze().to(torch.float32)
        full_seq = torch.cat((full_seq, torch.tensor([0.]))).to(self.device)
        return full_seq

    def __len__(self):
        return self.data.shape[0] 

def retain_non_masked_values(tensor1, tensor2, mask_token = MASK_TOKEN_ID):
    mask = tensor2 != mask_token
    result1 = tensor1[mask]
    result2 = tensor2[mask]
    return result1, result2


data = np.array([f'cell{i}.npz' for i in range(cell_num)])

print(f"Num cells: {len(data)}; Num genes: {len(gene_list)}")

#Preprocess Gene ontology weighted graph
edge_list = get_similarity_network(gene_list, data_path = args.go_data_path, num_similar_genes_go_graph=20)
sim_network = GeneSimNetwork(edge_list, gene_list)
G_go = sim_network.edge_index
G_go_weight = sim_network.edge_weight

data_train, data_val = train_test_split(data, test_size=VALID_RATE, random_state=SEED)

train_dataset = SCDataset(data_train, device = device, data_dir = args.data_path)
val_dataset = SCDataset(data_val, device = device, data_dir = args.data_path)

train_sampler = DistributedSampler(train_dataset)
val_sampler = SequentialDistributedSampler(val_dataset, batch_size=VALID_BATCH_SIZE, world_size=world_size)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE, sampler=val_sampler)

model_params = {
    "max_seq_len": SEQ_LEN,                        # max length of sequence
    "top_seq_len": 4096,
    "base_dim": 200,                                # dim of tokens
    "mini_enc_depth": 2,                              # layers in mini encoder
    "mini_enc_heads": 8,                              # num of heads in mini encoder
    "mini_enc_dim_head": 64,                       # dim of heads in mini encoder
    "large_dim": 1280,                                # dim of tokens in large encoder
    "large_enc_depth":42,                              # layers in large encoder
    "large_enc_heads":32,                              # num of heads in large encoder
    "large_enc_dim_head": 64,                      # dim of heads in large encoder
    "dec_depth": 2,                              # layers in decoder
    "dec_heads": 8,                              # num of heads decoder
    "dec_dim_head": 64,                      # dim of heads decoder
    "mask_token_thres": -1,                  #values smaller than -1 is a mask
    "g2v_position_emb" : POS_EMBED_USING,
    "G_go" : G_go,
    "G_go_weight" : G_go_weight,
    "go_num_layers" : 1,
    "device" : 'cuda',
    "go_use_gene2vec" : args.go_use_gene2vec, 
    "gene2vec_file" : args.gene2vec_data_path
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
            #print(f"Layer: {name} | Module: {module.__class__.__name__} | Number of Params: {total_params}")
            #print(f"Layer: {name} | Module: {module.__class__.__name__} | Number of Params_grad: {total_params_grad}")
            for key in list(param_num_dict.keys()):
                if key in name:
                    param_num_dict[key] += total_params_grad
                    break

    print(f"All model | Number of Params: {all_params}")
    print(f"All model | Number of Params_grad: {all_params_grad}")
    print(f"Modules param number dict: {param_num_dict}")

model.to(device)
path = args.model_path
if path != 'None':
    ckpt = torch.load(path, map_location = device)
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = True

model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
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
scaler = GradScaler(enabled=(AMP_DTYPE == 'float16'))

if path != 'None' and load_opt_and_sched:
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'scaler_state_dict' in ckpt.keys():
        scaler.load_state_dict(ckpt['scaler_state_dict'])
if COMPILE:
    model = torch.compile(model)
train_log_step = 0
valid_log_step = 0

loss_fn = nn.MSELoss(reduction='mean').to(local_rank)

assert EPOCHS != 0
dist.barrier()

for i in range(args.start_epoch + 1, args.start_epoch + EPOCHS + 1):
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0

    for index, data in enumerate(train_loader):
        index += 1
        data = data.to(device)
        data, labels = data_mask(data)

        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                with autocast(device_type='cuda', dtype=ptdtype):
                    logits = model(data)
                    logits, labels = retain_non_masked_values(logits.squeeze(dim=-1), labels)
                    loss = loss_fn(logits, labels) / GRADIENT_ACCUMULATION
                # Use scaler.scale() to adjust the grad
                scaler.scale(loss).backward()
        else:
            with autocast(device_type='cuda', dtype=ptdtype):
                logits = model(data)
                logits, labels = retain_non_masked_values(logits.squeeze(dim=-1), labels)
                loss = loss_fn(logits, labels) / GRADIENT_ACCUMULATION
            scaler.scale(loss).backward()
            # optimizer update
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # save model
            if is_master: 
                current_time = time.time()
                if current_time - last_save_time >= save_time_interval:
                    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    save_scaler_ckpt(time_stamp, model, optimizer, scheduler, scaler, loss.item(), model_name, ckpt_dir)
                    last_save_time = current_time

        if index % LOG_EVERY == 0:
            if is_master: 
                print(f"- epoch {i} step {index} train_loss {loss.item()} -")
            if is_master and args.use_wandb:
                wandb.log({"train_loss": loss.item(), "learning_rate": scheduler.optimizer.param_groups[0]['lr'], "train_log_step": train_log_step})
                train_log_step += 1
                save_scaler_ckpt(i, model, optimizer, scheduler, scaler, loss.item(), model_name, './latest_checkpoint_4096_48m_mix/')

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
                for val_index, data in enumerate(val_loader):
                    val_index_num += 1
                    data = data.to(device)
                    data, labels = data_mask(data)
                    with autocast(device_type='cuda', dtype=ptdtype):
                        logits = model(data)
                        logits, labels = retain_non_masked_values(logits.squeeze(dim=-1), labels)
                        loss_val = loss_fn(logits, labels)
                    running_loss_val += loss_val.item()
                    if val_index % LOG_EVERY == 0:
                        if is_master and args.use_wandb:
                            wandb.log({"val_step_loss": loss_val.item(), "valid_log_step": valid_log_step})
                            valid_log_step += 1
                        if is_master: 
                            current_time = time.time()
                            if current_time - last_save_time >= save_time_interval:
                                time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                                save_scaler_ckpt(time_stamp, model, optimizer, scheduler, scaler, loss.item(), model_name, ckpt_dir)
                                last_save_time = current_time
                del data, labels, logits
                # gather
                loss_val_all = running_loss_val / val_index_num
                loss_val_reduced = get_reduced(loss_val_all, local_rank, 0, world_size)
                    
            if is_master:
                current_time = time.time()
                time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                print(f'    ==  Epoch: {i} Step {index} Time {time_stamp} | Validation Loss: {loss_val_reduced:.6f}')
                if args.use_wandb:
                    wandb.log({"val_in_train_loss": loss_val_reduced})
            model.train()

    epoch_loss = running_loss / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}')
        if args.use_wandb:
            wandb.log({"train_epoch_loss": epoch_loss, "epoch": i})
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
            for index, data in enumerate(val_loader):
                index += 1
                data = data.to(device)
                data, labels = data_mask(data)
                with autocast(device_type='cuda', dtype=ptdtype):
                    logits = model(data)
                    logits, labels = retain_non_masked_values(logits.squeeze(dim=-1), labels)
                    loss = loss_fn(logits, labels)
                running_loss += loss.item()
                if index % LOG_EVERY == 0:
                    if is_master: 
                        print(f"- epoch {i} step {index} val_loss {loss.item()} -")
                    if is_master and args.use_wandb:
                        wandb.log({"val_loss": loss.item(), "valid_log_step": valid_log_step})
                        valid_log_step += 1
                    if is_master: 
                        current_time = time.time()
                        if current_time - last_save_time >= save_time_interval:
                            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                            save_scaler_ckpt(time_stamp, model, optimizer, scheduler, scaler, loss.item(), model_name, ckpt_dir)
                            last_save_time = current_time
            del data, labels, logits
            # gather
            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
        if is_master:
            print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f}')
            if args.use_wandb:
                wandb.log({"val_epoch_loss": val_loss, "epoch": i})
                
    if is_master:
        save_scaler_ckpt(f"epoch{i}", model, optimizer, scheduler, scaler, epoch_loss, model_name, ckpt_dir)

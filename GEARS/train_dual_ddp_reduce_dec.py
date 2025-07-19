import os
import time
import argparse
import pandas as pd
import scanpy as sc
from os.path import join as pjoin

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from gears_reduce.gears_ddp import GEARS
from gears_reduce import PertData
from gears_reduce.model import *
from gears_reduce.scfm_utils import *
from gears_reduce.utils import print_sys


def main(parser):
    
    # debugging utility in pytorch that identifies source of NaN / infinite values during backpropagation. 
    torch.autograd.set_detect_anomaly(True)
    
    args = parser.parse_args()

    # local rank refers to GPU index used by process on a machine. For example, if a machine has 4 GPUs, local rank will be [0,3]
    local_rank = args.local_rank
    rank = int(os.environ["RANK"])
    is_master = rank == 0

    # initializes the distributed training environment in pytorch using Nvidia Collective Communications Library (NCCL) backend
    # optimized for GPU training. can use Message Passing Interface (MPI) backend, for HPC when running across multiple nodes.
    # init_method = env:// initializes using environment variables such as RANK, WORLD_SIZE, MASTER_ADDR, etc.
    dist.init_process_group(backend='nccl', init_method='env://')
    print('DDP init!')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    world_size = torch.distributed.get_world_size()

    print("local_rank: ", local_rank)

    # presumably "validate every x", is an int with default=1
    valid_every = args.valid_every

    # get data
    pert_data = PertData(args.data_dir)
    # load dataset in paper: norman, adamson, dixit.
    try:
        if args.data_name in ['norman', 'adamson', 'dixit']:
            pert_data.load(data_name = args.data_name)
        else:
            print('load data')
            pert_data.load(data_path = pjoin(args.data_dir, args.data_name))
    except:
        adata = sc.read_h5ad(pjoin(args.data_dir, args.data_name+'.h5ad'))
        adata.uns['log1p'] = {}
        adata.uns['log1p']['base'] = None
        pert_data.new_data_process(dataset_name=args.data_name, adata=adata)
    
    #with open(f"{args.data_dir}/{args.data_name}/ens_id_list.txt", 'r') as f:
    #    input_genes_ens_ids = [line.rstrip('\n') for line in f.readlines()]
    input_genes_ens_ids = pert_data.adata.var.index.tolist()
    with open(args.scfm_genes_list_path, 'r') as f:
        scfm_genes_ens_ids = [line.rstrip('\n') for line in f.readlines()]
    if is_master:
        print("target and scfm genes intersect: ", len(np.intersect1d(input_genes_ens_ids, scfm_genes_ens_ids)))

    # specify data split
    pert_data.prepare_split(split = args.split, seed = args.seed, train_gene_set_size=args.train_gene_set_size)
    # get dataloader with batch size
    pert_data.get_dataloader(batch_size = args.batch_size, test_batch_size = args.test_batch_size)

    # set up and train a model
    if is_master:
        print_sys("EXPERIMENT: " + args.run_name)
    gears_model = GEARS(pert_data, 
                        local_rank = local_rank,
                        is_master = is_master,
                        world_size=world_size,
                        weight_bias_track = args.wandb,
                        train_bs=args.batch_size,
                        test_bs=args.test_batch_size,
                        device = device, 
                        exp_name = args.run_name)
    gears_model.model_initialize(hidden_size = args.hidden_size, 
                                 model_type = args.model_type,
                                 bin_set=args.bin_set,
                                 load_path=args.singlecell_model_path,
                                 finetune_method=args.finetune_method,
                                 accumulation_steps=args.accumulation_steps,
                                 mode=args.mode,
                                 input_genes_ens_ids = input_genes_ens_ids,
                                scfm_genes_ens_ids = scfm_genes_ens_ids,
                                scfm_hyper_params_path = args.scfm_hyper_params_path,
                                scfm_ckpt_path = args.scfm_ckpt_path,
                                scfm_class=DualEncoderSCFM,
                                key_enc="merged_decodings",
                                scfm_gene2vec_file = args.scfm_gene2vec_file,
                                record_pred=args.record_pred)
    print("finished initialization")
    gears_model.train(epochs = args.epochs, result_dir=args.result_dir,lr=args.lr, valid_every = valid_every)

    # save params
    if is_master:
        param_pd = pd.DataFrame(vars(args), index=['params']).T
        param_pd.to_csv(f'{args.result_dir}/params.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GEARS')

    parser.add_argument("--local-rank", type=int, default=-1, help='Local process rank.')
    parser.add_argument('--data_dir', type=str, default='/home/ding.bai/pert_new/mywork/data')
    parser.add_argument('--data_name', type=str, default='norman')
    parser.add_argument('--split', type=str, default='simulation')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valid_every', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--train_gene_set_size', type=float, default=0.75)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--bin_set', type=str, default=None)
    parser.add_argument('--singlecell_model_path', type=str, default=None)
    parser.add_argument('--finetune_method', type=str, default=None)
    parser.add_argument('--mode', type=str, default='v1')
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('-wandb', action='store_true')
    parser.add_argument('-record_pred', action='store_true')

    parser.add_argument('--scfm_genes_list_path', type=str, 
                        default="/home/ding.bai/scFound/check_datasets/selected/selected_genes_27k.txt")
    parser.add_argument('--scfm_hyper_params_path', type=str, 
                        default="/lustre/scratch/shared-folders/bio_project/dingbai/ckpts/gocont_27k_38m_pretrain.pkl")
    parser.add_argument('--scfm_ckpt_path', type=str, 
                        default="/l/users/ding.bai/Geneformer/gocont_4096_48m_pretrain_1b_mix_2024-01-31_16-04-29-004.pth")
    parser.add_argument('--run_name', type=str, 
                        default="default_run")
    parser.add_argument('--scfm_gene2vec_file', type=str, 
                        default="/l/users/ding.bai/Geneformer/checkpoints_ScFM/selected_gene2vec_27k.npy")


    parser.add_argument('--lr', type=float, default=1e-3)
    


    main(parser)
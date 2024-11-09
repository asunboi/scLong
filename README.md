# scLong

### Single-cell Foundation Model

# Install

scipy

numpy

pickle5

scanpy

PyTorch (GPU)

PyTorch Geometric

pandas

scikit-learn

transformers

# Pretraining Data

The pre-processed data for pretraining can be downloaded from this directory. 

[_release_cells_](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/EpvKzQW4hI5Bnb88-iM7vE0B_e2_U5r_ZGXb_FILCLTw3Q?e=TAmKk5)

These released datasets were compressed into tens of chunks. To use the data for pretraining, each cell needs to be extracted and saved as separate sparse matrix (npz) files.
```
python chunk_to_cells.py \
  --input_dir release_cells \
  --start_idx 0 \
  --end_idx 48024242 \
  --output_dir cell_files \
```

Other necessary files for pertaining: 

Gene2Vec initialization array [_selected_gene2vec_27k.npy_](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/EpvKzQW4hI5Bnb88-iM7vE0B_e2_U5r_ZGXb_FILCLTw3Q?e=TAmKk5)

Gene Ensembl id list [_selected_genes_27k.txt_](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/EpvKzQW4hI5Bnb88-iM7vE0B_e2_U5r_ZGXb_FILCLTw3Q?e=TAmKk5)

The Gene Ontology graph [_human_ens_gene2go_graph.csv_](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/EpvKzQW4hI5Bnb88-iM7vE0B_e2_U5r_ZGXb_FILCLTw3Q?e=TAmKk5)

Pretraining model initialization hyper-parameters, the Gene Ontology graph included [_gocont_4096_48m_pretrain_1b_mix.pkl_](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/EpvKzQW4hI5Bnb88-iM7vE0B_e2_U5r_ZGXb_FILCLTw3Q?e=TAmKk5)


# Pretrain

- Pre-train your own models
```
python pretrain_gocont_4096_all_1b_mix.py \
  --cell_num 48024242 \
  --data_path cell_files \
  --epoch 30 --batch_size 1 \
  --learning_rate 0.00005 \
  --ckpt_dir ckpts_1b/ \
  --model_name gocont_4096_48m_pretrain_1b_mix \
  --gene2vec_data_path selected_gene2vec_27k.npy \
  --gene_list_path selected_genes_27k.txt \
  --go_data_path human_ens_gene2go_graph.csv \
  --grad_acc 200
```


# Downstream tasks

Before launching any downstream task, download our pretrained model checkpoint here:

[_gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth_](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/EpvKzQW4hI5Bnb88-iM7vE0B_e2_U5r_ZGXb_FILCLTw3Q?e=TAmKk5)

## Gene perturbation effects prediction

This task is in the directory _./GEARS_. 

The Pre-processed Norman dataset will be downloaded when runned the first time.

The additional directory that needs to be downloaded is: [_gears_data_](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ding_bai_mbzuai_ac_ae/EpvKzQW4hI5Bnb88-iM7vE0B_e2_U5r_ZGXb_FILCLTw3Q?e=TAmKk5)

Then run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nnodes=1 --nproc_per_node 4 \
    train_dual_ddp_reduce_dec.py \
    --data_dir=./data \
    --data_name=norman \
    --seed=1 \
    --result_dir=res/output/ddp-hs1024-1b \
    --seed=${seed} \
    --epochs=${epochs} \
    --batch_size=16 \
    --valid_every=1 \
    --test_batch_size=16 \
    --hidden_size=1024 \
    --model_type GO_CONT_SCFM \
    --finetune_method frozen \
    --mode v1 \
    --accumulation_steps 8 \
    --lr=0.001 \
    -record_pred \
    --scfm_genes_list_path selected_genes_27k.txt \
    --scfm_hyper_params_path gocont_4096_48m_pretrain_1b_mix.pkl \
    --scfm_ckpt_path gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth \
    --run_name scfm_gears_downstream_hs1024-1b-02-05_dec 
```

## Gene profile prediction

This task is in the directory _./gene\_profile_. Please check the README inside it. 

## Cancer drug response

This task is in the directory _./drug\_response_. Please check the README inside it. 

## Drug combination response

This task is in the directory _./drug\_combination_. Please check the README inside it. 

## GRN inference

This task is in the directory _./GRN\_inference_. Please check the README inside it. 

# References

[scBERT](https://github.com/TencentAILabHealthcare/scBERT)

[Geneformer](https://huggingface.co/ctheodoris/Geneformer)

[Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M)

[GEARS](https://github.com/snap-stanford/GEARS)

[DeepCE](https://github.com/stealthcopter/deepce)

[DeepCDR](https://github.com/kimmo1019/DeepCDR)

[DeepDDS](https://github.com/Sinwang404/DeepDDS/tree/master)

[DeepSEM](https://github.com/HantaoShu/DeepSEM)




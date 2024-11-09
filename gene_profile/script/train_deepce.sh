#!/usr/bin/env bash

for i in 1 2 3 4 5
do
    python ../DeepCE/main_deepce.py --drug_file "../DeepCE/data/drugs_smiles.csv" \
    --gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train.csv" \
    --dev_file "../DeepCE/data/signature_dev.csv" --test_file "../DeepCE/data/signature_test.csv" \
    --dropout 0.1 --batch_size 32 --max_epoch 100 --seed $i #--scfm
done

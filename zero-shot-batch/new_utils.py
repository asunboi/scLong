

import os
from typing import List, Optional, Tuple, Dict, Union
import pandas as pd

import seaborn as sns
import scanpy as sc

import numpy as np

import torch
import pickle5 as pickle
import scipy.sparse as sparse
import sys
from tqdm import tqdm
import argparse
from copy import deepcopy
import scib
from scanpy import AnnData


def print_sys(s):
    print(s, flush = True, file = sys.stderr)


def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "cell_type",
    embedding_key: str = "X_scGPT"
) -> Dict:
    
    # if adata.uns["neighbors"] exists, remove it to make sure the optimal 
    # clustering is calculated for the correct embedding
    # print a warning for the user
    if "neighbors" in adata.uns:        
        print_sys(f"neighbors in adata.uns found \n {adata.uns['neighbors']} \nto make sure the optimal clustering is calculated for the correct embedding, removing neighbors from adata.uns.\nOverwriting calculation of neighbors with sc.pp.neighbors(adata, use_rep={embedding_key}).")
        adata.uns.pop("neighbors", None)
        sc.pp.neighbors(adata, use_rep=embedding_key)
        print_sys(f"neighbors in adata.uns removed, new neighbors calculated: {adata.uns['neighbors']}")


    # in case just one batch scib.metrics.metrics doesn't work 
    # call them separately
    results_dict = dict()

    # Calculate this only if there are multiple batches
    if len(adata.obs[batch_key].unique()) > 1:

        results_dict["ASW_label/batch"] = scib.metrics.silhouette_batch(
            adata, 
            batch_key,
            label_key, 
            embed=embedding_key, 
            metric="euclidean",
            return_all=False,
            verbose=False
        )


    print_sys(
        "\n".join([f"{k}: {v:.4f}" for k, v in results_dict.items()])
    )

    # remove nan value in result_dict
    results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}

    return results_dict


def evaluate(adata_,
             batch_key = None,
             label_key: list = ['celltype'],
             embedding_key: str = "X_scLong",
             res_path = '') -> pd.DataFrame:
        
    met_df = pd.DataFrame(columns = ["metric", "label", "value"])

    # get unique values in label_key preserving the order
    label_cols = [x for i, x in enumerate(label_key) 
                    if x not in label_key[:i]]
    # remove label columns that are not in adata_.obs
    label_cols = [x for x in label_cols if x in adata_.obs.columns]

    if len(label_cols) == 0:
        msg = f"No label columns {label_key} found in adata.obs"
        raise ValueError(msg)
    
    # check if the embeddings are in adata
    if embedding_key not in adata_.obsm.keys():
        msg = f"Embeddings {embedding_key} not found in adata.obsm"
        raise ValueError(msg)
    
    for label in label_cols:
        
        metrics = eval_scib_metrics(adata_,
                                            batch_key = batch_key, 
                                            label_key = label,
                                            embedding_key = embedding_key)
        for metric in metrics.keys():
            # add row to the dataframe
            met_df.loc[len(met_df)] = [metric, label, metrics[metric]]
    
    met_df.to_csv(res_path, index = False)

    ce_array = adata_.obsm[embedding_key]
    np.save(f'{res_path[:-4]}_ce.npy', ce_array)
    
    
    return met_df
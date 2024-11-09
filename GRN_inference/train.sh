python main.py \
    --task celltype_GRN \
    --data_file /data3/ruiyi/deepsem/input_hesc.csv \
    --net_file /data3/ruiyi/deepsem/label_hesc.csv \
    --setting new \
    --alpha 0.1 \
    --beta 0.01 \
    --n_epochs 100 \
    --save_name /data3/ruiyi/deepsem/hesc_scfm_nopre_matrix_out 
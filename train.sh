#!/bin/bash

MODE=${1:-vocalized}

if [ "$MODE" != "vocalized" ] && [ "$MODE" != "silent" ]; then
    echo "Error: Invalid mode '$MODE'."
    echo "Usage: ./train.sh [vocalized|silent]"
    exit 1
fi

if [ "$MODE" == "vocalized" ]; then
    X_PATH="./processed/X_vocalized.npy"
    Y_PATH="./processed/y_vocalized.npy"
    META_PATH="./processed/meta_vocalized.npz"
    PRETRAINED_ARG=""
    echo ">>> Running training in VOCALIZED mode"
else
    X_PATH="./processed/X_silent.npy"
    Y_PATH="./processed/y_silent.npy"
    META_PATH="./processed/meta_silent.npz"
    PRETRAINED_ARG="--pretrained ./runs/8ch/vocalized.pt"
    echo ">>> Running training in SILENT mode (using pretrained weights)"
fi


python pca.py \
    --mode "$MODE" \
    --x_path "$X_PATH" \
    --y_path "$Y_PATH" \
    --meta_path "$META_PATH" \
    --out_dir ./runs/8ch \
    $PRETRAINED_ARG \
    --channel_indices 1,2,3,4,5,6,7,8,9,10,11,12,13 \
    --split_mode random \
    --val_fraction 0.15 \
    --window_ms 1400 \
    --width_factor 2.0 \
    --n_res_blocks 4 \
    --use_se \
    --channel_attention \
    --dropout 0.5 \
    --channel_dropout 0.4 \
    --mixup_alpha 0.0 \
    --lr 5e-5 \
    --weight_decay 2e-2 \
    --epochs 200 \
    --batch_size 32 \
    --patience 20 \
    --plot_confusion \
    --seed 0
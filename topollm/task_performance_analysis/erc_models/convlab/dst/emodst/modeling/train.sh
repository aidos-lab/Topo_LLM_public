#!/bin/sh

python train_contextbert_ertod.py \
        --exp_id "folder_to_save_model" \
        --data_dir "/Users/shutong/Projects/EmoLoop/required_files/dst/data" \
        --pretrained_model_dir "/Users/shutong/Projects/EmoLoop/required_files/dst/bert-base-uncased" \
        --seed 42 \
        --epochs 5 \
        --do_train \
        --use_context \
        --emotion 
        # --augment fearful apologetic abusive excited \
        # --augment_src to-inferred \
        # --dialog_state \
        # --valence \
        # --elicitor \
        # --conduct \
        # --distance_loss
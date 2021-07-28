#!/bin/bash
MODEL_DIR=results
TASK_NAME=MRPC
CHECKPOINT_DIR=checkpoint/base_bert/${TASK_NAME}
DATASET_DIR=data/glue_data/${TASK_NAME}
OUTPUT_DIR=${MODEL_DIR}/${TASK_NAME}_search_lr_32
#export PYTHONPATH="$(pwd)"

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_nlp.py \
CUDA_VISIBLE_DEVICES=1 python train_nlp.py \
  --task_name $TASK_NAME \
  --model_type bert \
  --model_name_or_path ${CHECKPOINT_DIR} \
  --auto_continue true \
  --do_train \
  --do_eval \
  --do_lower_case \
  --hidden_size 128 \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 128 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --weight_decay 3e-4 \
  --alpha_kd 0.5 \
  --seed 42 \
  --total_iters 50000 \
  --val_interval 500 \
  --save_interval 500 \
  --display_interval 20 \
  --data_dir ${DATASET_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --available_gpus 3 \
  --need_gpus 1 \
  --conf_file ./confs/bert_glue.json

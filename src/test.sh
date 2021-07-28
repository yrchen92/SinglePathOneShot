#!/bin/bash
MODEL_DIR=results
TASK_NAME=MRPC
CHECKPOINT_DIR=checkpoint/base_bert/${TASK_NAME}
DATASET_DIR=data/glue_data/${TASK_NAME}
OUTPUT_DIR=${MODEL_DIR}/test
#export PYTHONPATH="$(pwd)"

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_nlp.py \
CUDA_VISIBLE_DEVICES=3 python train_nlp.py \
  --task_name $TASK_NAME \
  --model_type bert \
  --model_name_or_path ${CHECKPOINT_DIR} \
  --eval \
  --do_train \
  --do_eval \
  --do_lower_case \
  --hidden_size 256 \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 128 \
  --per_gpu_eval_batch_size 128 \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --seed 42 \
  --total_iters 150000 \
  --val_interval 10000 \
  --save_interval 10000 \
  --display_interval 1 \
  --data_dir ${DATASET_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --available_gpus 3 \
  --need_gpus 1 \
  --conf_file ./confs/bert_glue.json

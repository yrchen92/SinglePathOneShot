#!/bin/bash
MODEL_DIR=results
TASK_NAME=SST-2
CHECKPOINT_DIR=checkpoint/base_bert/${TASK_NAME}
DATASET_DIR=data/glue_data/${TASK_NAME}
#export PYTHONPATH="$(pwd)"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train_nlp.py \
  --task_name $TASK_NAME \
  --model_type bert \
  --model_name_or_path ${CHECKPOINT_DIR} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --hidden_size 512 \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 128 \
  --per_gpu_eval_batch_size 128 \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --seed 42 \
  --val_interval 20 \
  --save_interval 20 \
  --auto_continue true \
  --data_dir ${DATASET_DIR} \
  --output_dir ${MODEL_DIR}/${TASK_NAME} \
  --available_gpus 0 \
  --need_gpus 1 \
  --conf_file ./confs/bert_glue.json

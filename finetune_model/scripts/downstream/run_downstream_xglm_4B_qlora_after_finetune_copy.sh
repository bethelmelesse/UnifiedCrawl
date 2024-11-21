#!/usr/bin/env bash

MODEL="models/finetuned_output/amh/facebook/xglm-4.5B-epoch1-lr1e-4-qlora"
LANGUAGE="amh"
DATASET="amhqa"

BATCH_SIZE=4
ACCUM_STEPS=$((4 / BATCH_SIZE))
LR=3e-5
EPOCH=3

OUTPUT_DIR="models/downstream_output/$LANGUAGE/$MODEL-epoch$EPOCH-lr$LR-qlora2"
LOGGING_DIR="$OUTPUT_DIR/tensorboard"

# python -m debugpy --listen localhost:5678 --wait-for-client ./finetune_model/run_clm_downstream.py \
python ./finetune_model/run_clm_downstream.py \
    --finetune_with_qlora=True \
    --model_name_or_path="$MODEL" \
    --low_cpu_mem_usage=True \
    --dataset_name="$DATASET" \
    --language=$LANGUAGE \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOGGING_DIR \
    --evaluation_strategy=epoch \
    --overwrite_output_dir=False \
    --logging_steps=1 \
    --do_train=True \
    --do_eval=True \
    --per_device_train_batch_size=$BATCH_SIZE \
    --per_device_eval_batch_size=$BATCH_SIZE \
    --eval_accumulation_steps=$ACCUM_STEPS \
    --gradient_accumulation_steps=$ACCUM_STEPS \
    --learning_rate=$LR \
    --weight_decay=0.01 \
    --warmup_ratio=0.05 \
    --num_train_epochs=$EPOCH \
    --lr_scheduler_type=cosine \
    --bf16 \
    --gradient_checkpointing=False \
    --optim=paged_adamw_8bit \
    --streaming=False \
    --preprocessing_num_workers=4 \
    --dataloader_num_workers=1 \
    --data_seed=42 \
    --save_strategy=epoch \
    --save_total_limit=4 \
    --torch_dtype=bfloat16 \
    --prediction_loss_only
#!/usr/bin/env bash

MODEL="facebook/xglm-564M"
LANGUAGE="amh"
DATASET="datasets/${LANGUAGE}_dedup_filtered/*.arrow"

BATCH_SIZE=4
ACCUM_STEPS=$((8 / BATCH_SIZE))
LR=7e-5         #2.5e-4
EPOCH=2

OUTPUT_DIR="models/finetuned_output/$LANGUAGE/$MODEL-epoch$EPOCH-lr$LR-qlora"
LOGGING_DIR="$OUTPUT_DIR/tensorboard"

# python -m debugpy --listen localhost:5678 --wait-for-client ./finetune_model/run_clm.py \
python ./finetune_model/run_clm.py \
    --finetune_with_qlora=True \
    --model_name_or_path="$MODEL" \
    --low_cpu_mem_usage=True \
    --train_file="$DATASET" \
    --block_size=1000 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOGGING_DIR \
    --validation_split_percentage=3 \
    --evaluation_strategy=steps \
    --eval_steps=0.1 \
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
    --streaming=True \
    --preprocessing_num_workers=-1 \
    --dataloader_num_workers=4 \
    --data_seed=42 \
    --save_strategy=steps \
    --save_steps=0.34 \
    --save_total_limit=1 \
    --torch_dtype=bfloat16 \
    --prediction_loss_only
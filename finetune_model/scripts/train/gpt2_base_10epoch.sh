#!/usr/bin/env bash

MODEL=gpt2
LANGUAGE="amh"
TOKENIZER="models/tokenizer/gpt2-${LANGUAGE}-tokenizer"
DATASET="datasets/${LANGUAGE}_dedup_filtered/*.arrow"

BATCH_SIZE=1
ACCUM_STEPS=$((32 / BATCH_SIZE))
LR=2e-4
EPOCH=10

OUTPUT_DIR="models/trained_output/$LANGUAGE/$MODEL-epoch$EPOCH-lr$LR"
LOGGING_DIR="$OUTPUT_DIR/tensorboard"


# python -m debugpy --listen localhost:5678 --wait-for-client ./finetune_model/run_clm.py \
python ./finetune_model/run_clm.py\
    --finetune_with_qlora=False \
    --model_type=$MODEL \
    --tokenizer_name=$TOKENIZER \
    --low_cpu_mem_usage=True \
    --train_file="$DATASET" \
    --block_size=1000 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOGGING_DIR \
    --validation_split_percentage=3 \
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
    --optim=adamw_torch \
    --streaming=True \
    --preprocessing_num_workers=-1 \
    --dataloader_num_workers=4 \
    --data_seed=42 \
    --save_strategy=epoch \
    --save_total_limit=5 \
    --torch_dtype=bfloat16 \
    --prediction_loss_only

# # TODO: Change torch_dtype to suport quantization  --per_device_eval_batch_size=1 --eval_accumulation_steps=8 --dataloader_num_workers=4 --auto_find_batch_size=True

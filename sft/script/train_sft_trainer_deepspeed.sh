#!/bin/bash

OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=output_sft_trainer_bloom7bmt
fi
mkdir -p $OUTPUT
echo "output dir: $OUTPUT"

export WANDB_PROJECT=$OUTPUT

nohup torchrun --nproc_per_node=8    train/train_sft_trainer.py \
    --model_name_or_path  /data/pretrained_ckpt/bloomz-7b1-mt/  \
    --train_file data/big_data/zh_alpaca_gpt3.5_gpt4_sharegpt_belle_coig_evol_instruct_sharetranslate100w.train.json \
    --torch_dtype auto \
    --fp16 True \
    --seed 1234 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing True \
    --output_dir $OUTPUT   \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --model_max_length 1024 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.0001 \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --deepspeed config/deepspeed_config_stage3.json \
    --logging_steps 10   >$OUTPUT/train.log 2>&1 &
    
 
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=output_lora_qianyue_summary
fi
mkdir -p $OUTPUT
echo "output dir: $OUTPUT"

export WANDB_PROJECT=$OUTPUT

torchrun --nproc_per_node=8    train/train_sft_trainer.py \
    --model_name_or_path  /code/project/nlp2agi/output_sft_trainer_bloomz-7b/epoch_13  \
    --train_file /code/project/qianyue/data/qianyue_train0619.jsonl \
    --use_lora True \
    --lora_config config/lora/lora_config_bloom.json \
    --torch_dtype auto \
    --fp16 True \
    --seed 1234 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing True \
    --output_dir $OUTPUT   \
    --num_train_epochs 5\
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --model_max_length 4096 \
    --save_total_limit 10 \
    --learning_rate 1e-3 \
    --weight_decay 0.0001 \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --deepspeed config/deepspeed_config_stage2.json \
    --logging_steps 1 

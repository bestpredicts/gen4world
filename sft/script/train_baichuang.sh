#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=output_homegpt_bc
fi
mkdir -p $OUTPUT
echo "output dir: $OUTPUT"

export WANDB_PROJECT=homegpt

torchrun --nproc_per_node=8    train/train_sft_trainer.py \
    --model_name_or_path  /code/project/nlp2agi/output_sft_accelerate_baichuan-7B  \
    --train_file /code/project/nlp2agi/data/design_gpt/train_designgpt_train0615.jsonl \
    --fp16 True \
    --seed 1234 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing True \
    --output_dir $OUTPUT   \
    --num_train_epochs 2\
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_steps 100 \
    --model_max_length 2048 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.0001 \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --deepspeed config/deepspeed_config_stage3.json \
    --logging_steps 1 

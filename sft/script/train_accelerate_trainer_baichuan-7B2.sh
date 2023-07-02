#!/bin/bash
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=output_sft_accelerate_baichuan-7B-0627-shuffle
fi
mkdir -p $OUTPUT
echo "output dir: $OUTPUT"
export WANDB_PROJECT=output_sft_accelerate_baichuan-7B-0622
PLM=/code/PLM/baichuan-7B
DATA=/code/project/sft_data/sft_data/belle_paper_add_wizard_ocra_belle13w.jsonl

GRADIENT_ACCUMULATION_STEPS=8

nohup accelerate  launch  --config_file=config/80g_new/default_config1.yaml  \
--mixed_precision="fp16" --zero_stage=3 --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS --gradient_clipping=1.0 --offload_param_device="none" --offload_optimizer_device="none" --zero3_save_16bit_model="true" \
train/train_sft_accelerate.py \
--train_file=$DATA \
--model_name_or_path=$PLM \
--with_tracking \
--report_to wandb \
--seed=1234 \
--output_dir=$OUTPUT \
--max_length=4096 \
--num_train_epochs=5 \
--learning_rate=7e-6 \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=2 \
--gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
--checkpointing_steps=1000 \
--lr_scheduler_type="cosine" \
--warmup_ratio=0.01 \
--weight_decay=0.00001 \
--checkpoints_total_limit=10 \
--eval_step=10 \
--wandb_project=$WANDB_PROJECT \
--gradient_checkpointing_enable >$OUTPUT/train1.log 2>&1 &


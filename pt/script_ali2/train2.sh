#!/bin/bash
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none



OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=moss7b_pretrain_ct
fi
mkdir -p $OUTPUT
echo "output dir: $OUTPUT"
export WANDB_PROJECT=moss7b_pretrain
PLM=/data/dengyong/PLM/moss-base-7b
DATA=/data/dengyong/pt_data/instruction_merge_set_pt.jsonl



GRADIENT_ACCUMULATION_STEPS=8

nohup accelerate  launch  --config_file=config/80g_new/default_config1.yaml  \
--mixed_precision="bf16" --zero_stage=3 --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS --gradient_clipping=1.0 --offload_param_device="none" --offload_optimizer_device="none" --zero3_save_16bit_model="true" \
train/train_sft_accelerate.py \
--train_file=$DATA \
--model_name_or_path=$PLM \
--with_tracking \
--report_to wandb \
--seed=1234 \
--output_dir=$OUTPUT \
--max_length=2048 \
--num_train_epochs=1 \
--learning_rate=1e-4 \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4 \
--gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
--checkpointing_steps=5000 \
--lr_scheduler_type="cosine" \
--warmup_ratio=0.01 \
--weight_decay=0.00001 \
--checkpoints_total_limit=10 \
--log_step=10 \
--wandb_project=$WANDB_PROJECT \
--gradient_checkpointing_enable >$OUTPUT/train1.log 2>&1 &


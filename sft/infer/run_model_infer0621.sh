#!/bain/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
python gen4all_single_infer.py \
--model_path="/code/project/nlp2agi/output_sft_accelerate_llama65b_ft38w/step_2000" \
--model_type='llama' \
--output_file="test_0410_llama1w.jsonl" \

#!/bain/bash
export CUDA_VISIBLE_DEVICES=0
python gen4all_single_infer.py \
--model_path="/code/project/nlp2agi/output_sft_accelerate_baichuan-7B/step_3000" \
--model_type='bloom' \
--output_file="send_test_baichuang.jsonl" \

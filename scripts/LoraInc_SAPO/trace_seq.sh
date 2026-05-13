#!/bin/bash
set -x



export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_prompt.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path qwen3/Qwen3-8B \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/1-cstance \
   --instruction_file TRACE/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/1-cstance/prompts_train \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 0 \
   --deepspeed configs/ds_configs/stage3_eval.config \
   --run_name trace_round1 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 0 \
   --log_level info

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path qwen3/Qwen3-8B \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/1-cstance \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/trace/outputs/1-cstance/prompts_train/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/1-cstance \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name trace_round1 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.05 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 3000 \
   --max_num_instances_per_predict_task 3000 \
   --log_level info 

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_prompt.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/trace/outputs/1-cstance/adapter_ori \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/2-fomc \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/trace/outputs/1-cstance/prompts_train/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/2-fomc/prompts_train \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 0 \
   --deepspeed configs/ds_configs/stage3_eval.config \
   --run_name trace_round1 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 0 \
   --log_level info

sleep 5




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/trace/outputs/1-cstance/adapter \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/2-fomc \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/trace/outputs/2-fomc/prompts_train/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/2-fomc \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name trace_round2 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.05 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 3000 \
   --max_num_instances_per_predict_task 3000 \
   --log_level info 

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_prompt.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/trace/outputs/2-fomc/adapter_ori \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/3-meet \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/trace/outputs/2-fomc/prompts_train/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/3-meet/prompts_train \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 0 \
   --deepspeed configs/ds_configs/stage3_eval.config \
   --run_name trace_round1 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 0 \
   --log_level info

sleep 5




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/trace/outputs/2-fomc/adapter \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/3-meet \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/trace/outputs/3-meet/prompts_train/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/3-meet \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name trace_round3 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.05 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 3000 \
   --max_num_instances_per_predict_task 3000 \
   --log_level info 

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_prompt.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/trace/outputs/3-meet/adapter_ori \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/4-py150 \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/trace/outputs/3-meet/prompts_train/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/4-py150/prompts_train \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 0 \
   --deepspeed configs/ds_configs/stage3_eval.config \
   --run_name trace_round1 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 0 \
   --log_level info

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/trace/outputs/3-meet/adapter \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/4-py150 \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/trace/outputs/3-meet/prompts_train/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/4-py150 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name trace_round4 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.05 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 3000 \
   --max_num_instances_per_predict_task 3000 \
   --log_level info 

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_prompt.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/trace/outputs/4-py150/adapter_ori \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/5-sciqa \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/trace/outputs/3-meet/prompts_train/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/5-sciqa/prompts_train \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 0 \
   --deepspeed configs/ds_configs/stage3_eval.config \
   --run_name trace_round1 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 0 \
   --log_level info

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/trace/outputs/4-py150/adapter \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/5-sciqa \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/trace/outputs/5-sciqa/prompts_train/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/5-sciqa \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name trace_round5 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.05 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 3000 \
   --max_num_instances_per_predict_task 3000 \
   --log_level info 

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_prompt.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/trace/outputs/5-sciqa/adapter_ori \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/6-numgulecm \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/trace/outputs/5-sciqa/prompts_train/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/6-numgulecm/prompts_train \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 0 \
   --deepspeed configs/ds_configs/stage3_eval.config \
   --run_name trace_round1 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 0 \
   --log_level info

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/trace/outputs/5-sciqa/adapter \
   --data_dir TRACE \
   --task_config_dir CL_configs/order/trace/6-numgulecm \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/trace/outputs/6-numgulecm/prompts_train/instructions.json \
   --suffix_file TRACE/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/trace/outputs/6-numgulecm \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name trace_round5 \
   --max_source_length 1536 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.05 \
   --logging_strategy steps \
   --logging_steps 10 \
   --eval_strategy no \
   --save_strategy no \
   --lamda_1 0 \
   --lamda_2 0 \
   --max_num_instances_per_task 3000 \
   --max_num_instances_per_predict_task 3000 \
   --log_level info 

sleep 5
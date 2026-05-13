#!/bin/bash
set -x


export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_prompt.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path qwen3/Qwen3-8B \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_g1/1-task589 \
   --instruction_file SUPERNI/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/1-task589/prompts_train \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 0 \
   --deepspeed configs/ds_configs/stage3_eval.config \
   --run_name ni_seq_g1_round1 \
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
   --max_num_instances_per_task 250 \
   --max_num_instances_per_predict_task 0 \
   --log_level info

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path qwen3/Qwen3-8B \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_g1/1-task589 \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/1-task589/prompts_train/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/1-task589 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name ni_seq_g1_round1 \
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
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 1000 \
   --log_level info 

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_prompt.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/1-task589/adapter_ori \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_g1/2-task141 \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/1-task589/prompts_train/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/2-task141/prompts_train \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 0 \
   --deepspeed configs/ds_configs/stage3_eval.config \
   --run_name ni_seq_g1_round1 \
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
   --max_num_instances_per_task 250 \
   --max_num_instances_per_predict_task 0 \
   --log_level info

sleep 5




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/1-task589/adapter \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_g1/2-task141 \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/2-task141/prompts_train/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/2-task141 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name ni_seq_g1_round2 \
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
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 1000 \
   --log_level info 

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_prompt.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/2-task141/adapter_ori \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_g1/3-task618 \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/2-task141/prompts_train/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/3-task618/prompts_train \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 0 \
   --deepspeed configs/ds_configs/stage3_eval.config \
   --run_name ni_seq_g1_round1 \
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
   --max_num_instances_per_task 250 \
   --max_num_instances_per_predict_task 0 \
   --log_level info

sleep 5




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/2-task141/adapter \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_g1/3-task618 \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/3-task618/prompts_train/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/3-task618 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name ni_seq_g1_round3 \
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
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 1000 \
   --log_level info 

sleep 5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_prompt.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/3-task618/adapter_ori \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_g1/4-task339 \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/3-task618/prompts_train/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/4-task339/prompts_train \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 0 \
   --deepspeed configs/ds_configs/stage3_eval.config \
   --run_name ni_seq_g1_round1 \
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
   --max_num_instances_per_task 250 \
   --max_num_instances_per_predict_task 0 \
   --log_level info

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/3-task618/adapter \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_g1/4-task339 \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/4-task339/prompts_train/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/4-task339 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name ni_seq_g1_round4 \
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
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 1000 \
   --log_level info 

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_prompt.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/4-task339/adapter_ori \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_g1/5-task360 \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/4-task339/prompts_train/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/5-task360/prompts_train \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 0 \
   --deepspeed configs/ds_configs/stage3_eval.config \
   --run_name ni_seq_g1_round1 \
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
   --max_num_instances_per_task 250 \
   --max_num_instances_per_predict_task 0 \
   --log_level info

sleep 5


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/4-task339/adapter \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_g1/5-task360 \
   --instruction_file saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/5-task360/prompts_train/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/LoraInc_SAPO/qwen3-8b/ni_seq_g1/outputs/5-task360 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name ni_seq_g1_round5 \
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
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 1000 \
   --log_level info 



#!/bin/bash
set -x


export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)
 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path qwen3/Qwen3-8B \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_m1/1-task195 \
   --instruction_file SUPERNI/instructions.json  \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/O-LoRA/qwen3-8b/ni_seq_m1/outputs/1-task195 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name ni_seq_m1_round1 \
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
   --lamda_1 0.5 \
   --lamda_2 0 \
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 1000 \
   --log_level info 

sleep 5




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/O-LoRA/qwen3-8b/ni_seq_m1/outputs/1-task195/adapter \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_m1/2-task360 \
   --instruction_file SUPERNI/instructions.json  \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/O-LoRA/qwen3-8b/ni_seq_m1/outputs/2-task360 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name ni_seq_m1_round2 \
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
   --lamda_1 0.5 \
   --lamda_2 0 \
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 1000 \
   --log_level info 

sleep 5





CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/O-LoRA/qwen3-8b/ni_seq_m1/outputs/2-task360/adapter \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_m1/3-task611 \
   --instruction_file SUPERNI/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/O-LoRA/qwen3-8b/ni_seq_m1/outputs/3-task611 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name ni_seq_m1_round3 \
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
   --lamda_1 0.5 \
   --lamda_2 0 \
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 1000 \
   --log_level info 

sleep 5



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/O-LoRA/qwen3-8b/ni_seq_m1/outputs/3-task611/adapter \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_m1/4-task002 \
   --instruction_file SUPERNI/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/O-LoRA/qwen3-8b/ni_seq_m1/outputs/4-task002 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name ni_seq_m1_round4 \
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
   --lamda_1 0.5 \
   --lamda_2 0 \
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 1000 \
   --log_level info 

sleep 5



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port $port src/run_lorainc.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path saves/O-LoRA/qwen3-8b/ni_seq_m1/outputs/4-task002/adapter \
   --data_dir SUPERNI \
   --task_config_dir CL_configs/order/ni_seq_m1/5-task224 \
   --instruction_file SUPERNI/instructions.json \
   --suffix_file SUPERNI/instructions.json \
   --instruction_strategy single \
   --output_dir saves/O-LoRA/qwen3-8b/ni_seq_m1/outputs/5-task224 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 8 \
   --eval_accumulation_steps 1 \
   --gradient_accumulation_steps 8 \
   --learning_rate 1e-4 \
   --num_train_epochs 10 \
   --deepspeed configs/ds_configs/stage2_llama.config \
   --run_name ni_seq_m1_round5 \
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
   --lamda_1 0.5 \
   --lamda_2 0 \
   --max_num_instances_per_task 1000 \
   --max_num_instances_per_predict_task 1000 \
   --log_level info 



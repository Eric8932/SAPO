#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import gc

import torch
import torch.nn.functional as F
import math

import datasets
import nltk
import numpy as np
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from uie_collator import DataCollatorForUIE
from src.uie_dataset import gen_cache_path

from src.uie_trainer import UIETrainer, DenserEvalCallback, skip_instructions
from transformers import Trainer
from compute_metrics import compute_metrics, compute_grouped_metrics
from convert import convert_seqlora_adapter
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from prompt_gen.prompt_gen import prompt_gen


# off wandb
os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    lora_dim: Optional[int] = field(
        default=8,
        metadata={
            "help": "Intrinsic dimension of the latent space."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the UIE train/dev/test splits."}
    )
    task_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    suffix_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    instruction_strategy: Optional[str] = field(
        default='single', metadata={
            "help": "How many different instructions to use? Support 'single' and 'multiple' mode."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=400, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={"help": "The maximum number of instances we will consider for each validation task."}
    )
    max_num_instances_per_predict_task: int = field(
        default=400, metadata={"help": "The maximum number of instances we will consider for each test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )


@dataclass
class UIETrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use computing time to gain more memory"}
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})
    lamda_1: float = field(default = 0.5)
    lamda_2: float = field(default = 0)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UIETrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    full_path = Path(training_args.output_dir)

    full_path.mkdir(parents=True, exist_ok=True)
  
    set_seed(training_args.seed)
    data_cache_dir = gen_cache_path(training_args.output_dir, data_args)


    if 'adapter' in model_args.model_name_or_path:
        config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir = model_args.cache_dir,
            use_fast = model_args.use_fast_tokenizer,
            revision = model_args.model_revision,
            use_auth_token = True if model_args.use_auth_token else None,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id

    model_class = AutoModelForCausalLM 
    tokenizer.padding_side = 'left'
    
    if 'adapter' in model_args.model_name_or_path: # add lora-adapter to the original model
        model = model_class.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
    else:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None
        )

    model.resize_token_embeddings(len(tokenizer))
    if hasattr(model, 'generation_config'):
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id =  tokenizer.eos_token_id
        model.generation_config.pad_token_id =  tokenizer.pad_token_id

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )
        

    # Get the UIE dataset
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForUIE(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        num_examples=data_args.num_examples,
        input_record_file=data_args.input_record_file,
        all_train = True
    )
    training_args.remove_unused_columns = False
    

    print(f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    
    training_args.label_names = ["labels"]
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )
   
   
    instruction_file = data_args.instruction_file
    task_name = data_args.task_config_dir.split("/")[-1].split("-")[-1]
    task_in_trace = 0
    trace_dic = {'cstance':"C-STANCE",'fomc':"FOMC",'meet':"MeetingBank",'py150':'Py150','sciqa':'ScienceQA','numgulecm':'NumGLUE-cm'}
    if task_name in trace_dic:
        task_name = trace_dic[task_name]
        task_in_trace = 1

    task_k = ""
    with open(instruction_file,'r') as f:
        ori_prompts = json.load(f)
    for k,v in ori_prompts.items():
        if task_in_trace:
            if task_name in k:
                ori_prompt = v["instruction"]
                task_k = k
        else:
            if task_name+"_" in k:
                ori_prompt = v["instruction"]
                task_k = k

    num_prompts = 20

    checkpoint = True
    for i in [-1]+list(range(num_prompts)):
        if not os.path.exists(os.path.join(training_args.output_dir,f"instruction_{i}.json")):
            checkpoint = False

    if not checkpoint:
        para_prompts = prompt_gen(ori_prompt,num_prompts)

        print(f"-----Task Name: {task_k} -----")

        print("-----GEN PROMPTS-----")
        print(para_prompts)

        with open(os.path.join(training_args.output_dir,f"instruction_-1.json"), "w") as f:
            json.dump(ori_prompts, f, indent=1, ensure_ascii=False)
        for i,para_prompt in enumerate(para_prompts):
            ori_prompts[task_k]["instruction"] = para_prompt
            with open(os.path.join(training_args.output_dir,f"instruction_{i}.json"), "w") as f:
                json.dump(ori_prompts, f, indent=1, ensure_ascii=False)
        ins_index_list = [-1]+list(range(num_prompts))
    else:
        start_index = -1
        for i in [-1]+list(range(num_prompts)):
            if os.path.exists(os.path.join(training_args.output_dir,f"predict_results_{i}.json")):
                start_index = i+1
        ins_index_list = list(range(start_index,20))

    for ins_index in ins_index_list:

        instruction_file = os.path.join(training_args.output_dir,f"instruction_{ins_index}.json")

        raw_datasets = load_dataset(
            os.path.join(CURRENT_DIR, "uie_dataset.py"),
            data_dir=data_args.data_dir,
            task_config_dir=data_args.task_config_dir,
            instruction_file=instruction_file,
            suffix_file=data_args.suffix_file,
            instruction_strategy=data_args.instruction_strategy,
            cache_dir=data_cache_dir, 
            max_num_instances_per_task=data_args.max_num_instances_per_task,
            max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
            max_num_instances_per_predict_task=data_args.max_num_instances_per_predict_task,
            num_examples=data_args.num_examples
        )

        raw_datasets.cleanup_cache_files()

        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        logger.info(f"Total number of training samples: {len(train_dataset)}")

        num_gpus = training_args.world_size
        if num_gpus > 1:
            num_samples = len(train_dataset)
            if num_samples % num_gpus != 0:
                new_size = (num_samples // num_gpus) * num_gpus
                train_dataset = train_dataset.select(range(new_size))
                logger.info(f"Final number of predict samples: {len(train_dataset)}")
               
        eval_metrics = trainer.evaluate(
            eval_dataset=train_dataset, 
            metric_key_prefix=f"predict"
        )

        if trainer.is_world_process_zero():
            loss = eval_metrics.get(f"predict_loss")
            logger.info(f"Result for prompt {ins_index}: Loss = {loss:.4f}")

            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            eval_metrics["predict_samples"] = min(max_train_samples, len(train_dataset))



            with open(os.path.join(training_args.output_dir,f"predict_results_{ins_index}.json"), "w", encoding="utf-8") as f:
                json.dump(eval_metrics, f, indent=4)


        if 'eval_metrics' in locals(): 
            del eval_metrics
        gc.collect()
        torch.cuda.empty_cache()
    


    if trainer.is_world_process_zero():
        predict_eval_loss = 1000
        save_index = -2
        for ins_index in [-1]+list(range(num_prompts)):
            with open(os.path.join(training_args.output_dir,f"predict_results_{ins_index}.json"), "r") as f:
                loss_temp  = json.load(f)["predict_loss"]
                if loss_temp < predict_eval_loss:
                    predict_eval_loss = loss_temp
                    save_index = ins_index
        assert save_index != -2
        import shutil
        src_path = os.path.join(training_args.output_dir, f"instruction_{save_index}.json")
        dst_path = os.path.join(training_args.output_dir, "instructions.json")

        shutil.copyfile(src_path, dst_path)
    



if __name__ == "__main__":
    main()



import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import is_deepspeed_zero3_enabled

from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from src.uie_dataset import ANSWER_PREFIX
from typing import Dict, Union, Any, List, Optional,Tuple
import gc 

def skip_instructions(predictions_ids, tokenizer, model_path="", suffix_l =[],ignore_idx=-100,):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    
    final_predictions = []
    for i,pred in enumerate(predictions):
        pred = pred.strip()

        answer_prefix = suffix_l[i]
        if "llama3.1" in model_path.lower():
            answer_prefix+="assistant"
        if "qwen3" in model_path.lower():
            answer_prefix += "\nassistant\n<think>\n\n</think>\n\n"
       
        if "llama2" in model_path.lower():
            pred = pred.replace("[/INST]","").replace("[INST]","").strip()
        

        if answer_prefix in pred:
            splits = pred.split(answer_prefix)
            final_predictions.append(splits[-1].strip())
        else:
            final_predictions.append('')

    return final_predictions


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.eval_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True


        return control


class UIETrainer(Seq2SeqTrainer):

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()

        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)


        orthogonal_loss = 0.
        if self.args.lamda_1 >0:
            for name, param in self.model.named_parameters():
                if "lora_A" in name:
                    for name_, param_ in self.model.named_parameters():
                        if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                            orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
                            break # target modules have been matched

        l2_loss = 0.
        for name, param in self.model.named_parameters():
            if "loranew_" in name:
                l2_loss += torch.norm(param, p=2)

        lamda_1 = self.args.lamda_1
        lamda_2 = self.args.lamda_2

      
        loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps


        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            kwargs["scale_wrt_gas"] = False

        self.accelerator.backward(loss, **kwargs)

        return loss.detach()



    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        

        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


            if is_torch_xla_available():
                xm.mark_step()

            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)

            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None and not self._is_loss_eval_mode:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()

            if (step + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics

        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()

        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

     
        if self.compute_metrics is not None and all_preds is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
            metrics[f"{metric_key_prefix}_loss_variance"] = all_losses.var().item()

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        output = EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


        return output



    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        

        if hasattr(self, '_is_loss_eval_mode') and self._is_loss_eval_mode:
            
            return super().prediction_step(
                model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
            )


        with torch.no_grad():
            has_labels = all(inputs.get(k) is not None for k in self.label_names) if self.label_names else False
            
            inputs = self._prepare_inputs(inputs)

            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
                else:
                    ignore_keys = []
                    
            gen_kwargs = self._gen_kwargs.copy()
            synced_gpus = gen_kwargs.pop("synced_gpus", is_deepspeed_zero3_enabled())
            
            config_dict = self.model.generation_config.to_dict()
            generation_config = GenerationConfig(**config_dict)
            for k, v in gen_kwargs.items():
                setattr(generation_config, k, v)

            if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
                generation_inputs = inputs[self.model.encoder.main_input_name]
            else:
                generation_inputs = inputs[self.model.main_input_name]
                
            generated_tokens = self.model.generate(
                input_ids=generation_inputs,
                generation_config=generation_config,
                attention_mask=inputs.get("attention_mask"),
                synced_gpus=synced_gpus,
            )
            
            bs, source_len = inputs['input_ids'].shape
            if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
                max_length = source_len + generation_config.max_new_tokens
            else:
                max_length = generation_config.max_new_tokens

            if generated_tokens.shape[-1] < max_length:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

            loss = None
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, _ = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.detach().mean()

        labels = None
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
            if labels.shape[-1] < generation_config.max_new_tokens:
                labels = self._pad_tensors_to_max_len(labels, generation_config.max_new_tokens)

        return (loss, generated_tokens, labels)
    




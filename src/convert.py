import os
import json
import shutil

def convert_seqlora_adapter(source_adapter_path):
    """
    Copies a custom SeqLoRA adapter to a new directory named 'adapter_ori'
    and converts the config file within the new directory to be PEFT-compatible.
    The original 'adapter' directory is left untouched.
    """

    target_adapter_path = os.path.join(os.path.dirname(source_adapter_path), "adapter_ori")

    print(f"    - source Adapter: '{source_adapter_path}'")
    print(f"    - target Adapter: '{target_adapter_path}'")

    if os.path.exists(target_adapter_path):
        print(f"    - Skip: target folder '{target_adapter_path}' Already Exist。")
        return

    try:
        shutil.copytree(source_adapter_path, target_adapter_path)
       
    except Exception as e:
        return 

    config_to_modify_path = os.path.join(target_adapter_path, "adapter_config.json")

    try:
        with open(config_to_modify_path, 'r', encoding='utf-8') as f:
            old_config = json.load(f)
        if "r_sum" not in old_config:
            print("    - Note: Config file does not contain 'r_sum'")
            return

        accumulated_rank = old_config.get("r_sum", 0)

        new_config = {
            "r": accumulated_rank,
            "peft_type": old_config.get("peft_type", "LORA"),
            "task_type": old_config.get("task_type"),
            "base_model_name_or_path": old_config.get("base_model_name_or_path"),
            "target_modules": old_config.get("target_modules"),
            "lora_alpha": old_config.get("lora_alpha"),
            "lora_dropout": old_config.get("lora_dropout", 0.0),
            "fan_in_fan_out": old_config.get("fan_in_fan_out", False),
            "bias": old_config.get("bias", "none"),
            "modules_to_save": old_config.get("modules_to_save"),
            "init_lora_weights": old_config.get("init_lora_weights", True),
            "inference_mode": True,
            "megatron_core": "megatron.core",
            "auto_mapping": None,
            "eva_config": None,
            "exclude_modules": None,
            "layer_replication": None,
            "layers_pattern": None,
            "layers_to_transform": None,
            "megatron_config": None,
            "revision": None,
            "use_dora": False,
            "use_rslora": False,
            "alpha_pattern": {},
            "loftq_config": {},
            "rank_pattern": {},
        }
        
       
        with open(config_to_modify_path, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=4)

    except Exception as e:
        pass


def convert(base_path):
    """
    Main function to traverse directories and apply the conversion.
    """
    if not os.path.isdir(base_path):
        return
    for sub_dir_name in os.listdir(base_path):
        sub_dir_path = os.path.join(base_path, sub_dir_name)
        
        if os.path.isdir(sub_dir_path):
            source_adapter_path = os.path.join(sub_dir_path, "adapter")
            
            if os.path.isdir(source_adapter_path):
                convert_seqlora_adapter(source_adapter_path)

if __name__ == "__main__":
  
    path_A = ""
    
    convert(path_A)

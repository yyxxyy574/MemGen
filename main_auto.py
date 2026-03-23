# auto_main.py
import argparse
from datetime import datetime
import os
import random
import numpy as np
import torch

from common.config import Config
from common.logger import setup_logger

def set_seed(random_seed: int, use_gpu: bool):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False      

def parse_args():
    parser = argparse.ArgumentParser(description="Memory Generator Auto Router")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--options", nargs="+", help="override some settings.")
    return parser.parse_args()

def build_working_dir(config: Config) -> str:
    mode = config.run_cfg.mode
    dataset_name = config.dataset_cfg.name
    model_name = config.model_cfg.model_name.split("/")[-1]
    parent_dir = os.path.join(".cache", mode, dataset_name, model_name)

    max_prompt_aug_num = config.model_cfg.max_prompt_aug_num
    prompt_latents_len = config.model_cfg.weaver.prompt_latents_len
    max_inference_aug_num = config.model_cfg.max_inference_aug_num
    inference_latents_len = config.model_cfg.weaver.inference_latents_len
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    working_dir = f"pn={max_prompt_aug_num}_pl={prompt_latents_len}_in={max_inference_aug_num}_il={inference_latents_len}_{time}" 
    return os.path.join(parent_dir, working_dir)

def main():
    args = parse_args()
    config = Config(args)
    set_seed(config.run_cfg.seed, use_gpu=True)
    working_dir = build_working_dir(config)
    
    config.run_cfg.log_dir = os.path.join(working_dir, "logs")
    setup_logger(output_dir=config.run_cfg.log_dir)

    config_dict = config.to_dict()
    model_name = config_dict.get("model", {}).get("model_name", "").lower()

    # Automatically route to VLM or LLM implementation based on model name
    if "vl" in model_name:
        print("[INFO] VLM detected. Routing to VLM MemGen implementation...")
        from data import get_data_builder
        from memgen.model.modeling_memgen_vlm import VLM_MemGenModel
        from memgen.runner_vlm import VLM_MemGenRunner

        data_builder = get_data_builder(config_dict.get("dataset"))
        model = VLM_MemGenModel.from_config(config_dict.get("model"))
        runner = VLM_MemGenRunner(model=model, data_builder=data_builder, config=config_dict, working_dir=working_dir)
    else:
        print("[INFO] Pure LLM detected. Routing to original MemGen implementation...")
        from data import get_data_builder
        from memgen.model.modeling_memgen import MemGenModel
        from memgen.runner import MemGenRunner

        data_builder = get_data_builder(config_dict.get("dataset"))
        model = MemGenModel.from_config(config_dict.get("model"))
        runner = MemGenRunner(model=model, data_builder=data_builder, config=config_dict, working_dir=working_dir)

    if config.run_cfg.mode == "train":
        runner.train()
    elif config.run_cfg.mode == "evaluate":
        runner.evaluate()

if __name__ == "__main__":
    main()
# memgen/runner_vlm.py
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object
from trl import SFTTrainer
from trl.models import unwrap_model_for_generation
from transformers import TrainerCallback

from memgen.runner import MemGenRunner
from memgen.utils import create_tensorboard, StaticEvalRecorder, gather_objects
from interactions.base_interaction import InteractionDataProto
from interactions.singleturn_interaction import SingleTurnInteractionManager

class MemoryTrackerCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_alloc = torch.cuda.memory_allocated() / (1024**3)
            mem_res = torch.cuda.memory_reserved() / (1024**3)
            if state.is_world_process_zero:
                print(f"\n[Memory Debug - Step {state.global_step} 开始] "
                      f"已分配: {mem_alloc:.2f} GB, 已保留: {mem_res:.2f} GB", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024**3)
            mem_res = torch.cuda.memory_reserved() / (1024**3)
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            if state.is_world_process_zero:
                print(f"[Memory Debug - Step {state.global_step} 结束] "
                      f"已分配: {mem_alloc:.2f} GB, 已保留: {mem_res:.2f} GB, 峰值: {peak_alloc:.2f} GB\n", flush=True)

# ===== 自定义多模态 DataCollator =====
class VLM_MemGenDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        # 匹配 <|im_start|>assistant\n 和 <|im_end|>
        self.start_pattern = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.end_pattern = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def __call__(self, features):
        clean_features = []
        # 过滤掉非张量的数据（如 prompt 原文），防止 processor.pad 报错
        valid_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
        for f in features:
            clean_f = {k: v for k, v in f.items() if k in valid_keys}
            clean_features.append(clean_f)
            
        batch = self.processor.pad(clean_features, return_tensors="pt")
        
        # 默认全部 Mask 为 -100
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100
        
        # 找出 Assistant 实际回复的起始/结束位置，解除 Mask
        for b in range(batch["input_ids"].size(0)):
            seq = batch["input_ids"][b].tolist()
            labels_b = torch.full_like(batch["labels"][b], -100)
            
            i = 0
            found_assistant = False
            while i < len(seq):
                if seq[i : i + len(self.start_pattern)] == self.start_pattern:
                    start_idx = i + len(self.start_pattern)
                    end_idx = len(seq)
                    for j in range(start_idx, len(seq)):
                        if seq[j : j + len(self.end_pattern)] == self.end_pattern:
                            end_idx = j + len(self.end_pattern)
                            break
                    
                    # 仅保留 Assistant 内容的 Loss 计算
                    labels_b[start_idx:end_idx] = batch["input_ids"][b, start_idx:end_idx]
                    found_assistant = True
                    i = end_idx
                else:
                    i += 1
                    
            if found_assistant:
                batch["labels"][b] = labels_b
                
        return batch

# ===== 多模态交互管理器 =====
class VLM_SingleTurnInteractionManager(SingleTurnInteractionManager):
    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        initial_input_ids = gen_batch.batch["input_ids"]
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        rollings = gen_batch
        keys_to_cut = ['input_ids', 'attention_mask']
        cut_batch = self.tensor_fn.cut_to_effective_len(
            {k: v for k, v in rollings.batch.items() if k in keys_to_cut},
            keys=keys_to_cut    
        )
        
        rollings_active = {k: v for k, v in rollings.batch.items() if k not in keys_to_cut}
        rollings_active.update(cut_batch)  

        is_multimodal = 'pixel_values' in rollings_active or 'pixel_values_videos' in rollings_active
        original_max_inf_aug = self.actor_rollout_wg.config.max_inference_aug_num
        
        try:
            if is_multimodal:
                # 简化逻辑：多模态数据只在 Prompt 后加 Memory，中间不再额外加
                self.actor_rollout_wg.config.max_inference_aug_num = 0
            
            gen_kwargs = {k: v for k, v in rollings_active.items() if k not in ["input_ids", "attention_mask"]}
            
            gen_output = self.actor_rollout_wg.generate(
                input_ids=rollings_active["input_ids"], 
                attention_mask=rollings_active["attention_mask"], 
                generation_config=self.generation_config,
                **gen_kwargs
            )
        finally:
            if is_multimodal:
                self.actor_rollout_wg.config.max_inference_aug_num = original_max_inf_aug

        responses_ids = gen_output[:, rollings_active["input_ids"].size(1):]
        responses_ids = self.tensor_fn.erase_after_first_eos(responses_ids, self.tokenizer.eos_token_id)
        
        original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids=None)
        return self._compose_final_output(original_left_side, original_right_side)


# ===== 主运行器 =====
class VLM_MemGenRunner(MemGenRunner):
    def _create_weaver_trainer(self):
        if self.train_weaver_method == "sft":
            # 强制关闭 TRL 内部的 assistant_only_loss，避免 chat_template {% generation %} 报错
            if hasattr(self.weaver_sft_training_args, "assistant_only_loss"):
                self.weaver_sft_training_args.assistant_only_loss = False
            
            weaver_trainer = SFTTrainer(
                model=self.model,
                args=self.weaver_sft_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.weaver_valid_dataset,
                processing_class=self.processing_class, 
                data_collator=VLM_MemGenDataCollator(self.processing_class), # 使用自定义 Collator 遮蔽 Prompt
                callbacks=[MemoryTrackerCallback()],
            )
        elif self.train_weaver_method == 'grpo':
            weaver_trainer = super()._create_weaver_trainer()
            weaver_trainer.add_callback(MemoryTrackerCallback())
        return weaver_trainer

    def _filter_dataset(self, dataset):
        max_len = 1024
        if self.train_weaver and self.train_weaver_method == "sft":
            max_len = self.weaver_sft_training_args.max_length
        elif self.train_weaver and self.train_weaver_method == "grpo":
            max_len = self.weaver_grpo_training_args.max_prompt_length
        elif self.train_trigger and self.train_trigger_method == "grpo":
            max_len = self.trigger_grpo_training_args.max_prompt_length

        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)

        def filter_func(sample):
            try:
                if "messages" in sample:
                    text = self.processing_class.apply_chat_template(sample["messages"], tokenize=False)
                elif "prompt" in sample and sample["prompt"] is not None:
                    text = self.processing_class.apply_chat_template(sample["prompt"], tokenize=False)
                else:
                    return True
                
                seq_len = len(tokenizer(text)["input_ids"])
                
                if seq_len >= max_len:
                    print(f"[数据过滤] 序列长度 {seq_len} >= {max_len}。已被过滤！请在 shell 脚本的 --options 加上: run.weaver.sft.max_length 4096")
                    return False
                return True
            except Exception as e:
                print(f"[数据过滤警告] 无法估算 sample 长度，默认放行。原因: {e}")
                return True
                
        return dataset.filter(filter_func)

    def _static_evaluate(self):
        accelerator = Accelerator()
        if accelerator.is_main_process:
            writer = create_tensorboard(save_dir=self.working_dir)
            save_file = os.path.join(self.interaction_config.output_dir, "answer.json")
            recorder = StaticEvalRecorder(compute_metrics=[self.env_cls.compute_reward], writer=writer, log_file=save_file)
        else:
            writer, recorder = None, None

        batch_size = self.interaction_config.batch_size
        test_dataloader = accelerator.prepare(DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: batch))
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()

        if isinstance(self.generation_manager, SingleTurnInteractionManager):
            self.generation_manager = VLM_SingleTurnInteractionManager(self.processing_class, self.model, self.interaction_config)

        for test_batch in tqdm(test_dataloader, disable=not accelerator.is_main_process):
            with unwrap_model_for_generation(model_wrapped, accelerator) as unwrapped_model:
                is_multimodal = any(
                    "messages" in x and isinstance(x["messages"], list) and any(isinstance(msg.get("content"), list) for msg in x["messages"]) for x in test_batch
                ) or any("images" in x for x in test_batch)

                if is_multimodal and "messages" in test_batch[0]:
                    texts = [self.processing_class.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True) for x in test_batch]
                    images = [x.get("images", None) for x in test_batch] if "images" in test_batch[0] else None
                    prompt_inputs = self.processing_class(text=texts, images=images, padding=True, padding_side="left", return_tensors="pt")
                else:
                    prompts = [x["prompt"] for x in test_batch]
                    prompt_inputs = self.processing_class.apply_chat_template(prompts, add_generation_prompt=True, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=True, return_dict=True)

                gen_batch = InteractionDataProto()
                for k, v in prompt_inputs.items():
                    gen_batch.batch[k] = v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                gen_batch.no_tensor_batch["initial_prompts"] = [x.get("prompt", x.get("messages", "")) for x in test_batch]

                self.generation_manager.actor_rollout_wg = unwrapped_model
                gen_output = self.generation_manager.run_agent_loop(gen_batch)

                completion_ids = gen_output.batch["responses"]
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            local_completions, local_batches = completions, test_batch
            all_completions, all_batches = gather_objects(local_completions), gather_objects(local_batches)

            if accelerator.is_main_process:
                for comps, batch in zip(all_completions, all_batches):
                    recorder.record_batch(comps, batch)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            recorder.finalize()
            if writer: writer.close()
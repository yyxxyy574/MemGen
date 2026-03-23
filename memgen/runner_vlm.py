# memgen/runner_vlm.py
from memgen.runner import MemGenRunner
from trl import SFTTrainer
import torch
from transformers import TrainerCallback

class MemoryTrackerCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_alloc = torch.cuda.memory_allocated() / (1024**3)
            mem_res = torch.cuda.memory_reserved() / (1024**3)
            if state.is_world_process_zero:
                print(f"\n[Memory Debug - Step {state.global_step} 开始] "
                      f"已分配(Allocated): {mem_alloc:.2f} GB, "
                      f"已保留(Reserved): {mem_res:.2f} GB", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024**3)
            mem_res = torch.cuda.memory_reserved() / (1024**3)
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            if state.is_world_process_zero:
                print(f"[Memory Debug - Step {state.global_step} 结束] "
                      f"已分配: {mem_alloc:.2f} GB, "
                      f"已保留: {mem_res:.2f} GB, "
                      f"峰值(Peak): {peak_alloc:.2f} GB\n", flush=True)

class VLM_MemGenRunner(MemGenRunner):
    # Overriding the weaver trainer creation to utilize the processor for vision handling.
    def _create_weaver_trainer(self):
        if self.train_weaver_method == "sft":
            # Pass data collator implicitly by letting SFTTrainer use processor's pad logic
            # The dataset must return dicts with 'pixel_values', 'image_grid_thw' etc.
            weaver_trainer = SFTTrainer(
                model=self.model,
                args=self.weaver_sft_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.weaver_valid_dataset,
                # For Qwen3-VL processor should be passed so SFTTrainer handles it correctly.
                processing_class=self.processing_class, # This is the processor now, not just tokenizer
                callbacks=[MemoryTrackerCallback()],
            )
        elif self.train_weaver_method == 'grpo':
            # Implement VLM GRPO equivalent logic here if needed.
            # (Fallback to base class implementation if pure text handling is sufficient).
            weaver_trainer = super()._create_weaver_trainer()
            weaver_trainer.add_callback(MemoryTrackerCallback())
        return weaver_trainer

    def _filter_dataset(self, dataset):
        # Override filtering logic to support multimodal formats (list of dicts with images)
        def filter_func(sample):
            if "messages" in sample:
                try:
                    seq_len = len(self.processing_class.apply_chat_template(sample["messages"], tokenize=True))
                    # 打印过长被过滤的数据长度，排查是否过滤失效
                    if seq_len >= 1024:
                        print(f"[数据过滤] 发现超长序列，长度: {seq_len}，已过滤。")
                    return seq_len < 1024
                except Exception as e:
                    print(f"[数据过滤异常] 处理 sample 失败: {e}")
                    return False
            return True
        return dataset.filter(filter_func)
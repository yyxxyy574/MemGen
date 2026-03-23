# memgen/runner_vlm.py
from memgen.runner import MemGenRunner
from trl import SFTTrainer

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
            )
        elif self.train_weaver_method == 'grpo':
            # Implement VLM GRPO equivalent logic here if needed.
            # (Fallback to base class implementation if pure text handling is sufficient).
            weaver_trainer = super()._create_weaver_trainer()
        return weaver_trainer

    def _filter_dataset(self, dataset):
        # Override filtering logic to support multimodal formats (list of dicts with images)
        def filter_func(sample):
            if "messages" in sample:
                return len(self.processing_class.apply_chat_template(sample["messages"], tokenize=True)) < 1024
            return True 
        return dataset.filter(filter_func)
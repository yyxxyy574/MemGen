# data/eb_habitat/builder.py
import json
import os
from datasets import Dataset, DatasetDict
from data.base_builder import BaseBuilder
from data.eb_habitat.env import EBHabitatEnv

DATA_DIR = "/mnt/nfs_project_a/xinyi/personal_memory/EmbodiedBench/dataset/EB-Habitat_trajectory_dataset"

class EBHabitatBuilder(BaseBuilder):
    def get_env_cls(self):
        return EBHabitatEnv
    
    def _build_datasets(self) -> DatasetDict:
        
        # 加载 multi-step 和 single-step 数据
        with open(os.path.join(DATA_DIR, "eb-habitat_dataset_multi_step.json"), "r", encoding="utf-8") as f:
            multi_data = json.load(f)
        with open(os.path.join(DATA_DIR, "eb-habitat_dataset_single_step.json"), "r", encoding="utf-8") as f:
            single_data = json.load(f)
            
        # all_data = multi_data + single_data
        all_data = single_data
        
        processed_data = [self._preprocess(example) for example in all_data]
        hf_dataset = Dataset.from_list(processed_data)
        
        # 划分训练集和验证集
        train_ratio = self.config.get("train_ratio", 0.9)
        split = hf_dataset.train_test_split(test_size=1-train_ratio, seed=42)
        train_dataset, valid_dataset = split["train"], split["test"]
        
        dataset_dict = DatasetDict()
        dataset_dict["train"] = train_dataset
        dataset_dict["valid"] = valid_dataset
        dataset_dict["test"] = valid_dataset

        return dataset_dict

    def _build_sft_datasets(self) -> DatasetDict:
        return self._build_datasets()

    def _build_rl_datasets(self) -> DatasetDict:
        return self._build_datasets()
    
    @classmethod
    def _preprocess(cls, example):
        messages = []
        history = []
        instruction = example["instruction"]
        
        for i, step in enumerate(example["trajectory"]):
            img_path = step["input_image_path"]
            
            # 第一轮：直接使用 input 中提供的完整 Prompt
            if i == 0:
                user_text = example["input"]
            # 后续轮次：根据 vlm_planner.py 的逻辑拼接 history 和反馈
            else:
                user_text = f"The human instruction is: {instruction}.\n\n The action history:"
                for step_idx, hist_item in enumerate(history):
                    user_text += f"\nStep {step_idx}, action id {hist_item['id']}, {hist_item['name']}, env feedback: {hist_item['feedback']}"
                
                user_text += f"\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history and environment feedback and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the action id (0 ~ 69) from the available actions to execute."

            # Qwen-VL 标准的 message 内容格式
            user_content = [
                {"type": "image", "image": f"{DATA_DIR}/{img_path}"},
                {"type": "text", "text": user_text}
            ]
            messages.append({"role": "user", "content": user_content})
            
            # 助手输出：将其转化为标准 JSON 字符串
            assistant_dict = {
                "visual_description": step.get("visual_description", ""),
                "reasoning_and_reflection": step.get("reasoning_and_reflection", ""),
                "language_plan": step.get("language_plan", ""),
                "executable_plan": step.get("executable_plan", [])
            }
            assistant_text = json.dumps(assistant_dict, indent=4, ensure_ascii=False)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
            
            # 为下一轮更新交互历史
            if len(step.get("executable_plan", [])) > 0:
                exec_plan = step["executable_plan"][0]
                history.append({
                    "id": exec_plan["action"][0],
                    "name": exec_plan["action"][1],
                    "feedback": exec_plan.get("env_feedback", "")
                })
                
        return {"messages": messages}
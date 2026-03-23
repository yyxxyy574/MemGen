# memgen/model/modeling_vlm_memgen.py
import logging
import os
import random
from typing import Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, DynamicCache
from transformers.modeling_utils import PreTrainedModel

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_utils import MemGenOutputWithPast, MemGenLoraSwitchMixin, MemGenGenerationMixin
from memgen.model.trigger_vlm import VLM_MemGenTrigger
from memgen.model.weaver_vlm import VLM_MemGenWeaver
from memgen.utils import CONVERSATION_TEMPLATE, fix_model_parameters, log_trainable_params

class VLM_MemGenModel(PreTrainedModel, MemGenLoraSwitchMixin, MemGenGenerationMixin):
    config_class = MemGenConfig
    INSTRUCTION_STATE = 0
    CONVERSATION_STATE = 1

    def __init__(self, config: MemGenConfig, base_processor, reasoner_base_model, weaver_base_model, trigger_base_model):   
        super().__init__(config)
        self.config = config

        weaver_model_w_lora, trigger_model_w_lora = self._insert_lora_adapters(
            weaver_base_model, config.weaver_lora_config, trigger_base_model, config.trigger_lora_config,
        )

        self.weaver = VLM_MemGenWeaver(weaver_model_w_lora, config.prompt_latents_len, config.inference_latents_len)
        self.trigger = VLM_MemGenTrigger(trigger_model_w_lora, config.trigger_active)

        self.reasoner = reasoner_base_model
        self.processor = base_processor
        self.tokenizer = base_processor.tokenizer
        
        reasoner_hidden_size = reasoner_base_model.config.text_config.hidden_size
        weaver_hidden_size = weaver_base_model.config.text_config.hidden_size
        self.reasoner_to_weaver = nn.Linear(reasoner_hidden_size, weaver_hidden_size) 
        self.weaver_to_reasoner = nn.Linear(weaver_hidden_size, reasoner_hidden_size) 
        
        self.delimiters = [",", ".", "\n"]  
        self.state = None

        self._postprocess_models()
        logging.info("##### VLM MemGen Initialization #####")
        log_trainable_params(self)

    def _postprocess_models(self):
        fix_model_parameters(self.reasoner)  
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"

    @property
    def device(self): return self.reasoner.device

    def _postprocess_assistant_labels(self, input_ids: torch.Tensor, labels: torch.Tensor, tokenizer) -> torch.Tensor:
        pattern_ids: list[int] = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        batch_size, seq_len = input_ids.shape
        new_labels = labels.clone()
        for b in range(batch_size):
            seq = input_ids[b].tolist()
            for i in range(len(seq) - len(pattern_ids) + 1):
                if seq[i : i + len(pattern_ids)] == pattern_ids:
                    new_labels[b, i : i + len(pattern_ids)] = -100
        return new_labels

    def _get_fused_multimodal_embeddings(self, input_ids, pixel_values=None, image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None, **kwargs):
        inputs_embeds = self.reasoner.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_outputs = self.reasoner.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.reasoner.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
        if pixel_values_videos is not None:
            video_outputs = self.reasoner.model.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.reasoner.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            
        return inputs_embeds

    def _get_base_position_ids(self, input_ids, attention_mask, **kwargs):
        model = self.reasoner.model
        if hasattr(model, "compute_3d_position_ids"):
            try:
                pos = model.compute_3d_position_ids(
                    input_ids=input_ids,
                    inputs_embeds=None,
                    image_grid_thw=kwargs.get("image_grid_thw"),
                    video_grid_thw=kwargs.get("video_grid_thw"),
                    attention_mask=attention_mask,
                    mm_token_type_ids=kwargs.get("mm_token_type_ids")
                )
                if pos is not None: return pos
            except Exception:
                pass
                
        if hasattr(model, "get_rope_index"):
            rope_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "image_grid_thw" in kwargs: rope_kwargs["image_grid_thw"] = kwargs["image_grid_thw"]
            if "video_grid_thw" in kwargs: rope_kwargs["video_grid_thw"] = kwargs["video_grid_thw"]
            if "mm_token_type_ids" in kwargs: rope_kwargs["mm_token_type_ids"] = kwargs["mm_token_type_ids"]
            try:
                pos, _ = model.get_rope_index(**rope_kwargs)
                if pos is not None: return pos
            except Exception:
                pass
        
        pos_ids = attention_mask.long().cumsum(-1) - 1
        pos_ids = pos_ids.masked_fill(attention_mask == 0, 0)
        return pos_ids.unsqueeze(0).repeat(3, 1, 1)

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        B, seq_len = input_ids.shape
        inputs_embeds = self._get_fused_multimodal_embeddings(input_ids, **kwargs)
        position_ids = self._get_base_position_ids(input_ids, attention_mask, **kwargs)
        
        max_augment_num = self.config.max_inference_aug_num
        augmentation_indices = self._select_augment_points_after_delimiter(input_ids, labels, self.delimiters, self.tokenizer, max_augment_num)
        
        current_start_idx = 0
        hidden_size = inputs_embeds.size(-1) 
        
        current_inputs_embeds = torch.empty((B, 0, hidden_size), device=self.device, dtype=inputs_embeds.dtype)
        current_attention_mask = torch.empty((B, 0), device=self.device, dtype=attention_mask.dtype)
        current_latents_mask = torch.empty((B, 0), device=self.device, dtype=torch.bool)
        current_position_ids = torch.empty(*position_ids.shape[:-1], 0, device=self.device, dtype=position_ids.dtype)

        for aug_point_idx in augmentation_indices:
            segment_inputs_embeds = inputs_embeds[:, current_start_idx:aug_point_idx]
            segment_attention_mask = attention_mask[:, current_start_idx:aug_point_idx]
            segment_position_ids = position_ids[..., current_start_idx:aug_point_idx]
            segment_latents_mask = torch.zeros((B, segment_inputs_embeds.size(1)), device=self.device, dtype=torch.bool)

            current_inputs_embeds = torch.cat([current_inputs_embeds, segment_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, segment_attention_mask], dim=1)
            current_position_ids = torch.cat([current_position_ids, segment_position_ids], dim=-1)
            current_latents_mask = torch.cat([current_latents_mask, segment_latents_mask], dim=1)

            weaver_inputs_embeds = self.reasoner_to_weaver(current_inputs_embeds)
            is_prompt_end_aug = (labels[:, aug_point_idx] != -100).all() and (labels[:, aug_point_idx-1] == -100).all().item()
            
            if is_prompt_end_aug:
                weaver_hidden_states, latents_attn_mask, pos_ids = self.weaver.augment_prompt(weaver_inputs_embeds, current_attention_mask, current_position_ids)
            else:
                weaver_hidden_states, latents_attn_mask, pos_ids = self.weaver.augment_inference(weaver_inputs_embeds, current_attention_mask, current_position_ids)

            latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)

            current_inputs_embeds = torch.cat([current_inputs_embeds, latent_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, latents_attn_mask], dim=1)
            current_position_ids = torch.cat([current_position_ids, pos_ids], dim=-1)
            current_start_idx = aug_point_idx
            
            latent_mask = torch.ones((B, latent_inputs_embeds.size(1)), device=self.device, dtype=torch.bool)
            current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)
            
        remaining_inputs_embeds = inputs_embeds[:, current_start_idx:]
        remaining_attention_mask = attention_mask[:, current_start_idx:]
        remaining_position_ids = position_ids[..., current_start_idx:]
        latent_mask = torch.zeros((B, remaining_attention_mask.size(1)), device=self.device, dtype=torch.bool)
        
        current_inputs_embeds = torch.cat([current_inputs_embeds, remaining_inputs_embeds], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, remaining_attention_mask], dim=1)
        current_position_ids = torch.cat([current_position_ids, remaining_position_ids], dim=-1)
        current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)

        reasoner_outputs = self.reasoner(
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids
        )
        logits = reasoner_outputs.logits
        
        shifted = torch.zeros_like(current_latents_mask)
        shifted[:, :-1] = current_latents_mask[:, 1:]
        valid_mask = ~shifted
        valid_logits = logits[valid_mask].view(logits.size(0), -1, logits.size(2))  
        return valid_logits

    def _instructional_forward(self, input_ids, attention_mask, labels, **kwargs):
        logits = self._forward(input_ids, attention_mask, labels, **kwargs)
        return logits, labels

    def _conversational_forward(self, input_ids, attention_mask, labels, **kwargs):
        if kwargs.get('pixel_values') is not None or kwargs.get('pixel_values_videos') is not None:
            return self._instructional_forward(input_ids, attention_mask, labels, **kwargs)
        
        assert input_ids.shape[0] == 1
        seq_len = input_ids.shape[1]
        vocab_size = self.reasoner.config.text_config.vocab_size if hasattr(self.reasoner.config, "text_config") else self.reasoner.config.vocab_size
        device = input_ids.device
        label_row = labels[0]
        should_supervise = label_row != -100

        valid_mask = should_supervise.int()
        diff = torch.diff(torch.cat([torch.tensor([0], device=device), valid_mask]))
        valid_starts = (diff == 1).nonzero(as_tuple=True)[0].tolist() 
        ends = (diff == -1).nonzero(as_tuple=True)[0].tolist()         
        if len(ends) < len(valid_starts): ends.append(seq_len)
        
        triplets = [(start, s, e) for start, (s, e) in zip([0]+ends[:-1], zip(valid_starts, ends))]
        
        if len(triplets) <= self.config.max_prompt_aug_num:
            select_turns = [1] * len(triplets)
        else:
            selected_indices = set(random.sample(range(len(triplets)), self.config.max_prompt_aug_num))
            select_turns = [1 if i in selected_indices else 0 for i in range(len(triplets))]

        all_logits = torch.zeros(1, seq_len, vocab_size, device=device)
        all_labels = torch.full((1, seq_len), -100, device=device)

        for triplet, sup in zip(triplets, select_turns):
            start, valid_start, end = triplet
            if sup:
                cur_input_ids = input_ids[0, :end].unsqueeze(0)
                cur_attention = attention_mask[0, :end].unsqueeze(0)
                cur_labels = torch.full((1, end), -100, device=device)
                cur_labels[0, valid_start:end] = labels[0, valid_start:end]

                logits = self._forward(cur_input_ids, cur_attention, cur_labels, **kwargs)
                all_logits[0, start:end, :] = logits[0, start:end, :]
                all_labels[0, start:end] = labels[0, start:end]

        return all_logits, all_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, **kwargs) -> MemGenOutputWithPast:  
        assert labels is not None
        labels = self._postprocess_assistant_labels(input_ids, labels, self.tokenizer)
       
        if self.state is None:  
            self.state = VLM_MemGenModel.CONVERSATION_STATE if self._is_conversation(input_ids, self.tokenizer) else VLM_MemGenModel.INSTRUCTION_STATE

        forward_func = self._instructional_forward if self.state == VLM_MemGenModel.INSTRUCTION_STATE else self._conversational_forward
        
        logits, supervised_labels = [], []
        batch_size = 1
        iter_num = input_ids.size(0) // batch_size

        for i in range(iter_num):
            b_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            b_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            b_labels = labels[i * batch_size: (i + 1) * batch_size]
            
            b_kwargs = {}
            for k, v in kwargs.items():
                if iter_num == 1:
                    b_kwargs[k] = v
                elif isinstance(v, torch.Tensor) and v.dim() > 0:
                    if v.size(0) == input_ids.size(0):
                        b_kwargs[k] = v[i * batch_size : (i + 1) * batch_size]
                    else:
                        b_kwargs[k] = v
                else:
                    b_kwargs[k] = v

            b_logits, b_sup_labels = forward_func(input_ids=b_ids, attention_mask=b_mask, labels=b_labels, **b_kwargs)
            logits.append(b_logits)
            supervised_labels.append(b_sup_labels)
        
        all_logits = torch.concat(logits, dim=0)
        all_labels = torch.concat(supervised_labels, dim=0)

        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = all_labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs = MemGenOutputWithPast(loss=loss, logits=all_logits)
        outputs.supervised_labels = all_labels
        return outputs

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, generation_config=None, return_augmentation_mask=False, **kwargs): 
        
        B = input_ids.size(0)
        max_new_tokens = generation_config.max_new_tokens
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        prompt_len = input_ids.size(1)

        inputs_embeds = self._get_fused_multimodal_embeddings(input_ids, **kwargs)
        position_ids = self._get_base_position_ids(input_ids, attention_mask, **kwargs)
        
        current_inputs_embeds = inputs_embeds
        current_attention_mask = attention_mask
        current_position_ids = position_ids
        current_input_ids = input_ids
        current_cache = None

        sentence_augment_count = torch.zeros(B, dtype=torch.int, device=self.device)
        augmentation_pos = torch.full((B, max_new_tokens), fill_value=-100, device=self.device) 

        for i in range(max_new_tokens):
            augment_decision = self._should_augment(
                current_input_ids, sentence_augment_count=sentence_augment_count, 
                do_sample=generation_config.trigger_do_sample, temperature=generation_config.temperature, is_prompt=(i==0)  
            )
            augmentation_pos[:, i] = augment_decision
            augment_indices = torch.where(augment_decision == 1)[0]

            if len(augment_indices) > 0:
                if i != 0: sentence_augment_count[augment_indices] += 1
                candidate_inputs_embeds = current_inputs_embeds[augment_indices]
                candidate_attention_mask = current_attention_mask[augment_indices]
                candidate_position_ids = current_position_ids[..., augment_indices, :]
                
                weaver_inputs_embeds = self.reasoner_to_weaver(candidate_inputs_embeds)
                if i == 0:
                    weaver_hidden_states, latents_attn_mask, pos_ids = self.weaver.augment_prompt(weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids)                    
                else:
                    weaver_hidden_states, latents_attn_mask, pos_ids = self.weaver.augment_inference(weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids)
                latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)
                
                candidate_inputs_embeds = torch.cat([candidate_inputs_embeds, latent_inputs_embeds], dim=1)
                candidate_attention_mask = torch.cat([candidate_attention_mask, latents_attn_mask], dim=1)
                candidate_position_ids = torch.cat([candidate_position_ids, pos_ids], dim=-1)
                
                new_len = candidate_inputs_embeds.size(1)
                merged_inputs_embeds = torch.zeros((B, new_len, current_inputs_embeds.size(-1)), device=self.device, dtype=current_inputs_embeds.dtype)
                merged_attention_mask = torch.zeros((B, new_len), device=self.device, dtype=current_attention_mask.dtype)
                merged_position_ids = torch.zeros((*current_position_ids.shape[:-2], B, new_len), device=self.device, dtype=current_position_ids.dtype)
                   
                merged_inputs_embeds[augment_indices] = candidate_inputs_embeds
                merged_attention_mask[augment_indices] = candidate_attention_mask
                merged_position_ids[..., augment_indices, :] = candidate_position_ids
                
                non_augment_indices = torch.where(augment_decision != 1)[0]
                if len(non_augment_indices) > 0:
                    non_aug_inputs_embeds = current_inputs_embeds[non_augment_indices]
                    non_aug_attention_mask = current_attention_mask[non_augment_indices]
                    non_aug_position_ids = current_position_ids[..., non_augment_indices, :]
                    pad_len = self.weaver.prompt_latents_num if i == 0 else self.weaver.inference_latents_num
                    
                    non_aug_inputs_embeds, non_aug_attention_mask, _ = self._left_pad(
                        non_aug_inputs_embeds, non_aug_attention_mask, None, pad_len
                    )
                    pad_pos = torch.zeros((*current_position_ids.shape[:-2], len(non_augment_indices), pad_len), dtype=non_aug_position_ids.dtype, device=self.device)
                    non_aug_position_ids = torch.cat([pad_pos, non_aug_position_ids], dim=-1)
                    
                    merged_inputs_embeds[non_augment_indices] = non_aug_inputs_embeds
                    merged_attention_mask[non_augment_indices] = non_aug_attention_mask
                    merged_position_ids[..., non_augment_indices, :] = non_aug_position_ids
                
                current_inputs_embeds = merged_inputs_embeds
                current_attention_mask = merged_attention_mask
                current_position_ids = merged_position_ids
                current_cache = None  

            if (sentence_augment_count >= self.config.max_inference_aug_num).all():
                generation_config_continue = GenerationConfig(
                    do_sample=generation_config.weaver_do_sample, pad_token_id=pad_token_id, eos_token_id=eos_token_id,
                    use_cache=False, max_new_tokens=max_new_tokens-i
                )
                generated = self.reasoner.generate(
                    inputs_embeds=current_inputs_embeds, attention_mask=current_attention_mask,
                    generation_config=generation_config_continue
                )
                current_input_ids = torch.cat([current_input_ids, generated], dim=1)
                break            

            if current_cache is not None:
                reasoner_inputs_embeds = current_inputs_embeds[:, -1:]
                reasoner_position_ids = current_position_ids[..., -1:]
            else:
                reasoner_inputs_embeds = current_inputs_embeds
                reasoner_position_ids = current_position_ids

            outputs = self.reasoner(
                inputs_embeds=reasoner_inputs_embeds, attention_mask=current_attention_mask,
                position_ids=reasoner_position_ids, use_cache=True, past_key_values=current_cache
            )
            
            next_token_logits = outputs.logits[:, -1]
            next_token_ids = self._get_next_token(next_token_logits, do_sample=generation_config.weaver_do_sample, temperature=generation_config.temperature)
            current_input_ids = torch.cat([current_input_ids, next_token_ids], dim=1)
            
            next_token_embeds = self.reasoner.get_input_embeddings()(next_token_ids)
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embeds], dim=1)
            
            attn_mask = torch.ones((B, 1), dtype=current_attention_mask.dtype, device=self.device)
            current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
            
            next_position_id = current_position_ids.max(dim=-1, keepdim=True)[0] + 1
            current_position_ids = torch.cat([current_position_ids, next_position_id], dim=-1)

            current_cache = outputs.past_key_values
            if (current_input_ids[:, -1] == eos_token_id).all(): break  
            del outputs
        
        new_generated_len = current_input_ids.size(1) - prompt_len
        augmentation_pos = augmentation_pos[:, :new_generated_len]
        
        if return_augmentation_mask: return (current_input_ids, augmentation_pos)
        return current_input_ids

    @classmethod
    def from_config(cls, config_dict: dict):
        model_name = config_dict.get("model_name")
        weaver_model_name = config_dict.get("weaver", {}).get("model_name", None)
        trigger_model_name = config_dict.get("trigger", {}).get("model_name", None)
        
        memgen_config = MemGenConfig(
            max_prompt_aug_num=config_dict.get("max_prompt_aug_num", 1),
            max_inference_aug_num=config_dict.get("max_inference_aug_num", 0),
            prompt_latents_len=config_dict.get("weaver", {}).get("prompt_latents_len", 8),
            inference_latents_len=config_dict.get("weaver", {}).get("inference_latents_len", 8),
            weaver_lora_config=config_dict.get("weaver", {}).get("lora_config", None),
            trigger_active=config_dict.get("trigger", {}).get("active", False),
            trigger_lora_config=config_dict.get("trigger", {}).get("lora_config", None)
        )
        
        try:
            from transformers import Qwen3VLForConditionalGeneration
            VLMClass = Qwen3VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForCausalLM
            VLMClass = AutoModelForCausalLM
            
        processor = AutoProcessor.from_pretrained(model_name)
        reasoner = VLMClass.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        weaver = VLMClass.from_pretrained(weaver_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        trigger = VLMClass.from_pretrained(trigger_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        
        load_path = config_dict.get("load_model_path", None)
        if load_path:
            return cls.from_pretrained(load_path, config=memgen_config, base_processor=processor, reasoner_base_model=reasoner, weaver_base_model=weaver, trigger_base_model=trigger)
        
        return cls(memgen_config, processor, reasoner, weaver, trigger)
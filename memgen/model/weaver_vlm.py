# memgen/model/vlm_weaver.py
from peft import PeftModel
import torch
import torch.nn as nn

class VLM_MemGenWeaver(nn.Module):
    adapter_name = "weaver"

    def __init__(self, model: PeftModel, prompt_latents_len: int, inference_latents_len: int):
        super().__init__()
        self.model = model
        hidden_size = model.base_model.config.text_config.hidden_size

        self.prompt_query_latents = nn.Parameter(torch.randn(prompt_latents_len, hidden_size), requires_grad=True)
        self.inference_query_latents = nn.Parameter(torch.randn(inference_latents_len, hidden_size), requires_grad=True)

        self.prompt_latent_ln = nn.LayerNorm(hidden_size)
        self.inference_latent_ln = nn.LayerNorm(hidden_size)
        self.prompt_latent_scale = nn.Parameter(torch.ones(1))
        self.inference_latent_scale = nn.Parameter(torch.ones(1))

    @property
    def prompt_latents_num(self) -> int: return self.prompt_query_latents.size(0)

    @property
    def inference_latents_num(self) -> int: return self.inference_query_latents.size(0)

    def _augment(
        self, latents: torch.Tensor, latent_ln: nn.LayerNorm, latent_scale: torch.Tensor,
        inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = attention_mask.shape[0]
        latents_num = latents.size(0)

        latents = latent_ln(latents) * latent_scale
        latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)
        
        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)

        latents_mask = torch.ones(latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, latents_mask], dim=1)
        
        # 鲁棒处理任意维度的 position_ids (e.g. [3, B, Seq] or [4, B, Seq] or [B, Seq])
        seq_dim = position_ids.ndim - 1
        last_position_ids = position_ids.max(dim=seq_dim, keepdim=True)[0] 
        latents_relative = torch.arange(1, latents_num + 1, device=position_ids.device)
        
        # 将 relative tensor 扩展为与 position_ids 相同维度
        view_shape = [1] * position_ids.ndim
        view_shape[-1] = -1
        latents_pos = last_position_ids + latents_relative.view(*view_shape)
        
        position_ids = torch.cat([position_ids.long(), latents_pos.long()], dim=seq_dim) 

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,  
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        latents_hidden_states = hidden_states[:, -latents_num:, :]

        return latents_hidden_states, latents_mask, latents_pos

    def augment_prompt(self, inputs_embeds, attention_mask, position_ids):
        return self._augment(self.prompt_query_latents, self.prompt_latent_ln, self.prompt_latent_scale, inputs_embeds, attention_mask, position_ids)

    def augment_inference(self, inputs_embeds, attention_mask, position_ids):
        return self._augment(self.inference_query_latents, self.inference_latent_ln, self.inference_latent_scale, inputs_embeds, attention_mask, position_ids)
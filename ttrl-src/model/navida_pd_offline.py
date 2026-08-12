from dataclasses import dataclass
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
Qwen2_5_VLModel,
Qwen2_5_VLForConditionalGeneration, 
Qwen2_5_VLConfig,
Qwen2_5_VLCausalLMOutputWithPast,
Cache,
BaseModelOutputWithPast,
DynamicCache,
logger,
AttentionMaskConverter,
StaticCache,
SlidingWindowCache
)
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
import torch
from typing import Optional, List
from torch.nn import CrossEntropyLoss
from typing import Union, Tuple
import torch.nn.functional as F
from transformers import AutoProcessor, AutoConfig



@dataclass
class NaVIDACausalOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    alignment_loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None



class NaVIDAPDOfflineQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):

    def initialize_idm_teacher(self, teacher_model):
        self.teacher_model = teacher_model
        self.teacher_model.requires_grad_(False)
    
    @torch.no_grad()
    def get_teacher_hidden_states(
            self,
            input_ids_idm: torch.LongTensor = None,
            attention_mask_idm: Optional[torch.Tensor] = None,
            pixel_values_idm: Optional[torch.Tensor] = None,
            image_grid_thw_idm: Optional[torch.LongTensor] = None,
        ):
        outputs = self.teacher_model(
            input_ids = input_ids_idm,
            attention_mask = attention_mask_idm,
            pixel_values = pixel_values_idm,
            image_grid_thw = image_grid_thw_idm,
            # output_hidden_states = True,
        )
        last_hidden_states = outputs.hidden_states
        return last_hidden_states
        

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            input_ids_idm: torch.LongTensor = None,
            attention_mask_idm: Optional[torch.Tensor] = None,
            pixel_values_idm: Optional[torch.Tensor] = None,
            image_grid_thw_idm: Optional[torch.LongTensor] = None,
            labels_idm: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, NaVIDACausalOutputWithPast]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.visual.dtype)
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                    n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )

                    mask = input_ids == self.config.image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)

                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                    n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                    n_video_features = video_embeds.shape[0]
                    if n_video_tokens != n_video_features:
                        raise ValueError(
                            f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                        )

                    mask = input_ids == self.config.video_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    video_mask = mask_expanded.to(inputs_embeds.device)

                    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

            # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
            if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                # calculate RoPE index once per generation in the pre-fill stage only
                if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                ):
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        attention_mask,
                    )
                    self.rope_deltas = rope_deltas
                # then use the prev pre-calculated rope-deltas to get the correct position ids
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    delta = (
                        (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                        if cache_position is not None
                        else 0
                    )
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:  # otherwise `deltas` is an int `0`
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

            teacher_hidden_states = self.get_teacher_hidden_states(
                input_ids_idm,
                attention_mask_idm,
                pixel_values_idm,
                image_grid_thw_idm
            )

            loss = None
            ce_loss = None
            alignment_loss = None
            if labels is not None:
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                ce_loss = loss_fct(shift_logits, shift_labels)

                shift_vln_mask = torch.where(labels[..., 1:].contiguous()!=-100)
                shift_idm_mask = torch.where(labels_idm[..., 1:].contiguous()!=-100)
                vln_action_hidden_states = hidden_states[shift_vln_mask]
                idm_action_hidden_states = teacher_hidden_states[shift_idm_mask]
                # alignment_loss = F.mse_loss(vln_action_hidden_states, idm_action_hidden_states)
                alignment_loss = 1 - F.cosine_similarity(vln_action_hidden_states, idm_action_hidden_states, dim=-1).mean()

                loss = ce_loss + 1*alignment_loss

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return NaVIDACausalOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
                ce_loss = ce_loss,
                alignment_loss = alignment_loss,
            )


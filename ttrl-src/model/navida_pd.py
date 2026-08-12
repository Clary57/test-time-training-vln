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


# class NaVIDAQwen2_5VLConfig(Qwen2_5_VLConfig):
#     model_type = "navida_qwen2_5_vl"
#     def __init__(self, num_query = 4, dummy_aq_token_id = -1, *args, **kwargs):
#         self.num_query = num_query
#         self.dummy_aq_token_id = dummy_aq_token_id
#         super().__init__(*args, **kwargs)


# class NaVIDAQwen2_5_VLModel(Qwen2_5_VLModel):


#     def _update_causal_mask(
#         self,
#         attention_mask: torch.Tensor,
#         input_tensor: torch.Tensor,
#         cache_position: torch.Tensor,
#         past_key_values: Cache,
#         action_query_start_index: torch.Tensor,
#         output_attentions: bool = False,
#     ):
#         causal_mask = super()._update_causal_mask(attention_mask, input_tensor, cache_position, past_key_values, output_attentions=True)
        
#         return causal_mask
    
#     def _update_mask_for_action_query(
#         self, 
#         causal_mask: torch.Tensor,
#         action_query_start_index: torch.Tensor,
#         ):
#         # training -> [b, 1, seq_len, seq_len]
#         # inference -> [b, 1, 1, seq_len] seq_len increases after generate one token
#         # assert self.config._attn_implementation == "sdpa"
#         bs = causal_mask.size(0)
#         for bs_idx in range(bs):
#             start_index = action_query_start_index[bs_idx]
#             causal_mask[bs_idx, :, start_index+self.config.num_query:, :start_index] = torch.min(causal_mask)
#         return causal_mask

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         action_query_start_index: Optional[torch.LongTensor] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

#         if self.gradient_checkpointing and self.training:
#             if use_cache:
#                 logger.warning_once(
#                     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                 )
#                 use_cache = False

#         # torch.jit.trace() doesn't support cache objects in the output
#         if use_cache and past_key_values is None and not torch.jit.is_tracing():
#             past_key_values = DynamicCache()

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         if cache_position is None:
#             past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#             cache_position = torch.arange(
#                 past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#             )

#         # the hard coded `3` is for temporal, height and width.
#         if position_ids is None:
#             position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
#         elif position_ids.dim() == 2:
#             position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

#         causal_mask = self._update_causal_mask(
#             attention_mask, inputs_embeds, 
#             cache_position, past_key_values, 
#             action_query_start_index, output_attentions
#         )

#         hidden_states = inputs_embeds

#         # create position embeddings to be shared across the decoder layers
#         position_embeddings = self.rotary_emb(hidden_states, position_ids)

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = None

#         for decoder_layer in self.layers:
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     decoder_layer.__call__,
#                     hidden_states,
#                     causal_mask,
#                     position_ids,
#                     past_key_values,
#                     output_attentions,
#                     use_cache,
#                     cache_position,
#                     position_embeddings,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=causal_mask,
#                     position_ids=position_ids,
#                     past_key_value=past_key_values,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                     cache_position=cache_position,
#                     position_embeddings=position_embeddings,
#                 )

#             hidden_states = layer_outputs[0]

#             if use_cache:
#                 next_decoder_cache = layer_outputs[2 if output_attentions else 1]

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#         hidden_states = self.norm(hidden_states)

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         next_cache = next_decoder_cache if use_cache else None

#         if not return_dict:
#             return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )



class NaVIDAPDQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):

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
            # action_query_start_index: Optional[torch.LongTensor] = None,
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

                # vln_logits = shift_logits[0::2]
                # idm_logits = shift_logits[1::2]
                # vln_labels = shift_labels[0::2]
                # idm_labels = shift_labels[1::2]
                # vln_ids = input_ids[..., 1:][0::2]
                # idm_ids = input_ids[..., 1:][1::2]

                # log_probs_vln = vln_logits.log_softmax(dim=-1)
                # token_log_prob_vln = torch.gather(log_probs_vln, dim=1, index=vln_ids.unsqueeze(1)).squeeze(1)
                # token_log_prob_vln = token_log_prob_vln * (vln_labels!=-100)

                # log_probs_idm = idm_logits.log_softmax(dim=-1)
                # token_log_prob_idm = torch.gather(log_probs_idm, dim=1, index=idm_ids.unsqueeze(1)).squeeze(1)
                # token_log_prob_idm = token_log_prob_idm * (idm_labels!=-100)

                # vln_token_logps = token_log_prob_vln
                # idm_token_logps = token_log_prob_idm.detach()
                # alignment_loss = torch.exp(idm_token_logps - vln_token_logps) - (idm_token_logps - vln_token_logps) - 1
                # alignment_loss = alignment_loss.mean()

                shift_logits_ = logits[..., :-1, :].contiguous()
                shift_labels_ = labels[..., 1:].contiguous()
                vln_labels = shift_labels_[0::2]
                idm_labels = shift_labels_[1::2]
                vln_logits = shift_logits_[0::2]
                idm_logits = shift_logits_[1::2]
                alignment_loss = 0
                for cur_vln_logits, cur_idm_logits, cur_vln_labels, cur_idm_labels in zip(vln_logits, idm_logits, vln_labels, idm_labels):
                    vln_logits = cur_vln_logits[torch.where(cur_vln_labels!=-100)].reshape(-1, cur_vln_logits.shape[-1])
                    idm_logits = cur_idm_logits[torch.where(cur_idm_labels!=-100)].reshape(-1, cur_idm_logits.shape[-1])
                    student_log_probs = F.log_softmax(vln_logits, dim=-1)
                    teacher_probs = F.softmax(idm_logits, dim=-1)

                    kl_loss_pointwise = F.kl_div(
                        student_log_probs, 
                        teacher_probs.detach(), 
                        reduction='none',
                        log_target=False
                    )

                    alignment_loss = alignment_loss + kl_loss_pointwise.sum(-1).mean()

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


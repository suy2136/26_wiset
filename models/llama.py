import torch
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union


class LlamaNetworkingHeadModel(LlamaForCausalLM):
    """
    Llama with networking head.
    This class is implemented based on LlamaForCausalLM.
    """
    _tied_weights_keys = ["networking_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.networking_head = None

    def get_networking_head(self):
        return self.networking_head

    def set_networking_head(self, networking_head):
        self.networking_head = networking_head

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_ids_len: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        teacher_forcing: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Reimplement forward method for viewport prediction.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # PEFT 0.6.x can expose fp32 adapter/embedding-facing tensors while
        # the frozen Llama decoder remains fp16.  Anchor inputs_embeds to an
        # unadapted base projection dtype before entering the decoder so q/k/v
        # all receive the dtype expected by their frozen weights.
        if inputs_embeds is not None:
            base_dtype = self.model.layers[0].self_attn.k_proj.weight.dtype
            inputs_embeds = inputs_embeds.to(base_dtype)

        def forward_once():
            # decoder outputs consists of
            # (dec_features, layer_state, dec_hidden, dec_attn)
            current_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = current_outputs[0].float()
            if teacher_forcing:
                current_prediction = self.networking_head.teacher_forcing(
                    hidden_states
                )
            else:
                current_prediction = self.networking_head(hidden_states)
            return current_outputs, current_prediction

        outputs, prediction = forward_once()
        if getattr(self, "_vp_fp16_fallback_enabled", False):
            # Check the tiny task-head output rather than every hidden tensor or
            # every compact projection.  This preserves the normal fast path.
            prediction_finite = bool(
                torch.isfinite(prediction.detach()).all().item()
            )
            if not prediction_finite:
                from models.vp_numeric_safety import vp_safe_retry
                self.vp_fp16_fallback_calls = int(
                    getattr(self, "vp_fp16_fallback_calls", 0)
                ) + 1
                with vp_safe_retry(self.model, "fp16_prescaled"):
                    outputs, prediction = forward_once()
                if not bool(torch.isfinite(prediction.detach()).all().item()):
                    self.vp_fp32_fallback_calls = int(
                        getattr(self, "vp_fp32_fallback_calls", 0)
                    ) + 1
                    with vp_safe_retry(self.model, "fp32"):
                        outputs, prediction = forward_once()
                if not bool(torch.isfinite(prediction.detach()).all().item()):
                    raise FloatingPointError(
                        "non-finite VP prediction after prescaled-Q/K and "
                        "FP32-attention retries"
                    )

        loss = None

        if not return_dict:
            output = (prediction,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            # logits=logits,
            logits=prediction,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

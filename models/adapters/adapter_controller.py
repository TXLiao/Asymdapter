"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import os
import torch.nn as nn
from transformers import apply_chunking_to_forward
import torch.nn.functional as F
import torch
import math

from transformers.models.bert.modeling_bert import  BertAttention, BertOutput, BertEncoder, BertLayer, BaseModelOutputWithPastAndCrossAttentions
from typing import Optional, Tuple
from models.SimPRL import BERT_4_ID_GPS
import torch.nn as nn
from transformers.activations import get_activation


class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


class TopKGate(nn.Module):
    """Gate module to select top k experts."""

    def __init__(self, input_dim, num_experts, k=1):
        super().__init__()
        self.k = k
        # Linear layer to compute logits for experts
        self.gate_linear = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x):
        # x shape: [batch_size * seq_len, input_dim]
        # logits shape: [batch_size * seq_len, num_experts]
        x = self.gate_linear(x)   # logits

        # Select top-k experts
        # top_k_logits shape: [batch_size * seq_len, k]
        # top_k_indices shape: [batch_size * seq_len, k]
        top_k_logits, top_k_indices = torch.topk(
            x, self.k, dim=-1
        )

        # Apply softmax to top-k logits for weights
        # top_k_weights shape: [batch_size * seq_len, k]
        top_k_logits = F.softmax(top_k_logits, dim=-1)  # top_k_weights

        # Create a sparse weight matrix for combining outputs
        # full_weights shape: [batch_size * seq_len, num_experts]
        full_weights = torch.zeros_like(x)
        full_weights.scatter_(1, top_k_indices, top_k_logits)

        return full_weights, top_k_indices  # Return weights and indices

class SMoEAdapter_DOWN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['hidden_size']
        self.down_sample_size = self.input_dim // config['reduction_factor'] # config.reduction_factor： compacter / adapter is 32
        self.activation = Activations("gelu_new") # config.non_linearity.lower() compacter / adapter is "gelu_new"
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)
        # 将专家权重整合为单个参数矩阵 [expert_num, input_dim, down_sample_size]
        self.expert_num = config['expert_num']
        self.down_sampler_weights = nn.Parameter(
            torch.Tensor(self.expert_num, self.down_sample_size, self.input_dim)
        )
        self.down_sampler_bias = nn.Parameter(
            torch.Tensor(self.expert_num, self.down_sample_size)
        )

        self.k = config['num_top_k_expert']
        self.gate = TopKGate(self.input_dim, self.expert_num, self.k)

        # 初始化权重
        nn.init.xavier_uniform_(self.down_sampler_weights)
        nn.init.zeros_(self.down_sampler_bias)

    def calculate_load_balancing_loss(self, gate_weights, alpha=0.1):
        """计算负载均衡损失"""
        return self.expert_num * torch.sum(
            (torch.sum(gate_weights, dim=0) / gate_weights.shape[0]) * torch.mean(gate_weights, dim=0)) * alpha

    def forward(self, x: torch.Tensor, task_id: torch.Tensor=None):
        original_shape = x.shape
        N = original_shape[0] * original_shape[1]
        x = x.view(N, -1)  # [N, input_dim]

        # 获取门控权重和 top-k 专家索引
        gate_weights, top_k_indices = self.gate(x)  # [N, num_experts], [N, k]

        # 提取 top-k 权重 [N, k]
        top_k_weights = torch.gather(gate_weights, dim=1, index=top_k_indices)  # [N, k]
        top_k_weights = top_k_weights.view(-1)  # [N*k]

        # 展平专家索引 [N*k]
        top_k_indices = top_k_indices.view(-1)

        # 构建重复索引 [N*k]
        indices = torch.arange(N, device=x.device).repeat_interleave(self.k)

        # 使用 index_select 替代 x.repeat_interleave
        x_selected = x.index_select(0, indices)

        # 专家输出计算
        expert_output = torch.bmm(
            self.down_sampler_weights[top_k_indices],
            x_selected.unsqueeze(-1)
        ).squeeze(-1) + self.down_sampler_bias[top_k_indices]

        # 加权输出
        weighted_output = expert_output * top_k_weights.unsqueeze(-1)

        # 使用 index_add_ 替代 scatter_add_
        final_output = torch.zeros(N, self.down_sample_size, device=x.device, dtype=x.dtype)
        final_output.index_add_(0, indices, weighted_output)

        # 重塑输出
        final_output = final_output.view(original_shape[0], original_shape[1], self.down_sample_size)

        final_output = self.activation(final_output)
        final_output = self.up_sampler(final_output)

        return final_output, self.calculate_load_balancing_loss(gate_weights)

class AdapterController_MoE(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.identify = config['identify']
        self.low_rank_adapters = config['low_rank_adapters']
        self.hypercomplex_adapters = config['hypercomplex_adapters']
        self.adapter_moe_up_sampler = config['adapter_moe_up_sampler']
        self.adapter_moe_down_sampler = config['adapter_moe_down_sampler']
        self.adapter_moe_up_down_sampler = config['adapter_moe_up_down_sampler']

        self.adapter_moe_gate = config['adapter_moe_gate']
        self.adapter = self.construct_adapters()
        self.add_layer_norm_before_adapter = config['add_layer_norm_before_adapter']
        self.add_layer_norm_after_adapter =config['add_layer_norm_after_adapter']
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.input_dim)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(config.input_dim)

    def construct_adapters(self, ):
        self.adapter = SMoEAdapter_DOWN(self.config)
        return self.adapter

    def forward(self, inputs, task_id = None):

        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        # adapter_moe
        outputs, gate_weights = self.adapter(z, task_id)

        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs
        return outputs, gate_weights


class BertAttention_Adapter_MoE(BertAttention):
    def __init__(self, config, adapter_config):
        super().__init__(config)
        self.adapter_controller = AdapterController_MoE(adapter_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        task_id: torch.Tensor = None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add adapter after MHA and FFN and before Res-connection as paper of Adapter
        attention_output, gate_weights_loss = self.adapter_controller(attention_output, task_id)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs, gate_weights_loss


class BertOutput_Adapter_MoE(BertOutput):
    def __init__(self, config, adapter_config):
        super().__init__(config)
        self.adapter_controller = AdapterController_MoE(adapter_config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, task_id: torch.Tensor = None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states, gate_weights_loss = self.adapter_controller(hidden_states, task_id)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states, gate_weights_loss


class BertLayer_Adapter_MoE(BertLayer):
    def __init__(self, config, adapter_config):
        super().__init__(config)
        self.adapter_config = adapter_config
        self.add_adapter_in_self_attention = adapter_config['add_adapter_in_self_attention']
        self.add_adapter_in_feed_forward = adapter_config['add_adapter_in_feed_forward']
        self.num_div = int(self.add_adapter_in_self_attention) + int(self.add_adapter_in_feed_forward)
        self.num_div = self.num_div if self.num_div > 0 else 1
        if self.add_adapter_in_self_attention:
            self.attention = BertAttention_Adapter_MoE(config, adapter_config)
        if self.add_adapter_in_feed_forward:
            self.output = BertOutput_Adapter_MoE(config, adapter_config)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_Adapter(self, attention_output, task_id: torch.Tensor=None):
        intermediate_output = self.intermediate(attention_output)
        layer_output, gate_weights_loss = self.output(intermediate_output, attention_output, task_id)
        return layer_output, gate_weights_loss


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        task_id: torch.Tensor = None
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        gate_weights_losses = 0
        if self.add_adapter_in_self_attention:
            self_attention_outputs, gate_weights_loss = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=self_attn_past_key_value,
                task_id=task_id
            )
            gate_weights_losses += gate_weights_loss
        else:
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=self_attn_past_key_value
            )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # TODO: if config['add_adapter_in_feed_forward'], add  gate_weights_loss
        if self.add_adapter_in_feed_forward:
            layer_output, gate_weights_loss = apply_chunking_to_forward(
                self.feed_forward_chunk_Adapter, self.chunk_size_feed_forward, self.seq_len_dim, attention_output, task_id
            )
            gate_weights_losses += gate_weights_loss
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs, gate_weights_losses/self.num_div


class BertEncoder_Adapter_MoE(BertEncoder):
    def __init__(self, config, adapter_config):
        super().__init__(config)
        self.num_hidden_layers = config.num_hidden_layers
        self.layer = nn.ModuleList([BertLayer_Adapter_MoE(config, adapter_config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        task_id: torch.Tensor = None
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        gate_weights_losses = 0
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs,gate_weights_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    task_id
                )
            else:
                layer_outputs,gate_weights_loss = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    task_id
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
            gate_weights_losses += gate_weights_loss

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ), gate_weights_losses / self.num_hidden_layers


class BERT_4_ID_GPS_Adapter_MoE(BERT_4_ID_GPS):
    def __init__(self, config, bert_config):
        super().__init__(config, bert_config)
        self.encoder_4_seg_gps = BertEncoder_Adapter_MoE(bert_config, config)

    def forward(self,
                input_gps: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask_seg: Optional[torch.Tensor] = None,
                attention_mask_gps: Optional[torch.Tensor] = None,
                return_dict: Optional[bool] = None,
                task_id: Optional[torch.Tensor] = None
                ):
        sequence_output = []
        gate_weights_losses = 0
        if input_ids is not None:
            input_shape = input_ids.size()

            seg_embedding_output = self.seg_embedding(
                input_ids=input_ids
            )

            extended_attention_mask_seg = self.get_extended_attention_mask(attention_mask_seg, input_shape)

            seg_encoder_outputs, gate_weights_losses = self.encoder_4_seg_gps(
                seg_embedding_output,
                attention_mask=extended_attention_mask_seg,
                return_dict=return_dict,
                task_id=task_id
            )

            seg_sequence_output = seg_encoder_outputs[0]

            sequence_output.append(seg_sequence_output)
        # if input_gps is not None:
        #     input_shape = input_gps.size()[:-1]
        #     extended_attention_mask_gps = self.get_extended_attention_mask(attention_mask_gps, input_shape)
        #     gps_embedding_output = self.gps_embedding(input_gps)
        #     gps_encoder_outputs = self.encoder_4_seg_gps(
        #         gps_embedding_output,
        #         attention_mask=extended_attention_mask_gps,
        #         return_dict=return_dict
        #     )
        #     gps_sequence_output = gps_encoder_outputs[0]
        #     sequence_output.append(gps_sequence_output)

        return sequence_output, gate_weights_losses


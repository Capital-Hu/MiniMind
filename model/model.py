from transformers import PretrainedConfig

# hugging face config for MokioMind
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import math
import torch
from torch import nn
from typing import Optional, Tuple
from torch.nn import functional as F
from transformers.activations import ACT2FN

# 继承nn.Module类
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
    

def precompute_freqs_cis(
        dim:int, 
        end:int=int(32*1024), 
        rope_base:float=1e6, 
        rope_scaling:dict=None,
    ):
    # 写出最初的RoPE式子
    freqs = 1.0 / rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float())

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )
        # 计算curr_dim
        curr_dim = next((i for i in range(dim//2) if 2*math.pi/freqs[i]>orig_max), dim//2)

        # 计算power
        power = torch.arange(0, dim//2, device=freqs.device).float() / max(dim//2-1, 1)

        # 计算beta
        beta = beta_slow + (beta_fast - beta_slow) * power

        # 计算scale
        scale = torch.where(
            torch.arange(dim//2, device=freqs.device) < curr_dim,
            (beta * factor - beta + 1) / (beta * factor),
            1.0/factor
        )

        # 应用scale
        freqs = freqs * scale

    # 生成位置索引 与频率相乘
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  # [end, dim//2]

    # 计算sin和cos
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # [a, b] -> [-b, a]
    def rotate_half(x):
        # x.shape[-1]//2 取最后一个纬度的中点
        return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
    # 应用旋转位置编码
    # x_rotate = x * cos + rotate_half(x) * sin
    # unsqueeze_dim 用于在指定维度上扩展cos和sin的维度，以便与q和k进行广播操作
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x:torch.Tensor, n_rep:int):
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads

        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_key_value_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention

    # 投影 然后view拆分多个头 对于q/k使用rope 对于kv使用repeat_kv 最后attention计算 拼接返回
    def forward(
            self, 
            x:torch.Tensor, 
            positional_embedding:Tuple[torch.Tensor, torch.Tensor], 
            past_key_value:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, 
            use_cache=False, 
            attention_mask:Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 拆分多个头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # 应用RoPE
        cos, sin = positional_embedding
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len, :], sin[:seq_len, :])

        # 重复kv以匹配q的头数
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.transpose(1, 2),  # bsz, n_local_heads, seq_len, head_dim
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # bsz, n_local_heads, seq_len, head_dim
            repeat_kv(xv, self.n_rep).transpose(1, 2),  # bsz, n_local_heads, seq_len, head_dim
        )

        # 进行flash attention计算
        if (
            self.flash 
            and seq_len>1 
            and (attention_mask is None or torch.all(attention_mask==1))
        ):
            attn_mask = (
                None 
                if attention_mask is None 
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            scores = xq@xk.transpose(-2, -1) / math.sqrt(self.head_dim)  # bsz, n_local_heads, seq_len, seq_len
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=scores.device), 
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # bsz, 1, 1, seq_len
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv  # bsz, n_local_heads, seq_len, head_dim

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv
    

class FeedForward(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):
        return self.dropout(
            self.down_proj(
                self.act_fn(self.up_proj(x)) * self.gate_proj(x)
            )
        )
    

class MokioMindBlock(nn.Module):
    def __init__(self, layer_id:int, args:MokioMindConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(args)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(self.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=args.rms_norm_eps)
        self.mlp = FeedForward(args)

    def forward(self, hidden_states, position_embedding, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), 
            position_embedding, 
            past_key_value=past_key_value, 
            use_cache=use_cache, 
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

       

class MokioMindModel(nn.Module):
    def __init__(self, config:MokioMindConfig):
        super().__init__()
        self.config = config

        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size, 
            config.num_hidden_layers
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [MokioMindBlock(i, config) for i in range(self.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        ) 
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
            self, 
            input_ids:Optional[torch.Tensor]=None, 
            attention_mask:Optional[torch.Tensor]=None, 
            past_key_values:Optional[Tuple[Tuple[torch.Tensor]]]=None, 
            use_cache:bool=False,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor]]]]:
        batch_size, seq_len = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embedding = (
            self.freqs_cos[start_pos : start_pos + seq_len], 
            self.freqs_sin[start_pos : start_pos + seq_len]
        )

        presents = []

        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states, 
                position_embedding, 
                past_key_value=past_key_value, 
                use_cache=use_cache, 
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents
    
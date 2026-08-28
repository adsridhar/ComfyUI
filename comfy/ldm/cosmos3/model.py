# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from types import SimpleNamespace

import torch
import torch.nn as nn

import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension
from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
from comfy.ldm.modules.attention import optimized_attention, optimized_attention_masked

from .position_embedding import Cosmos3RotaryEmbedding, text_position_ids, vision_position_ids


class Cosmos3AttnProcessor:
    """Dual-pathway attention processor for Cosmos3.

    Projects, normalizes, applies rotary position embeddings, then runs separate causal (understanding) and full
    (generation) attention pathways. The generation pathway cross-attends to both und and gen keys/values.
    """

    def __call__(
        self,
        attn: "Cosmos3PackedMoTAttention",
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
        transformer_options: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Per-pathway projections
        q_und = attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.add_v_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)

        q_und = attn.norm_q(q_und)
        k_und = attn.norm_k(k_und)
        k_und_for_gen = attn.k_norm_und_for_gen(k_und) if attn.k_norm_und_for_gen is not None else k_und
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)

        # Apply rotary position embeddings per pathway
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        k_und_for_gen = k_und_for_gen * cos_und + _rotate_half(k_und_for_gen) * sin_und
        cos_gen = cos_gen.unsqueeze(1)
        sin_gen = sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen

        # Causal pathway (understanding): und tokens self-attend with causal masking.
        causal_out = optimized_attention_masked(
            q_und.transpose(0, 1).unsqueeze(0),
            k_und.transpose(0, 1).unsqueeze(0),
            v_und.transpose(0, 1).unsqueeze(0),
            heads=attn.num_attention_heads,
            mask=causal_mask,
            skip_reshape=True,
            enable_gqa=True,
            transformer_options=transformer_options,
        )
        causal_out = causal_out.squeeze(0)

        # Full pathway (generation): gen tokens cross-attend to all (und + gen) keys/values.
        all_k = torch.cat([k_und_for_gen, k_gen], dim=0)
        all_v = torch.cat([v_und, v_gen], dim=0)
        full_out = optimized_attention(
            q_gen.transpose(0, 1).unsqueeze(0),
            all_k.transpose(0, 1).unsqueeze(0),
            all_v.transpose(0, 1).unsqueeze(0),
            heads=attn.num_attention_heads,
            skip_reshape=True,
            enable_gqa=True,
            transformer_options=transformer_options,
        )
        full_out = full_out.squeeze(0)

        # Per-pathway output projection
        und_out = attn.to_out(causal_out)
        gen_out = attn.to_add_out(full_out)
        return und_out, gen_out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Cosmos3VLTextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        if hidden_act not in ("relu2", "silu"):
            raise ValueError(f"Cosmos3 only supports `hidden_act` values 'relu2' and 'silu', got {hidden_act!r}.")
        self.hidden_act = hidden_act
        if hidden_act == "silu":
            self.gate_proj = operations.Linear(
                hidden_size, intermediate_size, bias=False, device=device, dtype=dtype
            )
        self.up_proj = operations.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act_fn = nn.SiLU() if hidden_act == "silu" else None

    def forward(self, x):
        if self.hidden_act == "relu2":
            return self.down_proj(torch.relu(self.up_proj(x)).square())
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DomainAwareLinear(nn.Module):
    """Linear projection with one weight/bias pair per embodiment domain."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_domains: int,
        device=None,
        dtype=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains
        self.fc = operations.Embedding(
            self.num_domains, self.output_size * self.input_size, device=device, dtype=dtype
        )
        self.bias = operations.Embedding(self.num_domains, self.output_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        if domain_id.ndim == 0:
            domain_id = domain_id.unsqueeze(0)
        domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
        if x.shape[0] != domain_id.shape[0]:
            raise ValueError(
                "Cosmos3 action domain_id batch size must match action tokens: "
                f"tokens={x.shape[0]}, domain_id={domain_id.shape[0]}."
            )
        if torch.any((domain_id < 0) | (domain_id >= self.num_domains)):
            raise ValueError(f"Cosmos3 action domain_id must be in [0, {self.num_domains}), got {domain_id.tolist()}.")
        weight = self.fc(domain_id).view(domain_id.shape[0], self.input_size, self.output_size)
        bias = self.bias(domain_id).view(domain_id.shape[0], self.output_size)
        if x.ndim == 2:
            return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        if x.ndim == 3:
            return torch.bmm(x, weight) + bias.unsqueeze(1)
        raise ValueError(f"Cosmos3 DomainAwareLinear expected rank-2 or rank-3 input, got {tuple(x.shape)}.")


class Cosmos3PackedMoTAttention(nn.Module):
    """Dual-pathway packed attention with separate projections for the understanding and generation token streams."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        rms_norm_eps: float,
        qk_norm_for_text: bool = True,
        use_und_k_norm_for_gen: bool = False,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # Understanding pathway. norm_q / norm_k are applied per-head (only on
        # head_dim), so no reshape is needed after them.
        self.to_q = operations.Linear(
            hidden_size, num_attention_heads * head_dim, bias=attention_bias, device=device, dtype=dtype
        )
        self.to_k = operations.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias, device=device, dtype=dtype
        )
        self.to_v = operations.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias, device=device, dtype=dtype
        )
        self.to_out = operations.Linear(
            num_attention_heads * head_dim, hidden_size, bias=attention_bias, device=device, dtype=dtype
        )
        if not qk_norm_for_text:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = operations.RMSNorm(
                head_dim, eps=rms_norm_eps, elementwise_affine=True, device=device, dtype=dtype
            )
            self.norm_k = operations.RMSNorm(
                head_dim, eps=rms_norm_eps, elementwise_affine=True, device=device, dtype=dtype
            )

        if use_und_k_norm_for_gen and not qk_norm_for_text:
            self.k_norm_und_for_gen = operations.RMSNorm(
                head_dim, eps=rms_norm_eps, elementwise_affine=True, device=device, dtype=dtype
            )
        else:
            self.k_norm_und_for_gen = None

        # Generation pathway
        self.add_q_proj = operations.Linear(
            hidden_size, num_attention_heads * head_dim, bias=attention_bias, device=device, dtype=dtype
        )
        self.add_k_proj = operations.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias, device=device, dtype=dtype
        )
        self.add_v_proj = operations.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias, device=device, dtype=dtype
        )
        self.to_add_out = operations.Linear(
            num_attention_heads * head_dim, hidden_size, bias=attention_bias, device=device, dtype=dtype
        )
        self.norm_added_q = operations.RMSNorm(
            head_dim, eps=rms_norm_eps, elementwise_affine=True, device=device, dtype=dtype
        )
        self.norm_added_k = operations.RMSNorm(
            head_dim, eps=rms_norm_eps, elementwise_affine=True, device=device, dtype=dtype
        )
        self.processor = Cosmos3AttnProcessor()

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
        transformer_options: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.processor(self, und_seq, gen_seq, rotary_emb, causal_mask, transformer_options)


class Cosmos3VLTextMoTDecoderLayer(nn.Module):
    """Cosmos3 text MoT decoder layer for the Qwen3 and Nemotron dense backbones."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        attention_bias: bool,
        rms_norm_eps: float,
        hidden_act: str = "silu",
        qk_norm_for_text: bool = True,
        use_und_k_norm_for_gen: bool = False,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = Cosmos3PackedMoTAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
            qk_norm_for_text=qk_norm_for_text,
            use_und_k_norm_for_gen=use_und_k_norm_for_gen,
            device=device,
            dtype=dtype,
            operations=operations,
        )

        self.mlp = Cosmos3VLTextMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.mlp_moe_gen = Cosmos3VLTextMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            device=device,
            dtype=dtype,
            operations=operations,
        )

        self.input_layernorm = operations.RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True, device=device, dtype=dtype
        )
        self.input_layernorm_moe_gen = operations.RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True, device=device, dtype=dtype
        )
        self.post_attention_layernorm = operations.RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True, device=device, dtype=dtype
        )
        self.post_attention_layernorm_moe_gen = operations.RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True, device=device, dtype=dtype
        )

    def forward(
        self,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
        transformer_options: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        und_attn_out, gen_attn_out = self.self_attn(
            und_norm, gen_norm, rotary_emb, causal_mask, transformer_options
        )
        residual_und = und_seq + und_attn_out
        residual_gen = gen_seq + gen_attn_out

        mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
        mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))

        return residual_und + mlp_out_und, residual_gen + mlp_out_gen


class Cosmos3Model(nn.Module):
    def __init__(
        self,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        head_dim: int = 128,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        base_fps: int = 24,
        enable_fps_modulation: bool = True,
        latent_channel: int = 48,
        unified_3d_mrope_reset_spatial_ids: bool = True,
        unified_3d_mrope_temporal_modality_margin: int = 15000,
        latent_patch_size: int = 2,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 36,
        num_key_value_heads: int = 8,
        patch_latent_dim: int = 192,
        rms_norm_eps: float = 1e-6,
        rope_scaling: dict | None = None,
        rope_theta: float = 5000000.0,
        action_dim: int | None = None,
        action_gen: bool = False,
        num_embodiment_domains: int = 32,
        sound_dim: int | None = None,
        sound_gen: bool = False,
        sound_latent_fps: float = 25.0,
        timestep_scale: float = 0.001,
        vocab_size: int = 151936,
        hidden_act: str = "silu",
        qk_norm_for_text: bool = True,
        use_und_k_norm_for_gen: bool = False,
        rope_axes_dim: tuple[int, int, int] | list[int] | None = None,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype

        if rope_axes_dim is None:
            rope_axes_dim = (
                rope_scaling.get("mrope_section", [24, 20, 20]) if rope_scaling is not None else [24, 20, 20]
            )
        self.config = SimpleNamespace(
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            base_fps=base_fps,
            enable_fps_modulation=enable_fps_modulation,
            latent_channel=latent_channel,
            unified_3d_mrope_reset_spatial_ids=unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=unified_3d_mrope_temporal_modality_margin,
            latent_patch_size=latent_patch_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            patch_latent_dim=patch_latent_dim,
            rms_norm_eps=rms_norm_eps,
            rope_scaling=rope_scaling,
            rope_theta=rope_theta,
            action_dim=action_dim,
            action_gen=action_gen,
            num_embodiment_domains=num_embodiment_domains,
            sound_dim=sound_dim,
            sound_gen=sound_gen,
            sound_latent_fps=sound_latent_fps,
            timestep_scale=timestep_scale,
            vocab_size=vocab_size,
            hidden_act=hidden_act,
            qk_norm_for_text=qk_norm_for_text,
            use_und_k_norm_for_gen=use_und_k_norm_for_gen,
            rope_axes_dim=tuple(rope_axes_dim),
        )

        # Text-model layers live directly on the transformer (flat layout). The published
        # checkpoint must be re-keyed with the leading `model.` prefix stripped — see
        # scripts/build_flat_layout_repo.py for the rewrite.
        self.embed_tokens = operations.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Cosmos3VLTextMoTDecoderLayer(
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    intermediate_size=intermediate_size,
                    attention_bias=attention_bias,
                    rms_norm_eps=rms_norm_eps,
                    hidden_act=hidden_act,
                    qk_norm_for_text=qk_norm_for_text,
                    use_und_k_norm_for_gen=use_und_k_norm_for_gen,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = operations.RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True, device=device, dtype=dtype
        )
        self.norm_moe_gen = operations.RMSNorm(
            hidden_size, eps=rms_norm_eps, elementwise_affine=True, device=device, dtype=dtype
        )
        self.rotary_emb = Cosmos3RotaryEmbedding(head_dim, rope_theta, rope_axes_dim)

        # Modality projection heads + timestep embedding.
        self.vocab_size = vocab_size
        self.lm_head = operations.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
        self.proj_in = operations.Linear(patch_latent_dim, hidden_size, bias=True, device=device, dtype=dtype)
        self.proj_out = operations.Linear(hidden_size, patch_latent_dim, bias=True, device=device, dtype=dtype)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=hidden_size,
            device=device,
            dtype=torch.float32,
            operations=operations,
        )
        self.action_gen = action_gen
        self.action_dim = action_dim
        self.num_embodiment_domains = num_embodiment_domains
        if action_gen:
            if self.action_dim is None:
                raise ValueError("`action_dim` must be provided when `action_gen=True`.")
            self.action_proj_in = DomainAwareLinear(
                self.action_dim,
                hidden_size,
                self.num_embodiment_domains,
                device=device,
                dtype=dtype,
                operations=operations,
            )
            self.action_proj_out = DomainAwareLinear(
                hidden_size,
                self.action_dim,
                self.num_embodiment_domains,
                device=device,
                dtype=dtype,
                operations=operations,
            )
            self.action_modality_embed = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        if sound_gen:
            if sound_dim is None:
                raise ValueError("`sound_dim` must be provided when `sound_gen=True`.")
            self.audio_proj_in = operations.Linear(sound_dim, hidden_size, bias=True, device=device, dtype=dtype)
            self.audio_proj_out = operations.Linear(hidden_size, sound_dim, bias=True, device=device, dtype=dtype)
            self.audio_modality_embed = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))

    # -------------------------------------------------------------------------
    # Pure-tensor packing/unpacking helpers (no layer state).
    # -------------------------------------------------------------------------

    def _apply_timestep_embeds_to_noisy_tokens(
        self,
        packed_tokens: torch.Tensor,
        packed_timestep_embeds: torch.Tensor,
        noisy_frame_indexes: list[torch.Tensor],
        token_shapes: list[tuple[int, ...]],
    ) -> torch.Tensor:
        start_noisy_index = 0
        flattened_noisy_frame_indexes: list[torch.Tensor] = []
        for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
            spatial_numel_i = math.prod(token_shape_i[1:])
            spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)
            # Broadcast [N, 1] + [spatial_numel_i] → [N, spatial_numel_i]
            frame_offsets = (noisy_indexes_i * spatial_numel_i).unsqueeze(-1) + spatial_indexes_i + start_noisy_index
            flattened_noisy_frame_indexes.append(frame_offsets.flatten())
            start_noisy_index += token_shape_i[0] * spatial_numel_i
        flattened = torch.cat(flattened_noisy_frame_indexes, dim=0).unsqueeze(-1).expand(-1, packed_tokens.shape[1])
        return packed_tokens.scatter_add(dim=0, index=flattened, src=packed_timestep_embeds)

    def _patchify_and_pack_latents(
        self,
        tokens_vision: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        packed_latent: list[torch.Tensor] = []
        original_latent_shapes: list[tuple[int, int, int]] = []
        for latent in tokens_vision:
            latent = latent.squeeze(0)  # [C, T, H, W]
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (latent_channel, t_actual, h_padded, w_padded),
                    device=latent.device,
                    dtype=latent.dtype,
                )
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded
            h_patches = h_padded // p
            w_patches = w_padded // p
            latent = latent.reshape(latent_channel, t_actual, h_patches, p, w_patches, p)
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, p * p * latent_channel)
            packed_latent.append(latent)
        return torch.cat(packed_latent, dim=0), original_latent_shapes

    def _unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: list[tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: list[tuple[int, int, int]],
    ) -> list[torch.Tensor]:
        p = self.config.latent_patch_size
        latent_channel = self.config.latent_channel
        unpatchified_latents: list[torch.Tensor] = []
        start_idx = 0
        for token_shape, noisy_frame_indexes, original_shape in zip(
            token_shapes_vision, noisy_frame_indexes_vision, original_latent_shapes
        ):
            t_c = token_shape[0]
            _, h_orig, w_orig = original_shape
            h_padded = ((h_orig + p - 1) // p) * p
            w_padded = ((w_orig + p - 1) // p) * p
            h_patches = h_padded // p
            w_patches = w_padded // p
            t_n = len(noisy_frame_indexes)
            output_tensor = torch.zeros(
                (latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                latent_patches = packed_mse_preds[start_idx:end_idx]
                latent_patches = latent_patches.reshape(t_n, h_patches, w_patches, p, p, latent_channel)
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)
                latent = latent.reshape(latent_channel, t_n, h_patches * p, w_patches * p)
                latent = latent[:, :, :h_orig, :w_orig]
                output_tensor[:, noisy_frame_indexes] = latent
                start_idx = end_idx
            unpatchified_latents.append(output_tensor.unsqueeze(0))
        return unpatchified_latents

    def _pack_sound_latents(
        self,
        tokens_sound: list[torch.Tensor],
        token_shapes_sound: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """List of ``[C, T]`` tensors → packed ``[total_T, C]`` tensor."""
        return torch.cat(
            [sound[:, : shape[0]].permute(1, 0) for sound, shape in zip(tokens_sound, token_shapes_sound)],
            dim=0,
        )

    def _unpack_sound_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_sound: list[tuple[int, int, int]],
        noisy_frame_indexes_sound: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Packed ``[total_noisy_T, C]`` predictions → list of ``[C, T]`` tensors (zeros at conditioned positions)."""
        sound_dim = self.config.sound_dim
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_idxs in zip(token_shapes_sound, noisy_frame_indexes_sound):
            T = shape[0]
            output = torch.zeros((sound_dim, T), device=packed_preds.device, dtype=packed_preds.dtype)
            t_n = len(noisy_idxs)
            if t_n > 0:
                output[:, noisy_idxs] = packed_preds[start_idx : start_idx + t_n].T
                start_idx += t_n
            unpacked.append(output)
        return unpacked

    def _pack_action_latents(
        self,
        tokens_action: list[torch.Tensor],
        token_shapes_action: list[tuple[int, int, int]],
        domain_ids_action: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """List of ``[T, D]`` tensors → packed ``[total_T, D]`` plus per-token domain ids."""
        packed: list[torch.Tensor] = []
        domain_ids: list[torch.Tensor] = []
        for action, shape, domain_id in zip(tokens_action, token_shapes_action, domain_ids_action):
            token_count = shape[0]
            packed.append(action[:token_count])
            domain_ids.append(domain_id.reshape(1).expand(token_count))
        return torch.cat(packed, dim=0), torch.cat(domain_ids, dim=0)

    def _unpack_action_latents(
        self,
        packed_preds: torch.Tensor,
        token_shapes_action: list[tuple[int, int, int]],
        noisy_frame_indexes_action: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Packed ``[total_noisy_T, D]`` predictions → list of ``[T, D]`` tensors."""
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_idxs in zip(token_shapes_action, noisy_frame_indexes_action):
            T = shape[0]
            output = torch.zeros((T, self.action_dim), device=packed_preds.device, dtype=packed_preds.dtype)
            t_n = len(noisy_idxs)
            if t_n > 0:
                output[noisy_idxs] = packed_preds[start_idx : start_idx + t_n]
                start_idx += t_n
            unpacked.append(output)
        return unpacked

    # -------------------------------------------------------------------------
    # forward: full per-step pass — encode text/vision/sound/action → run layers →
    # decode vision/sound/action. Pipeline calls this once per CFG pass.
    # -------------------------------------------------------------------------

    def forward_orig(
        self,
        input_ids: torch.Tensor,
        text_indexes: torch.Tensor,
        position_ids: torch.Tensor,
        und_len: int,
        sequence_length: int,
        vision_tokens: list[torch.Tensor],
        vision_token_shapes: list[tuple[int, int, int]],
        vision_sequence_indexes: torch.Tensor,
        vision_mse_loss_indexes: torch.Tensor,
        vision_timesteps: torch.Tensor,
        vision_noisy_frame_indexes: list[torch.Tensor],
        sound_tokens: list[torch.Tensor] | None = None,
        sound_token_shapes: list[tuple[int, int, int]] | None = None,
        sound_sequence_indexes: torch.Tensor | None = None,
        sound_mse_loss_indexes: torch.Tensor | None = None,
        sound_timesteps: torch.Tensor | None = None,
        sound_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_tokens: list[torch.Tensor] | None = None,
        action_token_shapes: list[tuple[int, int, int]] | None = None,
        action_sequence_indexes: torch.Tensor | None = None,
        action_mse_loss_indexes: torch.Tensor | None = None,
        action_timesteps: torch.Tensor | None = None,
        action_noisy_frame_indexes: list[torch.Tensor] | None = None,
        action_domain_ids: list[torch.Tensor] | None = None,
        transformer_options: dict | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Run a full denoising-step forward pass.

        Args:
            input_ids: Text token IDs placed at ``text_indexes`` in the joint sequence.
            text_indexes: Indices of text tokens in the joint sequence.
            position_ids: ``[3, sequence_length]`` mRoPE position IDs for the full joint sequence.
            und_len: Length of the causal text (understanding) prefix; generation tokens follow.
            sequence_length: Total length of the joint packed sequence.
            vision_tokens: Per-item vision latent tensors before patchify.
            vision_token_shapes: Patch grid shapes ``(T, H, W)`` per vision item.
            vision_sequence_indexes: Indices of vision tokens in the joint sequence.
            vision_mse_loss_indexes: Indices used to read vision predictions after the backbone.
            vision_timesteps: Per-patch diffusion timesteps for vision tokens.
            vision_noisy_frame_indexes: Noisy frame indices per vision item.
            sound_tokens: Optional sound latent tensors before packing.
            sound_token_shapes: Optional patch grid shapes for sound items.
            sound_sequence_indexes: Optional indices of sound tokens in the joint sequence.
            sound_mse_loss_indexes: Optional indices used to read sound predictions.
            sound_timesteps: Optional per-token diffusion timesteps for sound.
            sound_noisy_frame_indexes: Optional noisy frame indices per sound item.
            action_tokens: Optional action latent tensors before packing.
            action_token_shapes: Optional patch grid shapes ``(T, H, W)`` per action item.
            action_sequence_indexes: Optional indices of action tokens in the joint sequence.
            action_mse_loss_indexes: Optional indices used to read action predictions after the backbone.
            action_timesteps: Optional per-token diffusion timesteps for action tokens.
            action_noisy_frame_indexes: Optional noisy frame indices per action item.
            action_domain_ids: Optional per-item domain IDs selecting the action head weights.
        Returns:
            A tuple of per-modality prediction lists. Optional modalities return ``None`` when their inputs are omitted.
        """
        if transformer_options is None:
            transformer_options = {}
        has_sound = sound_tokens is not None and sound_sequence_indexes is not None
        has_action = action_tokens is not None and action_sequence_indexes is not None

        # Embed text tokens into the joint hidden_states buffer at their sequence positions.
        packed_text_embedding = self.embed_tokens(input_ids)
        target_dtype = packed_text_embedding.dtype
        hidden_states = packed_text_embedding.new_zeros(size=(sequence_length, self.config.hidden_size))
        hidden_states[text_indexes] = packed_text_embedding

        # Patchify + project vision latents, then add timestep embeddings to noisy frames.
        packed_tokens_vision, original_latent_shapes = self._patchify_and_pack_latents(vision_tokens)
        packed_tokens_vision = self.proj_in(packed_tokens_vision)
        timesteps_vision = vision_timesteps * self.config.timestep_scale
        packed_timestep_embeds_vision = self.time_embedder(self.time_proj(timesteps_vision))
        packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(target_dtype)
        packed_tokens_vision = self._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed_tokens_vision,
            packed_timestep_embeds=packed_timestep_embeds_vision,
            noisy_frame_indexes=vision_noisy_frame_indexes,
            token_shapes=vision_token_shapes,
        )
        hidden_states[vision_sequence_indexes] = packed_tokens_vision

        # Pack + project sound latents (when present); all sound frames are noisy.
        if has_sound:
            packed_tokens_sound = self._pack_sound_latents(sound_tokens, sound_token_shapes).to(target_dtype)
            packed_tokens_sound = self.audio_proj_in(packed_tokens_sound)
            packed_tokens_sound = packed_tokens_sound + comfy.model_management.cast_to(
                self.audio_modality_embed, dtype=packed_tokens_sound.dtype, device=packed_tokens_sound.device
            )
            timesteps_sound = sound_timesteps * self.config.timestep_scale
            packed_timestep_embeds_sound = self.time_embedder(self.time_proj(timesteps_sound))
            packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(target_dtype)
            packed_tokens_sound = self._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound_noisy_frame_indexes,
                token_shapes=sound_token_shapes,
            )
            hidden_states[sound_sequence_indexes] = packed_tokens_sound

        # Pack + project action latents (when present). Domain ids select the action head weights.
        if has_action:
            packed_tokens_action, per_token_domain_ids = self._pack_action_latents(
                action_tokens, action_token_shapes, action_domain_ids
            )
            packed_tokens_action = packed_tokens_action.to(target_dtype)
            per_token_domain_ids = per_token_domain_ids.to(device=packed_tokens_action.device)
            packed_tokens_action = self.action_proj_in(packed_tokens_action, per_token_domain_ids)
            packed_tokens_action = packed_tokens_action + comfy.model_management.cast_to(
                self.action_modality_embed, dtype=packed_tokens_action.dtype, device=packed_tokens_action.device
            )
            if action_mse_loss_indexes.numel() > 0:
                timesteps_action = action_timesteps * self.config.timestep_scale
                packed_timestep_embeds_action = self.time_embedder(self.time_proj(timesteps_action))
                packed_timestep_embeds_action = packed_timestep_embeds_action.to(target_dtype)
                packed_tokens_action = self._apply_timestep_embeds_to_noisy_tokens(
                    packed_tokens=packed_tokens_action,
                    packed_timestep_embeds=packed_timestep_embeds_action,
                    noisy_frame_indexes=action_noisy_frame_indexes,
                    token_shapes=action_token_shapes,
                )
            hidden_states[action_sequence_indexes] = packed_tokens_action

        # Compute rotary embeddings once for the joint sequence, then slice into und/gen halves.
        cos, sin = self.rotary_emb(position_ids.unsqueeze(0), hidden_states.dtype)
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)

        und_seq = hidden_states[:und_len]
        gen_seq = hidden_states[und_len:]
        rotary_emb = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])
        causal_mask = torch.full(
            (und_len, und_len),
            torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).triu_(1)

        for decoder_layer in self.layers:
            und_seq, gen_seq = decoder_layer(
                und_seq, gen_seq, rotary_emb, causal_mask, transformer_options
            )
        und_out = self.norm(und_seq)
        gen_out = self.norm_moe_gen(gen_seq)

        last_hidden_state = torch.cat([und_out, gen_out], dim=0)

        # Decode vision predictions from the joint hidden state.
        preds_vision_packed = self.proj_out(last_hidden_state[vision_mse_loss_indexes])
        preds_vision = self._unpatchify_and_unpack_latents(
            preds_vision_packed,
            token_shapes_vision=vision_token_shapes,
            noisy_frame_indexes_vision=vision_noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
        )

        preds_sound: list[torch.Tensor] | None = None
        if has_sound:
            preds_sound_packed = self.audio_proj_out(last_hidden_state[sound_mse_loss_indexes])
            preds_sound = self._unpack_sound_latents(preds_sound_packed, sound_token_shapes, sound_noisy_frame_indexes)

        preds_action: list[torch.Tensor] | None = None
        if has_action:
            per_noisy_domain_ids = [
                domain_id.reshape(1).expand(len(noisy_idxs))
                for domain_id, noisy_idxs in zip(action_domain_ids, action_noisy_frame_indexes)
            ]
            per_noisy_domain_ids = torch.cat(per_noisy_domain_ids, dim=0).to(device=last_hidden_state.device)
            preds_action_packed = self.action_proj_out(
                last_hidden_state[action_mse_loss_indexes], per_noisy_domain_ids
            )
            preds_action = self._unpack_action_latents(
                preds_action_packed, action_token_shapes, action_noisy_frame_indexes
            )

        return preds_vision, preds_sound, preds_action

    def forward(
        self,
        x,
        timestep,
        context,
        attention_mask=None,
        text_input_ids=None,
        fps=24.0,
        condition_mask=None,
        transformer_options={},
        **kwargs,
    ):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options
            ),
        ).execute(x, timestep, context, attention_mask, text_input_ids, fps, condition_mask, transformer_options, **kwargs)

    def _forward(
        self,
        x,
        timestep,
        context,
        attention_mask=None,
        text_input_ids=None,
        fps=24.0,
        condition_mask=None,
        transformer_options={},
        **kwargs,
    ):
        del kwargs
        # context is a bf16 tensor which we replace with text_input_ids if provided, since text_input_ids is the actual input for the model
        if text_input_ids is not None:
            context = text_input_ids
        original_shape = x.shape
        patch_size = self.config.latent_patch_size
        x = comfy.ldm.common_dit.pad_to_patch_size(
            x, (1, patch_size, patch_size), padding_mode="constant"
        )

        if context.ndim == 1:
            context = context.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(context, dtype=torch.bool)
        elif attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        if condition_mask is None:
            condition_mask = torch.zeros((x.shape[0], x.shape[2]), device=x.device, dtype=torch.bool)
        elif condition_mask.ndim == 1:
            condition_mask = condition_mask.unsqueeze(0)

        timestep = timestep.reshape(-1)
        grid_t = x.shape[2]
        grid_h = x.shape[3] // patch_size
        grid_w = x.shape[4] // patch_size
        frame_token_stride = grid_h * grid_w
        outputs = []

        for batch_index in range(x.shape[0]):
            context_index = batch_index if context.shape[0] > 1 else 0
            input_ids = context[context_index][attention_mask[context_index].bool()]
            text_length = input_ids.shape[0]
            vision_length = grid_t * frame_token_stride
            noisy_frame_indexes = torch.nonzero(
                ~condition_mask[batch_index if condition_mask.shape[0] > 1 else 0].bool(),
                as_tuple=False,
            ).flatten()

            text_positions = text_position_ids(
                1,
                text_length,
                x.device,
                use_float_positions=self.config.enable_fps_modulation,
            )[0]
            temporal_offset = torch.tensor(
                [text_length + self.config.unified_3d_mrope_temporal_modality_margin],
                device=x.device,
                dtype=torch.float32 if self.config.enable_fps_modulation else torch.long,
            )
            fps_index = batch_index if torch.is_tensor(fps) and fps.numel() > 1 else 0
            sample_fps = fps.reshape(-1)[fps_index] if torch.is_tensor(fps) else fps
            vision_positions = vision_position_ids(
                1,
                grid_t,
                grid_h,
                grid_w,
                temporal_offset,
                x.device,
                fps=sample_fps if self.config.enable_fps_modulation else None,
                base_fps=float(self.config.base_fps),
                reset_spatial_ids=self.config.unified_3d_mrope_reset_spatial_ids,
            )[0]
            position_ids = torch.cat((text_positions, vision_positions), dim=1)

            vision_sequence_indexes = torch.arange(
                text_length, text_length + vision_length, device=x.device
            )
            spatial_indexes = torch.arange(frame_token_stride, device=x.device)
            vision_mse_loss_indexes = (
                text_length + noisy_frame_indexes.unsqueeze(1) * frame_token_stride + spatial_indexes
            ).flatten()
            timestep_index = batch_index if timestep.numel() > 1 else 0
            vision_timesteps = timestep[timestep_index].expand(vision_mse_loss_indexes.shape[0])

            preds_vision, _, _ = self.forward_orig(
                input_ids=input_ids,
                text_indexes=torch.arange(text_length, device=x.device),
                position_ids=position_ids,
                und_len=text_length,
                sequence_length=text_length + vision_length,
                vision_tokens=[x[batch_index : batch_index + 1]],
                vision_token_shapes=[(grid_t, grid_h, grid_w)],
                vision_sequence_indexes=vision_sequence_indexes,
                vision_mse_loss_indexes=vision_mse_loss_indexes,
                vision_timesteps=vision_timesteps,
                vision_noisy_frame_indexes=[noisy_frame_indexes],
                transformer_options=transformer_options,
            )
            outputs.append(preds_vision[0])

        return torch.cat(outputs, dim=0)[
            :, :, : original_shape[2], : original_shape[3], : original_shape[4]
        ]

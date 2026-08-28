import torch
from torch import nn


def text_position_ids(batch_size, num_tokens, device, use_float_positions=True):
    dtype = torch.float32 if use_float_positions else torch.long
    positions = torch.arange(num_tokens, device=device, dtype=dtype)
    return positions.view(1, 1, num_tokens).expand(batch_size, 3, -1)


def vision_position_ids(
    batch_size,
    grid_t,
    grid_h,
    grid_w,
    temporal_offset,
    device,
    fps=None,
    base_fps=24.0,
    reset_spatial_ids=True,
):
    if fps is not None and grid_t > 1:
        fps = torch.as_tensor(fps, device=device, dtype=torch.float32)
        if fps.ndim == 0:
            fps = fps.expand(batch_size)
        frame_ids = torch.arange(grid_t, device=device, dtype=torch.float32)
        temporal_ids = frame_ids.view(1, -1) * base_fps / fps.view(-1, 1)
        temporal_ids = temporal_ids + temporal_offset.view(-1, 1)
    else:
        temporal_ids = torch.arange(grid_t, device=device, dtype=torch.long)
        temporal_ids = temporal_ids.view(1, -1).expand(batch_size, -1)
        temporal_ids = temporal_ids + temporal_offset.to(dtype=torch.long).view(-1, 1)

    temporal_ids = temporal_ids.view(batch_size, grid_t, 1).expand(-1, -1, grid_h * grid_w).flatten(1)
    height_ids = torch.arange(grid_h, device=device).view(1, 1, grid_h, 1).expand(batch_size, grid_t, -1, grid_w)
    width_ids = torch.arange(grid_w, device=device).view(1, 1, 1, grid_w).expand(batch_size, grid_t, grid_h, -1)
    height_ids = height_ids.flatten(1)
    width_ids = width_ids.flatten(1)

    if not reset_spatial_ids:
        spatial_offset = temporal_offset.to(dtype=torch.long).view(-1, 1)
        height_ids = height_ids + spatial_offset
        width_ids = width_ids + spatial_offset

    if temporal_ids.dtype.is_floating_point:
        height_ids = height_ids.to(temporal_ids.dtype)
        width_ids = width_ids.to(temporal_ids.dtype)
    return torch.stack((temporal_ids, height_ids, width_ids), dim=1)


class Cosmos3RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rope_theta, rope_axes_dim):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.rope_axes_dim = tuple(rope_axes_dim)

    def forward(self, position_ids, dtype):
        inv_freq = self.inv_freq.to(device=position_ids.device)
        with torch.autocast(device_type=position_ids.device.type, enabled=False):
            freqs = position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, 1, -1)

        freqs_t = freqs[:, 0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            axis_length = self.rope_axes_dim[axis] * 3
            freqs_t[..., offset:axis_length:3] = freqs[:, axis, ..., offset:axis_length:3]

        freqs = torch.cat((freqs_t, freqs_t), dim=-1)
        return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)

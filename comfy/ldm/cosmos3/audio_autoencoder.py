import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

import comfy.ops


def wn_conv1d(operations, *args, **kwargs):
    return weight_norm(operations.Conv1d(*args, **kwargs))


def wn_conv_transpose1d(operations, *args, **kwargs):
    return weight_norm(operations.ConvTranspose1d(*args, **kwargs))


class Snake1d(nn.Module):
    def __init__(self, channels, device=None, dtype=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty((1, channels, 1), device=device, dtype=dtype))
        self.beta = nn.Parameter(torch.empty((1, channels, 1), device=device, dtype=dtype))

    def forward(self, x):
        alpha = comfy.ops.cast_to_input(self.alpha, x).exp()
        beta = comfy.ops.cast_to_input(self.beta, x).exp()
        return x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).square()


class Cosmos3AudioConvNeXtBlock(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, use_snake=True, device=None, dtype=None, operations=None):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.ConstantPad1d((3, 3), 0),
            operations.Conv1d(hidden_dim, hidden_dim, kernel_size=7, groups=hidden_dim, device=device, dtype=dtype),
        )
        self.norm = operations.LayerNorm(hidden_dim, eps=1e-5, bias=False, device=device, dtype=dtype)
        self.pwconv1 = operations.Conv1d(hidden_dim, intermediate_dim, kernel_size=1, device=device, dtype=dtype)
        self.act = Snake1d(intermediate_dim, device=device, dtype=dtype) if use_snake else nn.GELU()
        self.pwconv2 = operations.Conv1d(intermediate_dim, hidden_dim, kernel_size=1, device=device, dtype=dtype)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.pwconv1(x)
        x = self.act(x)
        return residual + self.pwconv2(x)


class Cosmos3AudioEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        stereo,
        channels,
        latent_dim,
        channel_multiples,
        strides,
        num_blocks,
        n_fft,
        hop_length,
        use_snake=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        if len(channel_multiples) != len(strides):
            raise ValueError("Cosmos3 audio encoder channel multipliers and strides must have the same length")
        self.input_channels = input_channels * (2 if stereo else 1)
        self.n_fft = n_fft
        self.hop_length = hop_length

        layers = [
            wn_conv1d(
                operations,
                (n_fft + 2) * self.input_channels,
                channel_multiples[0] * channels,
                kernel_size=1,
                bias=False,
                device=device,
                dtype=dtype,
            )
        ]
        for index, stride in enumerate(strides):
            input_dim = channel_multiples[index] * channels
            output_dim = channel_multiples[min(index + 1, len(channel_multiples) - 1)] * channels
            for _ in range(num_blocks):
                layers.append(
                    Cosmos3AudioConvNeXtBlock(
                        input_dim,
                        input_dim * 4,
                        use_snake=use_snake,
                        device=device,
                        dtype=dtype,
                        operations=operations,
                    )
                )
            layers.append(
                wn_conv1d(
                    operations,
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    padding_mode=padding_mode,
                    device=device,
                    dtype=dtype,
                )
            )
        layers.append(
            wn_conv1d(
                operations,
                channel_multiples[-1] * channels,
                latent_dim,
                kernel_size=1,
                bias=False,
                device=device,
                dtype=dtype,
            )
        )
        self.layers = nn.Sequential(*layers)

    def spectrogram(self, waveform):
        pad_left = (self.n_fft - self.hop_length) // 2
        pad_right = self.n_fft - self.hop_length - pad_left
        waveform = F.pad(waveform, (pad_left, pad_right)).float()
        window = torch.hann_window(self.n_fft, device=waveform.device, dtype=waveform.dtype)
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

    def forward(self, audio):
        batch_size, num_channels, num_samples = audio.shape
        if num_channels != self.input_channels:
            raise ValueError(f"Cosmos3 audio encoder expected {self.input_channels} channels, got {num_channels}")
        if num_channels > 1:
            audio = audio.reshape(batch_size * num_channels, 1, num_samples)

        spectrogram = torch.view_as_real(self.spectrogram(audio.squeeze(1)))
        real, imaginary = spectrogram.chunk(2, dim=-1)
        spectrogram = torch.cat((real, imaginary), dim=1).squeeze(-1).to(audio.dtype)
        if num_channels > 1:
            spectrogram = spectrogram.reshape(batch_size, num_channels * spectrogram.shape[1], spectrogram.shape[2])
        return self.layers(spectrogram).transpose(1, 2)


class Cosmos3AudioResidualUnit(nn.Module):
    def __init__(self, channels, dilation, device=None, dtype=None, operations=None):
        super().__init__()
        padding = 3 * dilation
        self.snake1 = Snake1d(channels, device=device, dtype=dtype)
        self.conv1 = wn_conv1d(
            operations,
            channels,
            channels,
            kernel_size=7,
            dilation=dilation,
            padding=padding,
            device=device,
            dtype=dtype,
        )
        self.snake2 = Snake1d(channels, device=device, dtype=dtype)
        self.conv2 = wn_conv1d(operations, channels, channels, kernel_size=1, device=device, dtype=dtype)

    def forward(self, x):
        output = self.conv1(self.snake1(x))
        output = self.conv2(self.snake2(output))
        padding = (x.shape[-1] - output.shape[-1]) // 2
        if padding > 0:
            x = x[..., padding:-padding]
        return x + output


class Cosmos3AudioDecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride, device=None, dtype=None, operations=None):
        super().__init__()
        self.snake1 = Snake1d(input_dim, device=device, dtype=dtype)
        self.conv_t1 = wn_conv_transpose1d(
            operations,
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=stride % 2,
            device=device,
            dtype=dtype,
        )
        self.res_unit1 = Cosmos3AudioResidualUnit(output_dim, 1, device, dtype, operations)
        self.res_unit2 = Cosmos3AudioResidualUnit(output_dim, 3, device, dtype, operations)
        self.res_unit3 = Cosmos3AudioResidualUnit(output_dim, 9, device, dtype, operations)

    def forward(self, x):
        x = self.conv_t1(self.snake1(x))
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        return self.res_unit3(x)


class Cosmos3AudioDecoder(nn.Module):
    def __init__(
        self,
        channels,
        input_channels,
        audio_channels,
        upsampling_ratios,
        channel_multiples,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        channel_multiples = [1] + list(channel_multiples)
        self.conv1 = wn_conv1d(
            operations,
            input_channels,
            channels * channel_multiples[-1],
            kernel_size=7,
            padding=3,
            device=device,
            dtype=dtype,
        )
        self.block = nn.ModuleList([
            Cosmos3AudioDecoderBlock(
                channels * channel_multiples[len(upsampling_ratios) - index],
                channels * channel_multiples[len(upsampling_ratios) - index - 1],
                stride,
                device,
                dtype,
                operations,
            )
            for index, stride in enumerate(upsampling_ratios)
        ])
        self.snake1 = Snake1d(channels, device=device, dtype=dtype)
        self.conv2 = wn_conv1d(
            operations,
            channels,
            audio_channels,
            kernel_size=7,
            padding=3,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        x = self.conv1(x)
        for block in self.block:
            x = block(x)
        return self.conv2(self.snake1(x))


class Cosmos3AudioVAE(nn.Module):
    def __init__(
        self,
        sampling_rate=48000,
        vocoder_input_dim=64,
        dec_dim=320,
        dec_c_mults=(1, 2, 4, 8, 16),
        dec_strides=(2, 4, 5, 6, 8),
        dec_out_channels=2,
        stereo=True,
        normalize_volume=True,
        hop_size=None,
        input_channels=1,
        enc_dim=192,
        enc_num_blocks=2,
        enc_n_fft=64,
        enc_hop_length=16,
        enc_latent_dim=128,
        enc_c_mults=(1, 2, 4),
        enc_strides=(4, 5, 6),
        enc_use_snake=True,
        padding_mode="zeros",
        encoder_enabled=True,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        if len(dec_c_mults) != len(dec_strides):
            raise ValueError("Cosmos3 audio decoder channel multipliers and strides must have the same length")
        self.sampling_rate = sampling_rate
        self.hop_size = math.prod(dec_strides) if hop_size is None else hop_size
        self.normalize_volume = normalize_volume

        self.encoder = None
        if encoder_enabled:
            self.encoder = Cosmos3AudioEncoder(
                input_channels,
                stereo,
                enc_dim,
                enc_latent_dim,
                tuple(enc_c_mults),
                tuple(enc_strides),
                enc_num_blocks,
                enc_n_fft,
                enc_hop_length,
                enc_use_snake,
                padding_mode,
                device,
                dtype,
                operations,
            )
        self.decoder = Cosmos3AudioDecoder(
            dec_dim,
            vocoder_input_dim,
            dec_out_channels,
            list(reversed(dec_strides)),
            dec_c_mults,
            device,
            dtype,
            operations,
        )

    def encode(self, audio):
        if self.encoder is None:
            raise ValueError("This Cosmos3 audio VAE does not contain encoder weights")
        if self.normalize_volume:
            audio = audio / (audio.abs().max() + 1e-5) * 0.95
        padding = (self.hop_size - audio.shape[-1] % self.hop_size) % self.hop_size
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        moments = self.encoder(audio).transpose(1, 2)
        mean, _ = moments.chunk(2, dim=1)
        return mean

    def decode(self, latents):
        squeeze = latents.ndim == 2
        if squeeze:
            latents = latents.unsqueeze(0)
        audio = self.decoder(latents).clamp(-1.0, 1.0)
        return audio.squeeze(0) if squeeze else audio

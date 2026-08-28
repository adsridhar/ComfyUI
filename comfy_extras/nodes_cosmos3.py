from fractions import Fraction
import json
from pathlib import Path

from typing_extensions import override

import torch
import nodes
import node_helpers
import folder_paths
import latent_preview

import comfy.latent_formats
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
from comfy.text_encoders.cosmos3 import Cosmos3Tokenizer
from comfy_api.latest import ComfyExtension, InputImpl, Types, io


Cosmos3TokenizerIO = io.Custom("COSMOS3_TOKENIZER")
Cosmos3GuardrailIO = io.Custom("COSMOS3_GUARDRAIL")
COSMOS3_DISTILL_SIGMAS = (1.0, 0.9375, 0.8333333333333334, 0.625, 0.0)


class Cosmos3GuardrailLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Cosmos3GuardrailLoader",
            display_name="Load Cosmos3 Guardrail",
            category="loaders/cosmos3",
            inputs=[],
            outputs=[Cosmos3GuardrailIO.Output(display_name="guardrail")],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        try:
            from cosmos_guardrail import CosmosSafetyChecker
        except ImportError as error:
            raise ImportError("Install the cosmos_guardrail package to use Cosmos3 guardrails.") from error
        return io.NodeOutput(CosmosSafetyChecker())


class Cosmos3ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Cosmos3ModelLoader",
            display_name="Load Cosmos3 Model",
            category="loaders/cosmos3",
            inputs=[
                io.Combo.Input("model_name", options=["distill-t2i", "distill-i2v", "nano"], default="distill-t2i"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                Cosmos3TokenizerIO.Output(display_name="tokenizer"),
                io.Vae.Output(display_name="vae"),
            ],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", f"cosmos3_{model_name}_diffusion_model.safetensors")
        tokenizer_path = Path(__file__).resolve().parents[1] / "comfy" / "text_encoders" / f"cosmos3_{model_name}_tokenizer"
        vae_path = folder_paths.get_full_path_or_raise("vae", f"cosmos3_{model_name}_vae.safetensors")

        model = comfy.sd.load_diffusion_model(model_path)
        tokenizer = Cosmos3Tokenizer(tokenizer_path)

        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()

        return io.NodeOutput(
            model,
            tokenizer,
            vae
        )


def _load_prompt(prompt):
    prompt_path = Path(prompt).expanduser()
    if not prompt_path.is_file():
        return prompt
    return json.dumps(json.loads(prompt_path.read_text(encoding="utf-8")), ensure_ascii=True, separators=(",", ":"))


def _check_prompt(prompt, guardrail):
    prompt = _load_prompt(prompt)
    if guardrail is None:
        return prompt
    try:
        safe = guardrail.check_text_safety(prompt)
    except PermissionError as error:
        # cosmos_guardrail 0.3.1 loads WordNet from a Hugging Face snapshot, while NLTK 3.10.3 refuses its symlinks.
        message = str(error)
        if "pathsec.open" not in message or "symlink" not in message:
            raise
        raise RuntimeError(
            "Cosmos guardrail cannot read NLTK data from Hugging Face's symlinked cache. "
            "Install WordNet into a regular NLTK data directory with `python -m nltk.downloader wordnet`."
        ) from error
    if not safe:
        raise ValueError("Cosmos guardrail blocked the prompt.")
    return prompt


def _check_video(images, guardrail):
    if guardrail is None:
        return images
    if images.ndim == 4:
        videos = images.unsqueeze(1)
    elif images.ndim == 5:
        videos = images
    else:
        raise ValueError(f"Expected decoded Cosmos output with 4 or 5 dimensions, got {images.ndim}.")

    checked_videos = []
    for video in videos:
        frames = video.clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu().numpy()
        checked = guardrail.check_video_safety(frames)
        if checked is None:
            raise ValueError("Cosmos guardrail blocked the generated output.")
        checked_videos.append(torch.from_numpy(checked.copy()).to(dtype=images.dtype).div(255.0))

    checked_videos = torch.stack(checked_videos)
    return checked_videos[:, 0] if images.ndim == 4 else checked_videos


def _conditioning(tokenizer, prompt, negative_prompt, width, height, num_frames=1, fps=24.0):
    prompt = _load_prompt(prompt)
    negative_prompt = _load_prompt(negative_prompt)
    metadata = {
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": fps,
    }
    positive_ids = tokenizer.encode(prompt, height, width, num_frames=num_frames, fps=fps)
    negative_ids = tokenizer.encode(negative_prompt, height, width, num_frames=num_frames, fps=fps, negative=True)
    positive = [[torch.zeros((*positive_ids.shape, 1)), {**metadata, "text_input_ids": positive_ids}]]
    negative = [[torch.zeros((*negative_ids.shape, 1)), {**metadata, "text_input_ids": negative_ids}]]
    return positive, negative


def _sample_distill(model, seed, cfg, positive, negative, latent):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"))
    batch_inds = latent.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    noise_mask = latent.get("noise_mask")
    sigmas = torch.FloatTensor(COSMOS3_DISTILL_SIGMAS)
    callback = latent_preview.prepare_callback(model, len(COSMOS3_DISTILL_SIGMAS) - 1)
    samples = comfy.sample.sample_custom(
        model,
        noise,
        cfg,
        comfy.samplers.sampler_object("lcm"),
        sigmas,
        positive,
        negative,
        latent_image,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


class Cosmos3DistillT2I(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Cosmos3DistillT2I",
            display_name="Cosmos3 Distill T2I",
            category="model/conditioning/cosmos3",
            inputs=[
                io.Model.Input("model"),
                Cosmos3TokenizerIO.Input("tokenizer"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Int.Input("width", default=1280, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=720, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Float.Input("cfg", default=6.0, min=0.0, max=100.0, step=0.1),
                Cosmos3GuardrailIO.Input("guardrail", optional=True),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        tokenizer,
        vae,
        prompt,
        negative_prompt,
        width,
        height,
        batch_size,
        seed,
        cfg,
        guardrail=None,
    ) -> io.NodeOutput:
        prompt = _check_prompt(prompt, guardrail)
        positive, negative = _conditioning(tokenizer, prompt, negative_prompt, width, height)
        latent = {
            "samples": torch.zeros(
                [batch_size, 48, 1, height // 16, width // 16],
                device=comfy.model_management.intermediate_device(),
            )
        }
        sampled = _sample_distill(model, seed, cfg, positive, negative, latent)
        images = vae.decode(sampled["samples"])
        images = _check_video(images, guardrail)
        if images.ndim == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return io.NodeOutput(images)


class Cosmos3DistillI2V(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Cosmos3DistillI2V",
            display_name="Cosmos3 Distill I2V",
            category="model/conditioning/cosmos3",
            inputs=[
                io.Model.Input("model"),
                Cosmos3TokenizerIO.Input("tokenizer"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Image.Input("image"),
                io.Int.Input("num_frames", default=189, min=5, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("fps", default=24, min=1, max=120),
                io.Int.Input("width", default=1280, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=720, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Float.Input("cfg", default=6.0, min=0.0, max=100.0, step=0.1),
                Cosmos3GuardrailIO.Input("guardrail", optional=True),
            ],
            outputs=[io.Video.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        tokenizer,
        vae,
        prompt,
        negative_prompt,
        image,
        num_frames,
        fps,
        width,
        height,
        batch_size,
        seed,
        cfg,
        guardrail=None,
    ) -> io.NodeOutput:
        prompt = _check_prompt(prompt, guardrail)
        positive, negative = _conditioning(tokenizer, prompt, negative_prompt, width, height, num_frames, fps)
        latent = torch.zeros(
            [1, 48, ((num_frames - 1) // 4) + 1, height // 16, width // 16],
            device=comfy.model_management.intermediate_device(),
        )
        noise_mask = torch.ones(
            [1, 1, latent.shape[2], latent.shape[3], latent.shape[4]],
            device=latent.device,
        )

        image = comfy.utils.common_upscale(image[:1, :, :, :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        anchor = vae.encode(image)
        latent[:, :, :anchor.shape[2]] = anchor
        noise_mask[:, :, :anchor.shape[2]] = 0.0

        latent = comfy.latent_formats.Wan22().process_out(latent) * noise_mask + latent * (1.0 - noise_mask)
        condition_mask = (noise_mask[:, 0, :, 0, 0] == 0).repeat(batch_size, 1)
        positive = node_helpers.conditioning_set_values(positive, {"condition_mask": condition_mask})
        negative = node_helpers.conditioning_set_values(negative, {"condition_mask": condition_mask})
        latent = {
            "samples": latent.repeat((batch_size,) + (1,) * (latent.ndim - 1)),
            "noise_mask": noise_mask.repeat((batch_size,) + (1,) * (noise_mask.ndim - 1)),
        }
        sampled = _sample_distill(model, seed, cfg, positive, negative, latent)
        images = vae.decode(sampled["samples"])
        images = _check_video(images, guardrail)
        if images.ndim == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, frame_rate=Fraction(fps)))
        return io.NodeOutput(video)


class Cosmos3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Cosmos3GuardrailLoader,
            Cosmos3ModelLoader,
            Cosmos3DistillT2I,
            Cosmos3DistillI2V,
        ]


async def comfy_entrypoint() -> Cosmos3Extension:
    return Cosmos3Extension()

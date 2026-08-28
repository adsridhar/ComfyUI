import argparse
import json
from pathlib import Path
import re
import shutil

from safetensors.torch import load_file, save_file


COMFYUI_ROOT = Path(__file__).resolve().parents[1]


def merge_transformer(transformer_dir: Path, output: Path, overwrite: bool) -> None:
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    config_path = transformer_dir / "config.json"

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    for key in ("sound_dim", "sound_gen", "sound_latent_fps"):
        config.pop(key, None)

    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    weight_map = {key: value for key, value in index["weight_map"].items() if not key.startswith("audio_")}
    state_dict = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = transformer_dir / shard_name
        shard = {key: value for key, value in load_file(shard_path, device="cpu").items() if key in weight_map}
        duplicate_keys = state_dict.keys() & shard.keys()
        if duplicate_keys:
            raise ValueError(f"Duplicate tensor keys in {shard_name}: {sorted(duplicate_keys)[:5]}")
        state_dict.update(shard)

    expected_keys = set(weight_map)
    if state_dict.keys() != expected_keys:
        missing = sorted(expected_keys - state_dict.keys())
        unexpected = sorted(state_dict.keys() - expected_keys)
        raise ValueError(f"Shard index mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "pt",
        "comfy.model_family": "cosmos3",
        "comfy.cosmos3_config": json.dumps(config, separators=(",", ":")),
    }
    save_file(state_dict, output, metadata=metadata)


def copy_tokenizer(tokenizer_dir: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace its files")
    shutil.copytree(tokenizer_dir, output, dirs_exist_ok=overwrite)


def _convert_residual_key(prefix: str, suffix: str) -> str:
    replacements = {
        "norm1.gamma": "residual.0.gamma",
        "conv1.weight": "residual.2.weight",
        "conv1.bias": "residual.2.bias",
        "norm2.gamma": "residual.3.gamma",
        "conv2.weight": "residual.6.weight",
        "conv2.bias": "residual.6.bias",
        "conv_shortcut.weight": "shortcut.weight",
        "conv_shortcut.bias": "shortcut.bias",
    }
    if suffix not in replacements:
        raise ValueError(f"Unsupported Diffusers Wan VAE residual key: {prefix}{suffix}")
    return prefix + replacements[suffix]


def _convert_attention_key(prefix: str, suffix: str) -> str:
    if suffix not in {"norm.gamma", "to_qkv.weight", "to_qkv.bias", "proj.weight", "proj.bias"}:
        raise ValueError(f"Unsupported Diffusers Wan VAE attention key: {prefix}{suffix}")
    return prefix + suffix


def _convert_resample_key(prefix: str, suffix: str) -> str:
    if suffix not in {"resample.1.weight", "resample.1.bias", "time_conv.weight", "time_conv.bias"}:
        raise ValueError(f"Unsupported Diffusers Wan VAE resample key: {prefix}{suffix}")
    return prefix + suffix


def convert_vae_key(key: str) -> str:
    top_level = {
        "quant_conv": "conv1",
        "post_quant_conv": "conv2",
        "encoder.conv_in": "encoder.conv1",
        "encoder.norm_out": "encoder.head.0",
        "encoder.conv_out": "encoder.head.2",
        "decoder.conv_in": "decoder.conv1",
        "decoder.norm_out": "decoder.head.0",
        "decoder.conv_out": "decoder.head.2",
    }
    match = re.fullmatch(r"(.+)\.(weight|bias|gamma)", key)
    if match is not None and match.group(1) in top_level:
        return f"{top_level[match.group(1)]}.{match.group(2)}"

    match = re.fullmatch(r"(encoder|decoder)\.mid_block\.resnets\.(0|1)\.(.+)", key)
    if match is not None:
        middle_index = 0 if match.group(2) == "0" else 2
        return _convert_residual_key(f"{match.group(1)}.middle.{middle_index}.", match.group(3))

    match = re.fullmatch(r"(encoder|decoder)\.mid_block\.attentions\.0\.(.+)", key)
    if match is not None:
        return _convert_attention_key(f"{match.group(1)}.middle.1.", match.group(2))

    match = re.fullmatch(r"encoder\.down_blocks\.([0-3])\.resnets\.([0-1])\.(.+)", key)
    if match is not None:
        return _convert_residual_key(f"encoder.downsamples.{match.group(1)}.downsamples.{match.group(2)}.", match.group(3))

    match = re.fullmatch(r"encoder\.down_blocks\.([0-2])\.downsampler\.(.+)", key)
    if match is not None:
        return _convert_resample_key(f"encoder.downsamples.{match.group(1)}.downsamples.2.", match.group(2))

    match = re.fullmatch(r"decoder\.up_blocks\.([0-3])\.resnets\.([0-2])\.(.+)", key)
    if match is not None:
        return _convert_residual_key(f"decoder.upsamples.{match.group(1)}.upsamples.{match.group(2)}.", match.group(3))

    match = re.fullmatch(r"decoder\.up_blocks\.([0-2])\.upsampler\.(.+)", key)
    if match is not None:
        return _convert_resample_key(f"decoder.upsamples.{match.group(1)}.upsamples.3.", match.group(2))

    raise ValueError(f"Unsupported Diffusers Wan VAE key: {key}")


def convert_vae_state_dict(state_dict: dict) -> dict:
    converted = {}
    for key, value in state_dict.items():
        converted_key = convert_vae_key(key)
        if converted_key in converted:
            raise ValueError(f"Duplicate converted VAE key: {converted_key}")
        converted[converted_key] = value
    return converted


def convert_vae(source: Path, output: Path) -> None:
    state_dict = convert_vae_state_dict(load_file(source, device="cpu"))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output, metadata={"format": "pt"})


def prepare_cosmos3(model_dir: Path, *, model: str = "nano", overwrite: bool = False) -> None:
    if re.fullmatch(r"[A-Za-z0-9_-]+", model) is None:
        raise ValueError("model must contain only letters, numbers, underscores, and hyphens")

    diffusion_output = COMFYUI_ROOT / "models" / "diffusion_models" / f"cosmos3_{model}_diffusion_model.safetensors"
    tokenizer_output = COMFYUI_ROOT / "comfy" / "text_encoders" / f"cosmos3_{model}_tokenizer"
    vae_output = COMFYUI_ROOT / "models" / "vae" / f"cosmos3_{model}_vae.safetensors"

    for output in (diffusion_output, tokenizer_output, vae_output):
        if output.exists() and not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    merge_transformer(model_dir / "transformer", diffusion_output, overwrite)
    copy_tokenizer(model_dir / "text_tokenizer", tokenizer_output, overwrite)
    convert_vae(model_dir / "vae" / "diffusion_pytorch_model.safetensors", vae_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Diffusers Cosmos3 model for ComfyUI.")
    parser.add_argument("model_dir", type=Path, help="Cosmos3 model directory.")
    parser.add_argument("--model", default="nano", help="Model name used in destination filenames (default: nano).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_cosmos3(args.model_dir, model=args.model, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

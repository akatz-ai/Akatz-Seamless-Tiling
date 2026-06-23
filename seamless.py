import copy
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from .base_node import ImageNode, LatentNode

TILING_MODES = ["enable", "x_only", "y_only", "disable"]
SEAM_REPAIR_MODES = ["both", "x_only", "y_only"]


class SeamlessTile(LatentNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tiling": (TILING_MODES,),
                "copy_model": (["Make a copy", "Modify in place"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"

    def run(self, model, copy_model, tiling):
        model_copy = _clone_model_patcher(model, copy_model == "Make a copy")
        target_model = _get_model_object_for_tiling(model_copy)
        tile_x, tile_y = _tiling_flags(tiling)
        make_circular_asymm(target_model, tile_x, tile_y)
        return (model_copy,)


class PeriodicLatentConstraint(LatentNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tiling": (TILING_MODES,),
                "strength": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "seam_width": (
                    "INT",
                    {"default": 4, "min": 1, "max": 128, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    def patch(self, model, tiling, strength, seam_width):
        model_copy = _clone_model_patcher(model, make_copy=True)
        tile_x, tile_y = _tiling_flags(tiling)

        def post_cfg(args):
            denoised = args["denoised"]
            return enforce_periodic_latent_edges(
                denoised,
                tile_x=tile_x,
                tile_y=tile_y,
                strength=strength,
                seam_width=seam_width,
            )

        model_copy.set_model_sampler_post_cfg_function(post_cfg)
        return (model_copy,)


class CircularVAEDecode(LatentNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tiling": (TILING_MODES,),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    def decode(self, samples, vae, tiling):
        vae_copy = _clone_vae(vae, make_copy=True)
        tile_x, tile_y = _tiling_flags(tiling)
        make_circular_asymm(vae_copy.first_stage_model, tile_x, tile_y)
        return (vae_copy.decode(samples["samples"]),)


class MakeCircularVAE(LatentNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "tiling": (TILING_MODES,),
                "copy_vae": (["Make a copy", "Modify in place"],),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "run"

    def run(self, vae, tiling, copy_vae):
        vae_copy = _clone_vae(vae, make_copy=copy_vae == "Make a copy")
        tile_x, tile_y = _tiling_flags(tiling)
        make_circular_asymm(vae_copy.first_stage_model, tile_x, tile_y)
        return (vae_copy,)


class OffsetImage(ImageNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "x_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1},
                ),
                "y_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, pixels, x_percent, y_percent):
        _, height, width, _ = pixels.size()
        y_offset = round(height * y_percent / 100)
        x_offset = round(width * x_percent / 100)
        return (_roll_image_pixels(pixels, y_offset, x_offset),)


class SeamRepairPrepare(ImageNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (SEAM_REPAIR_MODES,),
                "x_offset_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.5},
                ),
                "y_offset_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.5},
                ),
                "seam_width": (
                    "INT",
                    {"default": 64, "min": 1, "max": 1024, "step": 1},
                ),
                "feather": (
                    "INT",
                    {"default": 32, "min": 0, "max": 1024, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("shifted_image", "seam_mask")
    FUNCTION = "prepare"

    def prepare(self, image, mode, x_offset_percent, y_offset_percent, seam_width, feather):
        shifted, y_offset, x_offset = _roll_image_percent(
            image,
            y_offset_percent,
            x_offset_percent,
        )
        mask = _make_shifted_seam_mask(
            image,
            mode=mode,
            y_offset=y_offset,
            x_offset=x_offset,
            seam_width=seam_width,
            feather=feather,
        )
        return (shifted, mask)


class SeamRepairFinish(ImageNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_shifted_image": ("IMAGE",),
                "repaired_shifted_image": ("IMAGE",),
                "seam_mask": ("MASK",),
                "x_offset_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.5},
                ),
                "y_offset_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.5},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("restored_image", "shifted_composite")
    FUNCTION = "finish"

    def finish(
        self,
        original_shifted_image,
        repaired_shifted_image,
        seam_mask,
        x_offset_percent,
        y_offset_percent,
    ):
        mask = _image_mask(seam_mask, repaired_shifted_image)
        composite = original_shifted_image * (1.0 - mask) + repaired_shifted_image * mask
        restored, _, _ = _roll_image_percent(composite, -y_offset_percent, -x_offset_percent)
        return (restored, composite)


class MaskedLatentNoise(LatentNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "mask": ("MASK",),
                "strength": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 18446744073709551615},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"

    def apply(self, samples, mask, strength, seed):
        output = samples.copy()
        latent = samples["samples"]
        blend_strength = float(max(0.0, min(1.0, strength)))
        if blend_strength <= 0 or latent.ndim < 4:
            output["samples"] = latent.clone()
            return (output,)

        latent_mask = _latent_mask(mask, latent) * blend_strength
        noise = _seeded_noise_like(latent, seed)
        output["samples"] = latent * (1.0 - latent_mask) + noise * latent_mask
        return (output,)


def _tiling_flags(tiling):
    if tiling == "enable":
        return True, True
    if tiling == "x_only":
        return True, False
    if tiling == "y_only":
        return False, True
    return False, False


def _clone_model_patcher(model, make_copy):
    if not make_copy:
        return model

    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("SeamlessTile requires a ComfyUI model patcher with clone() support")

    try:
        return clone(disable_dynamic=True)
    except TypeError:
        return clone()
    except Exception:
        return clone()


def _get_model_object_for_tiling(model_patcher):
    model = getattr(model_patcher, "model", None)
    if model is not None:
        return model

    get_model_object = getattr(model_patcher, "get_model_object", None)
    if callable(get_model_object):
        try:
            return get_model_object("diffusion_model")
        except Exception as exc:
            raise TypeError("Could not access diffusion model object for tiling") from exc

    raise TypeError("SeamlessTile could not access the underlying model object")


def _clone_vae(vae, make_copy):
    if not make_copy:
        return vae

    patcher = getattr(vae, "patcher", None)
    if patcher is not None and hasattr(patcher, "clone"):
        vae_copy = copy.copy(vae)
        vae_copy.patcher = patcher.clone()
        patched_model = getattr(vae_copy.patcher, "model", None)
        if patched_model is not None:
            vae_copy.first_stage_model = patched_model
        return vae_copy

    return copy.deepcopy(vae)


def _roll_image_percent(image: Tensor, y_percent: float, x_percent: float):
    _, height, width, _ = image.size()
    y_offset = round(height * float(y_percent) / 100.0)
    x_offset = round(width * float(x_percent) / 100.0)
    return _roll_image_pixels(image, y_offset, x_offset), y_offset, x_offset


def _roll_image_pixels(image: Tensor, y_offset: int, x_offset: int):
    return torch.roll(image, shifts=(int(y_offset), int(x_offset)), dims=(1, 2))


def _make_shifted_seam_mask(
    image: Tensor,
    *,
    mode: str,
    y_offset: int,
    x_offset: int,
    seam_width: int,
    feather: int,
):
    batch, height, width, _ = image.size()
    mask = torch.zeros((height, width), device=image.device, dtype=image.dtype)

    if mode in ("both", "x_only"):
        x_mask = _periodic_axis_band(
            width,
            seam_position=x_offset % width,
            seam_width=seam_width,
            feather=feather,
            device=image.device,
            dtype=image.dtype,
        )
        mask = torch.maximum(mask, x_mask.view(1, width).expand(height, width))

    if mode in ("both", "y_only"):
        y_mask = _periodic_axis_band(
            height,
            seam_position=y_offset % height,
            seam_width=seam_width,
            feather=feather,
            device=image.device,
            dtype=image.dtype,
        )
        mask = torch.maximum(mask, y_mask.view(height, 1).expand(height, width))

    return mask.unsqueeze(0).expand(batch, height, width).contiguous()


def _periodic_axis_band(
    size: int,
    *,
    seam_position: int,
    seam_width: int,
    feather: int,
    device,
    dtype,
):
    coords = torch.arange(size, device=device, dtype=dtype) + 0.5
    center = torch.as_tensor(float(seam_position), device=device, dtype=dtype)
    distance = torch.abs(coords - center)
    distance = torch.minimum(
        distance,
        torch.as_tensor(float(size), device=device, dtype=dtype) - distance,
    )

    solid_radius = max(0.5, float(seam_width) / 2.0)
    feather_width = max(0.0, float(feather))
    if feather_width <= 0:
        return (distance <= solid_radius).to(dtype=dtype)

    falloff = (distance - solid_radius) / feather_width
    return (1.0 - falloff.clamp(0.0, 1.0)).clamp(0.0, 1.0)


def _image_mask(mask: Tensor, image: Tensor):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and image.shape[0] > 1:
        mask = mask.expand(image.shape[0], mask.shape[1], mask.shape[2])
    return mask.to(device=image.device, dtype=image.dtype).unsqueeze(-1)


def _latent_mask(mask: Tensor, latent: Tensor):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[1] != 1:
        mask = mask[:, :1]

    mask = mask.to(device=latent.device, dtype=torch.float32)
    if mask.shape[0] == 1 and latent.shape[0] > 1:
        mask = mask.expand(latent.shape[0], mask.shape[1], mask.shape[2], mask.shape[3])
    elif mask.shape[0] != latent.shape[0]:
        mask = mask[:1].expand(latent.shape[0], mask.shape[1], mask.shape[2], mask.shape[3])

    mask = F.interpolate(mask, size=latent.shape[-2:], mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0).to(dtype=latent.dtype)


def _seeded_noise_like(latent: Tensor, seed: int):
    seed = int(seed) % (2**63)
    try:
        generator = torch.Generator(device=latent.device)
        generator.manual_seed(seed)
        return torch.randn(
            latent.shape,
            generator=generator,
            device=latent.device,
            dtype=latent.dtype,
        )
    except RuntimeError:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randn(latent.shape, generator=generator, dtype=latent.dtype).to(latent.device)


def make_circular_asymm(model, tile_x: bool, tile_y: bool):
    for layer in model.modules():
        if not isinstance(layer, torch.nn.Conv2d):
            continue

        if not hasattr(layer, "_akatz_original_conv_forward"):
            layer._akatz_original_conv_forward = layer._conv_forward

        if not tile_x and not tile_y:
            layer._conv_forward = layer._akatz_original_conv_forward
            continue

        layer.padding_mode_x = "circular" if tile_x else "constant"
        layer.padding_mode_y = "circular" if tile_y else "constant"
        layer.padding_x = (
            layer._reversed_padding_repeated_twice[0],
            layer._reversed_padding_repeated_twice[1],
            0,
            0,
        )
        layer.padding_y = (
            0,
            0,
            layer._reversed_padding_repeated_twice[2],
            layer._reversed_padding_repeated_twice[3],
        )
        layer._conv_forward = _replacement_conv2d_forward.__get__(layer, Conv2d)
    return model


def enforce_periodic_latent_edges(
    latent: Tensor,
    *,
    tile_x: bool,
    tile_y: bool,
    strength: float,
    seam_width: int,
):
    if strength <= 0 or seam_width <= 0 or (not tile_x and not tile_y):
        return latent
    if latent.ndim < 4:
        return latent

    output = latent.clone()
    blend_strength = float(max(0.0, min(1.0, strength)))

    if tile_x:
        output = _blend_periodic_axis(output, axis=-1, strength=blend_strength, seam_width=seam_width)
    if tile_y:
        output = _blend_periodic_axis(output, axis=-2, strength=blend_strength, seam_width=seam_width)

    return output


def _blend_periodic_axis(latent: Tensor, *, axis: int, strength: float, seam_width: int):
    size = latent.shape[axis]
    band = min(int(seam_width), size // 2)
    if band <= 0:
        return latent

    front_slice = [slice(None)] * latent.ndim
    back_slice = [slice(None)] * latent.ndim
    front_slice[axis] = slice(0, band)
    back_slice[axis] = slice(size - band, size)

    front = latent[tuple(front_slice)]
    back = latent[tuple(back_slice)]
    back_aligned = torch.flip(back, dims=(axis,))
    average = (front + back_aligned) * 0.5

    weight = _edge_blend_weight(latent, axis=axis, band=band) * strength
    latent[tuple(front_slice)] = front.lerp(average, weight)
    latent[tuple(back_slice)] = back.lerp(torch.flip(average, dims=(axis,)), torch.flip(weight, dims=(axis,)))
    return latent


def _edge_blend_weight(latent: Tensor, *, axis: int, band: int):
    weights = torch.linspace(
        1.0,
        0.0,
        steps=band + 1,
        device=latent.device,
        dtype=latent.dtype,
    )[:-1]
    shape = [1] * latent.ndim
    shape[axis] = band
    return weights.reshape(shape)


def _replacement_conv2d_forward(
    self,
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
):
    working = F.pad(input, self.padding_x, mode=self.padding_mode_x)
    working = F.pad(working, self.padding_y, mode=self.padding_mode_y)
    return F.conv2d(
        working,
        weight,
        bias,
        self.stride,
        _pair(0),
        self.dilation,
        self.groups,
    )


SEAMLESS_NODE_CLASS_MAPPINGS = {
    "AK_SeamlessTile": SeamlessTile,
    "AK_PeriodicLatentConstraint": PeriodicLatentConstraint,
    "AK_CircularVAEDecode": CircularVAEDecode,
    "AK_MakeCircularVAE": MakeCircularVAE,
    "AK_OffsetImage": OffsetImage,
    "AK_SeamRepairPrepare": SeamRepairPrepare,
    "AK_SeamRepairFinish": SeamRepairFinish,
    "AK_MaskedLatentNoise": MaskedLatentNoise,
    # Compatibility aliases for workflows authored against ComfyUI-seamless-tiling.
    "SeamlessTile": SeamlessTile,
    "CircularVAEDecode": CircularVAEDecode,
    "MakeCircularVAE": MakeCircularVAE,
    "OffsetImage": OffsetImage,
}

SEAMLESS_NODE_DISPLAY_NAME_MAPPINGS = {
    "AK_SeamlessTile": "Akatz Seamless Tile",
    "AK_PeriodicLatentConstraint": "Akatz Periodic Latent Constraint",
    "AK_CircularVAEDecode": "Akatz Circular VAE Decode",
    "AK_MakeCircularVAE": "Akatz Make Circular VAE",
    "AK_OffsetImage": "Akatz Offset Image",
    "AK_SeamRepairPrepare": "Akatz Seam Repair Prepare",
    "AK_SeamRepairFinish": "Akatz Seam Repair Finish",
    "AK_MaskedLatentNoise": "Akatz Masked Latent Noise",
    "SeamlessTile": "Akatz Seamless Tile (compat)",
    "CircularVAEDecode": "Akatz Circular VAE Decode (compat)",
    "MakeCircularVAE": "Akatz Make Circular VAE (compat)",
    "OffsetImage": "Akatz Offset Image (compat)",
}

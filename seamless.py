import copy
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from .base_node import ImageNode, LatentNode

TILING_MODES = ["enable", "x_only", "y_only", "disable"]


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
        return (pixels.roll((y_offset, x_offset), (1, 2)),)


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
    "AK_CircularVAEDecode": CircularVAEDecode,
    "AK_MakeCircularVAE": MakeCircularVAE,
    "AK_OffsetImage": OffsetImage,
    # Compatibility aliases for workflows authored against ComfyUI-seamless-tiling.
    "SeamlessTile": SeamlessTile,
    "CircularVAEDecode": CircularVAEDecode,
    "MakeCircularVAE": MakeCircularVAE,
    "OffsetImage": OffsetImage,
}

SEAMLESS_NODE_DISPLAY_NAME_MAPPINGS = {
    "AK_SeamlessTile": "Akatz Seamless Tile",
    "AK_CircularVAEDecode": "Akatz Circular VAE Decode",
    "AK_MakeCircularVAE": "Akatz Make Circular VAE",
    "AK_OffsetImage": "Akatz Offset Image",
    "SeamlessTile": "Akatz Seamless Tile (compat)",
    "CircularVAEDecode": "Akatz Circular VAE Decode (compat)",
    "MakeCircularVAE": "Akatz Make Circular VAE (compat)",
    "OffsetImage": "Akatz Offset Image (compat)",
}

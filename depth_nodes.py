import numpy as np
import torch

from .base_node import ImageNode

SEAM_AXIS_CHOICES = ["both axes", "left/right only", "top/bottom only"]
NORMALIZE_CHOICES = ["clip to 0-1", "none", "frame min/max", "batch min/max"]
EPSILON = 1e-8


class AK_MakeDepthmapSeamless(ImageNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depthmap_batch": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_depthmap_seamless"

    def make_depthmap_seamless(self, depthmap_batch):
        depthmap_np = depthmap_batch.detach().cpu().numpy().copy()
        squeeze_batch = False

        if depthmap_np.ndim == 3:
            depthmap_np = np.expand_dims(depthmap_np, axis=0)
            squeeze_batch = True

        average = np.mean(depthmap_np, axis=0)
        average_gray = _to_grayscale(average)
        plane = _fit_plane_least_squares(average_gray)

        for index in range(depthmap_np.shape[0]):
            depthmap_gray = _to_grayscale(depthmap_np[index])
            seamless = depthmap_gray - plane
            min_value = seamless.min()
            max_value = seamless.max()
            range_value = max_value - min_value
            if range_value:
                seamless = (seamless - min_value) / range_value
            else:
                seamless = np.zeros_like(seamless)
            depthmap_np[index] = np.stack([seamless] * 3, axis=-1)

        if squeeze_batch:
            depthmap_np = depthmap_np[0]

        depthmap_seamless = torch.from_numpy(depthmap_np).to(
            device=depthmap_batch.device,
            dtype=depthmap_batch.dtype,
        )
        return (depthmap_seamless,)


class AK_MakeDepthmapSeamlessAdvanced(ImageNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depthmap_batch": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "seam_axis": (SEAM_AXIS_CHOICES,),
                "normalize": (NORMALIZE_CHOICES,),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_depthmap_seamless_advanced"

    def make_depthmap_seamless_advanced(
        self,
        depthmap_batch,
        strength,
        seam_axis,
        normalize,
    ):
        depthmap_np = depthmap_batch.detach().cpu().numpy().copy()
        squeeze_batch = False

        if depthmap_np.ndim == 3:
            depthmap_np = np.expand_dims(depthmap_np, axis=0)
            squeeze_batch = True

        axis_key = _seam_axis_key(seam_axis)
        original_gray_frames = np.stack(
            [_to_grayscale(frame).astype(np.float64, copy=False) for frame in depthmap_np],
            axis=0,
        )
        corrected_frames = []

        for depthmap_gray in original_gray_frames:
            periodic_depth = _periodic_plus_smooth_component(depthmap_gray, axis_key)
            corrected = depthmap_gray * (1.0 - strength) + periodic_depth * strength
            corrected_frames.append(corrected)

        corrected_batch = np.stack(corrected_frames, axis=0)
        corrected_batch = _normalize_depth_batch(corrected_batch, normalize)
        output = np.repeat(corrected_batch[..., np.newaxis], 3, axis=-1)

        if squeeze_batch:
            output = output[0]

        depthmap_seamless = torch.from_numpy(output).to(
            device=depthmap_batch.device,
            dtype=depthmap_batch.dtype,
        )
        return (depthmap_seamless,)


def _to_grayscale(depthmap):
    if depthmap.shape[-1] == 3:
        return np.mean(depthmap, axis=-1)
    return depthmap.squeeze(-1)


def _fit_plane_least_squares(depthmap):
    height, width = depthmap.shape
    x_axis = np.arange(width)
    y_axis = np.arange(height)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    design = np.c_[x_grid.flatten(), y_grid.flatten(), np.ones_like(x_grid).flatten()]
    coefficients, _, _, _ = np.linalg.lstsq(design, depthmap.flatten(), rcond=None)
    return coefficients[0] * x_grid + coefficients[1] * y_grid + coefficients[2]


def _seam_axis_key(seam_axis):
    if seam_axis == "left/right only":
        return "x"
    if seam_axis == "top/bottom only":
        return "y"
    return "both"


def _periodic_plus_smooth_component(depthmap, seam_axis):
    height, width = depthmap.shape
    boundary = np.zeros_like(depthmap, dtype=np.float64)

    if seam_axis in {"both", "y"}:
        top_bottom_delta = depthmap[0, :] - depthmap[-1, :]
        boundary[0, :] += top_bottom_delta
        boundary[-1, :] -= top_bottom_delta

    if seam_axis in {"both", "x"}:
        left_right_delta = depthmap[:, 0] - depthmap[:, -1]
        boundary[:, 0] += left_right_delta
        boundary[:, -1] -= left_right_delta

    y_axis = np.arange(height)
    x_axis = np.arange(width)
    denominator = (
        4
        - 2 * np.cos(2 * np.pi * y_axis / height)[:, np.newaxis]
        - 2 * np.cos(2 * np.pi * x_axis / width)[np.newaxis, :]
    )
    denominator[0, 0] = 1.0

    smooth_fft = np.fft.fft2(boundary) / denominator
    smooth_fft[0, 0] = 0.0
    smooth_component = np.real(np.fft.ifft2(smooth_fft))
    return depthmap - smooth_component


def _normalize_depth_batch(depthmaps, normalize):
    if normalize == "none":
        return depthmaps

    if normalize == "frame min/max":
        return np.stack([_normalize_minmax(frame) for frame in depthmaps], axis=0)

    if normalize == "batch min/max":
        return _normalize_minmax(depthmaps)

    return np.clip(depthmaps, 0.0, 1.0)


def _normalize_minmax(depthmap):
    min_value = depthmap.min()
    max_value = depthmap.max()
    range_value = max_value - min_value
    if range_value < EPSILON:
        return np.full_like(depthmap, 0.5)
    return (depthmap - min_value) / range_value


DEPTH_NODE_CLASS_MAPPINGS = {
    "AK_MakeDepthmapSeamless": AK_MakeDepthmapSeamless,
    "AK_MakeDepthmapSeamlessAdvanced": AK_MakeDepthmapSeamlessAdvanced,
}

DEPTH_NODE_DISPLAY_NAME_MAPPINGS = {
    "AK_MakeDepthmapSeamless": "Make Depthmap Seamless",
    "AK_MakeDepthmapSeamlessAdvanced": "Make Depthmap Seamless Advanced",
}

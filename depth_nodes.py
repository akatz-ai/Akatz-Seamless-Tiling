import numpy as np
import torch

from .base_node import ImageNode


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


DEPTH_NODE_CLASS_MAPPINGS = {
    "AK_MakeDepthmapSeamless": AK_MakeDepthmapSeamless,
}

DEPTH_NODE_DISPLAY_NAME_MAPPINGS = {
    "AK_MakeDepthmapSeamless": "Make Depthmap Seamless",
}


import cv2
import numpy as np
import torch

from .base_node import MaskNode


class AK_AnimatedDilationMaskLinear(MaskNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "shape": (["circle", "square"],),
                "dilate_per_frame": (
                    "INT",
                    {"default": 1, "min": 0, "max": 9999, "step": 1},
                ),
                "delay": (
                    "INT",
                    {"default": 0, "min": 0, "max": 99999999, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "dilate_mask_linear"

    def dilate_mask_linear(self, mask, shape, dilate_per_frame, delay):
        masks = mask.detach().cpu().numpy().copy()
        radius = 0

        for index, frame_mask in enumerate(masks):
            if index < delay:
                continue

            radius += dilate_per_frame

            if index > 0 and np.all(masks[index - 1] == 1):
                size = 1000
                kernel = np.ones((size * 2 + 1, size * 2 + 1), np.uint8)
                masks[index] = cv2.dilate(frame_mask, kernel, iterations=1)
                continue

            size = abs(int(radius))
            kernel = np.zeros((size * 2 + 1, size * 2 + 1), np.uint8)
            if shape == "circle":
                kernel = cv2.circle(kernel, (size, size), size, 1, -1)
            else:
                kernel += 1

            if radius > 0:
                masks[index] = cv2.dilate(frame_mask, kernel, iterations=1)
            else:
                masks[index] = cv2.erode(frame_mask, kernel, iterations=1)

        return (torch.from_numpy(masks).to(device=mask.device, dtype=mask.dtype),)


MASK_NODE_CLASS_MAPPINGS = {
    "AK_AnimatedDilationMaskLinear": AK_AnimatedDilationMaskLinear,
}

MASK_NODE_DISPLAY_NAME_MAPPINGS = {
    "AK_AnimatedDilationMaskLinear": "Animated Dilation Mask Linear",
}


import math

import torch

from .base_node import ImageNode

STRIP_DIRECTIONS = [
    "horizontal",
    "vertical",
    "diagonal up-right",
    "diagonal down-right",
    "diagonal up-left",
    "diagonal down-left",
]
BLEND_CURVES = ["cosine", "linear"]


class AK_TileStripJoin(ImageNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "direction": (STRIP_DIRECTIONS,),
                "overlap_pixels": (
                    "INT",
                    {"default": 96, "min": 0, "max": 512, "step": 1},
                ),
                "loop_preview": ("BOOLEAN", {"default": False}),
                "blend_curve": (BLEND_CURVES,),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "join_strip"

    def join_strip(self, images, direction, overlap_pixels, loop_preview, blend_curve):
        _validate_image_batch(images)

        frames = images
        if loop_preview and frames.shape[0] > 1:
            frames = torch.cat([frames, frames[:1]], dim=0)

        if frames.shape[0] == 1:
            return (frames,)

        image_axis = 2 if direction == "horizontal" else 1
        max_overlap = max(0, frames.shape[image_axis] - 1)
        overlap = min(max(0, int(overlap_pixels)), max_overlap)
        strip = _join_frames(frames, direction, overlap, blend_curve)
        return (strip.unsqueeze(0),)


class AK_TileStripWindow(ImageNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strip": ("IMAGE",),
                "direction": (STRIP_DIRECTIONS,),
                "window_width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 8192, "step": 8},
                ),
                "window_height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": 8192, "step": 8},
                ),
                "frames": (
                    "INT",
                    {"default": 24, "min": 1, "max": 512, "step": 1},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 100.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "wrap": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_windows"

    def make_windows(
        self,
        strip,
        direction,
        window_width,
        window_height,
        frames,
        start_percent,
        end_percent,
        wrap,
    ):
        _validate_image_batch(strip)

        image = strip[0]
        frame_count = max(1, int(frames))
        width = max(1, int(window_width))
        height = max(1, int(window_height))

        windows = []
        for index in range(frame_count):
            t = _window_t(index, frame_count, start_percent, end_percent, wrap)
            percent = start_percent + (end_percent - start_percent) * t
            y_offset, x_offset = _window_offsets(image, direction, width, height, percent, wrap)
            windows.append(_sample_window(image, y_offset, x_offset, height, width, wrap))

        return (torch.stack(windows, dim=0),)


def _validate_image_batch(images):
    if images.ndim != 4:
        raise ValueError(f"Expected IMAGE batch with shape [B,H,W,C], got {tuple(images.shape)}")
    if images.shape[0] < 1:
        raise ValueError("Expected at least one image frame")


def _window_t(index, frame_count, start_percent, end_percent, wrap):
    if frame_count == 1:
        return 0.0

    if _is_wrapped_full_loop(start_percent, end_percent, wrap):
        return index / frame_count

    return index / (frame_count - 1)


def _is_wrapped_full_loop(start_percent, end_percent, wrap):
    if not wrap:
        return False

    delta = end_percent - start_percent
    if math.isclose(delta, 0.0, abs_tol=1e-9):
        return False

    periods = delta / 100.0
    return math.isclose(periods, round(periods), abs_tol=1e-6)


def _join_frames(frames, direction, overlap, blend_curve):
    strip = frames[0]
    concat_dim = 1 if direction == "horizontal" else 0

    for index in range(1, frames.shape[0]):
        strip = _append_frame(strip, frames[index], concat_dim, overlap, blend_curve)

    return strip


def _append_frame(strip, frame, concat_dim, overlap, blend_curve):
    if overlap <= 0:
        return torch.cat([strip, frame], dim=concat_dim)

    if concat_dim == 1:
        keep = strip[:, :-overlap, :]
        strip_edge = strip[:, -overlap:, :]
        frame_edge = frame[:, :overlap, :]
        frame_tail = frame[:, overlap:, :]
        seam = _blend_edges(strip_edge, frame_edge, concat_dim, blend_curve)
        return torch.cat([keep, seam, frame_tail], dim=concat_dim)

    keep = strip[:-overlap, :, :]
    strip_edge = strip[-overlap:, :, :]
    frame_edge = frame[:overlap, :, :]
    frame_tail = frame[overlap:, :, :]
    seam = _blend_edges(strip_edge, frame_edge, concat_dim, blend_curve)
    return torch.cat([keep, seam, frame_tail], dim=concat_dim)


def _blend_edges(first, second, concat_dim, blend_curve):
    weights = torch.linspace(
        0.0,
        1.0,
        first.shape[concat_dim],
        device=first.device,
        dtype=first.dtype,
    )
    if blend_curve == "cosine":
        weights = 0.5 - 0.5 * torch.cos(weights * math.pi)

    if concat_dim == 1:
        weights = weights.view(1, -1, 1)
    else:
        weights = weights.view(-1, 1, 1)

    return first * (1.0 - weights) + second * weights


def _window_offsets(image, direction, width, height, percent, wrap):
    source_height, source_width = image.shape[0], image.shape[1]

    if direction == "horizontal":
        travel = source_width if wrap else max(0, source_width - width)
        x_offset = round(travel * percent / 100.0)
        y_offset = round((source_height - height) / 2)
        return y_offset, x_offset

    travel = source_height if wrap else max(0, source_height - height)
    if direction == "vertical":
        y_offset = round(travel * percent / 100.0)
        x_offset = round((source_width - width) / 2)
        return y_offset, x_offset

    x_travel = source_width if wrap else max(0, source_width - width)
    y_travel = source_height if wrap else max(0, source_height - height)
    x_delta = round(x_travel * percent / 100.0)
    y_delta = round(y_travel * percent / 100.0)

    x_center = round((source_width - width) / 2)
    y_center = round((source_height - height) / 2)

    if direction == "diagonal up-right":
        return y_center - y_delta, x_center + x_delta
    if direction == "diagonal down-right":
        return y_center + y_delta, x_center + x_delta
    if direction == "diagonal up-left":
        return y_center - y_delta, x_center - x_delta
    if direction == "diagonal down-left":
        return y_center + y_delta, x_center - x_delta

    raise ValueError(f"Unsupported tile window direction: {direction}")


def _sample_window(image, y_offset, x_offset, height, width, wrap):
    source_height, source_width = image.shape[0], image.shape[1]

    if wrap:
        y_indexes = (torch.arange(height, device=image.device) + y_offset) % source_height
        x_indexes = (torch.arange(width, device=image.device) + x_offset) % source_width
    else:
        y_indexes = (torch.arange(height, device=image.device) + y_offset).clamp(0, source_height - 1)
        x_indexes = (torch.arange(width, device=image.device) + x_offset).clamp(0, source_width - 1)

    return image.index_select(0, y_indexes).index_select(1, x_indexes)


STRIP_NODE_CLASS_MAPPINGS = {
    "AK_TileStripJoin": AK_TileStripJoin,
    "AK_TileStripWindow": AK_TileStripWindow,
}

STRIP_NODE_DISPLAY_NAME_MAPPINGS = {
    "AK_TileStripJoin": "Akatz Tile Strip Join",
    "AK_TileStripWindow": "Akatz Tile Strip Window",
}

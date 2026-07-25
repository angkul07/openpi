"""On-the-fly image augmentation applied during training only.

Design constraints (from the YAM teleop-oversampling experiment spec, section 5):

  * Fresh random parameters on *every* dataloader visit to a frame -- there are no
    enumerated "augmented versions" of the dataset. Repeated visits to the same
    frame (unavoidable at 3-6 epochs) therefore see different pixels.
  * Photometric augmentation on ALL cameras, identical distribution for every
    source dataset. Augmenting only one source would create a spurious
    source-identifying feature the policy could latch onto.
  * Geometric augmentation on the TOP camera only. Wrist views are pose-coupled:
    shifting the image changes the implied end-effector pose while the action
    label stays fixed, which injects observation/action misalignment.
  * No flips or rotations (they swap the left/right arms visually while the
    14-dim action does not swap) and no noise on state/actions.
  * Applied to uint8 HWC images *before* the model transforms; image values are
    converted to [-1, 1] floats later, inside `model.Observation.from_dict`.
  * Wired through `DataConfig.train_only_transforms`, which is applied by the
    training data loader and NOT by the inference/policy transform chain, so
    augmentation is off at eval/serving by construction.

Randomness comes from torch's global RNG, which `torch.utils.data.DataLoader`
seeds per worker from its `generator` (openpi seeds that with `TrainConfig.seed`),
so a run is reproducible for a fixed seed and worker count.
"""

import dataclasses
import functools
import math

import numpy as np

import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class ImageAugmentConfig:
    """Ranges for the training-time image augmentation stack.

    Photometric values follow torchvision `ColorJitter` conventions: a scalar `x`
    means the factor is drawn from U[1-x, 1+x] (U[-x, x] for hue).
    """

    # Photometric -- applied to every camera.
    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.10
    hue: float = 0.03

    # Geometric -- random square crop covering this fraction of the image *area*,
    # resized back to the original resolution. Applied only to `geometric_keys`.
    # Set to (1.0, 1.0) to disable.
    crop_scale: tuple[float, float] = (0.95, 1.0)
    geometric_keys: tuple[str, ...] = ("base_0_rgb",)

    def __post_init__(self):
        lo, hi = self.crop_scale
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError(f"crop_scale must satisfy 0 < lo <= hi <= 1, got {self.crop_scale}")


@functools.lru_cache(maxsize=8)
def _color_jitter(config: "ImageAugmentConfig"):
    """Build (and cache per process) the ColorJitter op for a config.

    Cached because the transform is pickled into each spawned dataloader worker;
    we only want to pay construction once per worker process.
    """
    from torchvision.transforms import v2  # noqa: PLC0415  (heavy import, workers only)

    return v2.ColorJitter(
        brightness=config.brightness,
        contrast=config.contrast,
        saturation=config.saturation,
        hue=config.hue,
    )


def _sample_crop(height: int, width: int, crop_scale: tuple[float, float]):
    """Sample a square crop box covering `crop_scale` of the area (torch RNG)."""
    import torch  # noqa: PLC0415

    lo, hi = crop_scale
    scale = float(torch.empty(1).uniform_(lo, hi).item())
    side = int(round(math.sqrt(scale * height * width)))
    side = max(1, min(side, height, width))
    top = int(torch.randint(0, height - side + 1, (1,)).item())
    left = int(torch.randint(0, width - side + 1, (1,)).item())
    return top, left, side


def _augment_image(image: np.ndarray, *, geometric: bool, config: ImageAugmentConfig) -> np.ndarray:
    """Augment a single HWC uint8 image and return it in the same layout/dtype."""
    import torch  # noqa: PLC0415
    from torchvision.transforms.v2 import functional as tv_f  # noqa: PLC0415

    if image.dtype != np.uint8:
        # Everything upstream (YamInputs._parse_image) hands us uint8; fail loudly
        # rather than silently mangling a float image's range.
        raise ValueError(f"ImageAugment expects uint8 images, got {image.dtype}")

    height, width = image.shape[:2]
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)  # HWC -> CHW

    if geometric and config.crop_scale != (1.0, 1.0):
        top, left, side = _sample_crop(height, width, config.crop_scale)
        tensor = tv_f.resized_crop(tensor, top, left, side, side, [height, width], antialias=True)

    tensor = _color_jitter(config)(tensor)

    return tensor.permute(1, 2, 0).contiguous().numpy()  # CHW -> HWC


@dataclasses.dataclass(frozen=True)
class ImageAugment(_transforms.DataTransformFn):
    """Applies `ImageAugmentConfig` to `data["image"]`. Training only."""

    config: ImageAugmentConfig = dataclasses.field(default_factory=ImageAugmentConfig)

    def __call__(self, data: dict) -> dict:
        images = data.get("image")
        if not images:
            return data
        data["image"] = {
            key: _augment_image(
                np.asarray(img),
                geometric=key in self.config.geometric_keys,
                config=self.config,
            )
            for key, img in images.items()
        }
        return data

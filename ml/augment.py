"""Training augmentations that need more than torchvision.

paste_on_background: many field datasets photograph picked leaves on white or
grey paper. The leaf is cut out (paper is bright and colourless) and pasted on a
real field background, so the model learns the leaf, not the paper.
"""

import random

from PIL import Image, ImageFilter, ImageOps

PAPER_MAX_SATURATION = 45  # HSV saturation (0-255) below which a pixel counts as paper or shadow
WORK_SIZE = 512


def leaf_mask(im):
    """Soft mask of the leaf, or None when segmentation looks unreliable."""
    s = im.convert("HSV").getchannel("S")
    mask = s.point(lambda v: 255 if v >= PAPER_MAX_SATURATION else 0)
    # Close small holes (pale lesions, mold), then remove specks.
    mask = mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
    mask = mask.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    coverage = sum(mask.histogram()[128:]) / (mask.width * mask.height)
    if not 0.04 <= coverage <= 0.85:
        return None
    return mask.filter(ImageFilter.GaussianBlur(2))


def paste_on_background(im, background_paths, rng=random):
    im = im.copy()
    im.thumbnail((WORK_SIZE, WORK_SIZE))
    mask = leaf_mask(im)
    if mask is None:
        return im
    with Image.open(rng.choice(background_paths)) as bg:
        bg = ImageOps.fit(bg.convert("RGB"), im.size)
    return Image.composite(im, bg, mask)

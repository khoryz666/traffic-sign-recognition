"""Shared ROI preprocessing for both HOG and HSV feature extraction.

Previously HOG and HSV extracted their ROI differently: HOG cropped to the
detected contour's bounding box, resized to 32x32, and fell back to the
whole image when no contour was found (so it never dropped a sample). HSV
masked by the exact contour shape (no crop/resize) and returned None
(dropping the sample) when no contour was found. Those are now unified into
one function that combines both approaches - crop to the bounding box *and*
keep the exact contour mask within that crop, both resized to ROI_SIZE - and
one fallback behavior: drop the sample when segmentation finds nothing.
"""

import cv2
import numpy as np
from PIL import Image

from pipeline.segmentation import (
    extract_roi_from_mask,
    segment_blue,
    segment_red,
    segment_yellow,
)

ROI_SIZE = 256


def get_combined_mask(image_bgr):
    red_mask = segment_red(image_bgr)
    blue_mask = segment_blue(image_bgr)
    yellow_mask = segment_yellow(image_bgr)
    return red_mask | blue_mask | yellow_mask


def get_roi(image_rgb):
    """Detect the traffic sign, crop to its bounding box, and resize both
    the cropped RGB image and a matching contour-shape mask to ROI_SIZE x
    ROI_SIZE.

    Returns (roi_rgb, roi_mask) as uint8 arrays, or None if no traffic-sign
    contour was found - callers should drop the sample in that case, rather
    than silently falling back to the whole image.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    combined_mask = get_combined_mask(image_bgr)

    if np.count_nonzero(combined_mask) == 0:
        return None

    roi_rgb, bbox, best_contour = extract_roi_from_mask(image_rgb, combined_mask)

    if roi_rgb is None:
        return None

    x, y, w, h = bbox

    # The exact sign silhouette within the crop (HSV's shape-precision
    # advantage), rather than the whole rectangular bounding box.
    shifted_contour = best_contour - [x, y]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(roi_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

    roi_rgb_resized = np.array(
        Image.fromarray(roi_rgb).resize((ROI_SIZE, ROI_SIZE), Image.Resampling.LANCZOS)
    )
    roi_mask_resized = cv2.resize(
        roi_mask, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_NEAREST
    )

    return roi_rgb_resized, roi_mask_resized

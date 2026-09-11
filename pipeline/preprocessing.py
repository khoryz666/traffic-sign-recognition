"""Shared segment + ROI-extraction routine for both HOG and HSV feature
extraction.

This is a direct port of 04_hsv_color_histogram_chinese.py's
extract_hsv_features_from_image() segmentation/contour-selection logic - now
the single standard for both feature types, replacing the earlier version of
this module that unified it with HOG's separate bbox-crop-and-resize-to-32x32
behavior.

Like 04, there is no crop and no resize: the best-scoring contour is filled
into a mask at the image's original resolution. Unlike the version before
this fix, the returned image is not the original untouched photo - it is
that image with the mask already applied, so every pixel outside the sign's
contour is pure black. Both feature extractors therefore only ever see the
segmented region, never the surrounding background. A sample is dropped
(returns None) when no color region or no contour is found - exactly 04's
behavior.

Consequence for HOG: pipeline.features.extract_hog_features receives a
variable-size input (this module never resizes), so it crops to the mask's
bounding box and resizes that crop itself to reach a constant 1764-dim
vector - see pipeline/features.py for that step. HSV needs none of this: a
masked histogram's dimensionality never depended on the input's spatial
size.
"""

import cv2
import numpy as np

from pipeline.segmentation import find_best_contour, segment_blue, segment_red, segment_yellow


def get_combined_mask(image_bgr):
    red_mask = segment_red(image_bgr)
    blue_mask = segment_blue(image_bgr)
    yellow_mask = segment_yellow(image_bgr)
    return red_mask | blue_mask | yellow_mask


def get_roi(image_rgb):
    """Segment the traffic sign and return (roi_rgb, mask), unchanged in
    size from the input, or None if no sign was found.

    mask is zero everywhere except the filled shape of the best-scoring
    contour, at image_rgb's original resolution. roi_rgb is image_rgb with
    that mask already applied - every pixel outside the sign's contour is
    pure black (0, 0, 0); only the segmented region keeps its original
    pixel values. Callers that need a fixed-size input (HOG) are
    responsible for resizing/cropping themselves.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    combined_mask = get_combined_mask(image_bgr)

    if np.count_nonzero(combined_mask) == 0:
        return None

    best_contour = find_best_contour(combined_mask)

    if best_contour is None:
        return None

    mask = np.zeros_like(combined_mask)
    cv2.drawContours(mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    roi_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    return roi_rgb, mask

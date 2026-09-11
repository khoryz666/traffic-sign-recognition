"""Shared segment + ROI-extraction routine for both HOG and HSV feature
extraction.

Segmentation used to OR red/blue/yellow masks together and pick one
contour from the merged mask, scored on area x circularity x solidity
only. Ported from @kahyikang's
03_automatic_colour_segmentation_for_dataset.ipynb, it now instead finds
each colour's own best candidate contour independently and picks the
overall winner across colours (segmentation.select_best_contour_multi_colour)
- so two different-coloured blobs sitting next to each other in the same
image can no longer merge into one bad contour - and the scoring adds a
hue-agreement and colour-coverage check the merged-mask version never had.
See pipeline/segmentation.py for that scoring.

Like before, there is no crop and no resize here: the winning contour is
filled into a mask at the image's original resolution. The returned image
is that image with the mask already applied, so every pixel outside the
sign's contour is pure black. Both feature extractors therefore only ever
see the segmented region, never the surrounding background. A sample is
dropped (returns None) when no colour produces a usable candidate contour.

Consequence for HOG: pipeline.features.extract_hog_features receives a
variable-size input (this module never resizes), so it crops to the mask's
bounding box and resizes that crop itself to reach a constant 1764-dim
vector - see pipeline/features.py for that step. HSV needs none of this: a
masked histogram's dimensionality never depended on the input's spatial
size.
"""

import cv2
import numpy as np

from pipeline.segmentation import select_best_contour_multi_colour


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
    best_contour = select_best_contour_multi_colour(image_bgr)

    if best_contour is None:
        return None

    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    roi_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    return roi_rgb, mask

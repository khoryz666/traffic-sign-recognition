"""HOG and HSV-histogram feature extraction over a preprocessed ROI.

Both functions consume the same `pipeline.preprocessing.get_roi()` output
unchanged - no crop, no resize happens in preprocessing itself, by either
feature extractor - so the two methods are compared on an identical
starting ROI, and a dropped sample (no contour found) is dropped from both
feature sets identically. That ROI already has the background blacked out
(see pipeline/preprocessing.py) - pixels outside the sign's contour are
pure black - so HOG's gradients, not just HSV's histogram, are computed
only over the segmented sign, never the surrounding scene.

extract_hog_features needs a fixed-size input to produce a classifier-ready,
constant-length vector. That sizing is HOG's own concern, not
preprocessing's: it crops roi_rgb to roi_mask's bounding box (discarding
the black margin outside the sign) and resizes that crop (Lanczos) to a
fixed HOG_ROI_SIZE x HOG_ROI_SIZE canvas - the same crop + resize steps,
and the same 32x32/pixels_per_cell=(4,4) configuration, as deprecated
03_hog_features_chinese.py, reproducing its 1764-dim output. HSV needs
none of this: a masked histogram's dimensionality never depended on the
input's spatial size in the first place.
"""

import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog

HOG_ORIENTATIONS = 9
HOG_ROI_SIZE = 32  # deprecated 03's resize target; paired with a 4x4 cell
# size this reproduces its exact 1764-dim output.
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

HSV_HUE_BINS = 30
HSV_SATURATION_BINS = 32


def prepare_hog_input(roi_rgb, roi_mask):
    """Crop roi_rgb to roi_mask's bounding box (dropping the black margin
    outside the sign) and resize that crop to a fixed HOG_ROI_SIZE x
    HOG_ROI_SIZE canvas (Lanczos). This is HOG's own sizing step, kept out
    of pipeline.preprocessing.get_roi() so HSV extraction and the shared
    segmentation stay untouched by it."""
    x, y, w, h = cv2.boundingRect(roi_mask)
    w = max(1, w)
    h = max(1, h)
    cropped = roi_rgb[y:y + h, x:x + w]

    return np.array(
        Image.fromarray(cropped).resize(
            (HOG_ROI_SIZE, HOG_ROI_SIZE), Image.Resampling.LANCZOS
        )
    )


def extract_hog_features(roi_rgb, roi_mask):
    """HOG over the segmented sign, cropped and resized to a fixed canvas
    by prepare_hog_input() - every image produces the same 1764-dim
    vector."""
    hog_input = prepare_hog_input(roi_rgb, roi_mask)

    features = hog(
        hog_input,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        transform_sqrt=True,  # power-law compression, helps with shadows
        channel_axis=-1,
    )
    return features


def expected_hog_dim(
    image_shape=(HOG_ROI_SIZE, HOG_ROI_SIZE),
    pixels_per_cell=HOG_PIXELS_PER_CELL,
    cells_per_block=HOG_CELLS_PER_BLOCK,
    orientations=HOG_ORIENTATIONS,
):
    """The feature-vector length extract_hog_features() produces. Defaults
    to the fixed HOG_ROI_SIZE canvas every crop is resized to (so this is a
    single constant, 1764); image_shape is overridable for
    reference/testing against a different configuration."""
    height, width = image_shape[:2]
    cells_y = height // pixels_per_cell[0]
    cells_x = width // pixels_per_cell[1]
    blocks_y = max(cells_y - cells_per_block[0] + 1, 0)
    blocks_x = max(cells_x - cells_per_block[1] + 1, 0)
    return blocks_y * blocks_x * cells_per_block[0] * cells_per_block[1] * orientations


def extract_hsv_features(roi_rgb, roi_mask):
    """Normalized Hue/Saturation histogram, masked to the sign's exact
    contour shape so background pixels inside the bounding box don't
    contribute."""
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        roi_mask,
        [HSV_HUE_BINS, HSV_SATURATION_BINS],
        [0, 180, 0, 256],
    )

    hist = cv2.normalize(hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist.flatten()

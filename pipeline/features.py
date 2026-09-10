"""HOG and HSV-histogram feature extraction over a preprocessed ROI.

Both functions consume the same `pipeline.preprocessing.get_roi()` output,
so a dropped sample (no contour found) is dropped from both feature sets
identically - no row-alignment drift between the HOG and HSV notebooks.
"""

import cv2
from skimage.feature import hog

# pixels_per_cell scales with ROI_SIZE (pipeline.preprocessing.ROI_SIZE) to
# keep the same 8x8-cells-per-image grid the classifiers were tuned around
# (32x32 image / (4, 4) cells == 256x256 image / (32, 32) cells), so the
# feature vector stays exactly 1764-dimensional.
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (32, 32)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

HSV_HUE_BINS = 30
HSV_SATURATION_BINS = 32


def extract_hog_features(roi_rgb):
    """HOG over the full RGB ROI (all 3 color channels)."""
    features = hog(
        roi_rgb,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        transform_sqrt=True,  # power-law compression, helps with shadows
        channel_axis=-1,
    )
    return features


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

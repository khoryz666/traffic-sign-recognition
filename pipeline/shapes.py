"""Shape classification (circle/triangle/octagon/...) for the ROI-segmentation
demo notebook.

Moved out of 01-roi-segmentation/02_shape_detection.ipynb, which used to pull
in segmentation.py's functions via a Jupyter `%run "Color Segmentation.ipynb"`
magic (it works, but only inside a live Jupyter kernel with the right
working directory - a real import is simpler and more robust). extract_features
and classify_shape have since moved into pipeline/segmentation.py itself
(find_best_candidate needs shape_score to compare red/blue/yellow
candidates), so this module now just re-imports them.

Diagnostic/demo only - shape classification does not feed the classifiers.
"""

import cv2

from pipeline.segmentation import (  # noqa: F401  (re-exported for convenience)
    classify_shape,
    extract_features,
)


def preprocess_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blurred = cv2.GaussianBlur(img_rgb, (5, 5), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    return img_rgb, hsv


def draw_result(img_rgb, contour, shape, bounding_box):

    result = img_rgb.copy()

    x, y, w, h = bounding_box

    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        shape,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    return result

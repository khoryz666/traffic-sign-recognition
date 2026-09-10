"""Shape classification (circle/triangle/octagon/...) for the ROI-segmentation
demo notebook.

Moved out of 01-roi-segmentation/02_shape_detection.ipynb, which used to pull
in segmentation.py's functions via a Jupyter `%run "Color Segmentation.ipynb"`
magic (it works, but only inside a live Jupyter kernel with the right
working directory - a real import is simpler and more robust). This module
also drops Shape Detection's own duplicate `find_best_contour` (tuned with a
1000px minimum area and interactive debug prints) in favor of the single
`segmentation.find_best_contour` used everywhere else in the pipeline.

Diagnostic/demo only - shape classification does not feed the classifiers.
"""

import cv2
import numpy as np

from pipeline.segmentation import find_best_contour  # noqa: F401  (re-exported for convenience)


def preprocess_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blurred = cv2.GaussianBlur(img_rgb, (5, 5), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    return img_rgb, hsv


def extract_features(contour):

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return None

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    if area < 3000:
        epsilon = 0.018 * perimeter
    else:
        epsilon = 0.02 * perimeter

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    vertices = len(approx)

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = w / float(h)

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "vertices": vertices,
        "aspect_ratio": aspect_ratio,
        "bounding_box": (x, y, w, h),
        "approx": approx
    }


def classify_shape(features):

    vertices = features["vertices"]
    circularity = features["circularity"]
    aspect_ratio = features["aspect_ratio"]

    if vertices == 3:
        return "Triangle"

    elif vertices == 4:

        if 0.9 <= aspect_ratio <= 1.1:
            return "Square"
        else:
            return "Rectangle"

    elif circularity >= 0.82:
        return "Circle"

    elif 7 <= vertices <= 9:
        return "Octagon"

    else:
        return "Unknown"


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

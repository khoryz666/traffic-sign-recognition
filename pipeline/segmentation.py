"""Color segmentation and contour selection for traffic-sign ROI detection.

Moved out of 01-roi-segmentation/01_color_segmentation.ipynb, which used to be
copy-pasted (byte-for-byte, confirmed) into
02-feature-extraction/01_hog_features_chinese.ipynb,
02-feature-extraction/02_hsv_color_histogram_chinese.ipynb, and
03-classifier/03_svm_classifier.ipynb. This is the single source of truth
now; those notebooks import it instead.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def load_image(filepath):
    img = cv2.imread(filepath)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def apply_gaussian_blur(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)


# ---------------------------------------------------------------------
# Red segmentation
# ---------------------------------------------------------------------


def get_red_diff_map(image):
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)

    # In OpenCV HSV, Hue is 0-179. Pure red is around 0 and 180.
    # Calculate distance from pure red.
    dist = np.minimum(H, 180 - H)

    # Map distance to a redness score (0 to 1)
    # If dist is 0 (pure red), score is 1.0. If dist >= 15 (orange/yellow), score is 0.0.
    redness = np.maximum(0, 15 - dist) / 15.0

    # Multiply by Saturation to ignore white/gray pixels, but be tolerant of glare.
    # If S > 80, factor is 1.0. If S < 30, factor is 0.0. This prevents glare from breaking the ring.
    sat = np.clip((S - 30) / 50.0, 0, 1)

    # Combine to form the difference map
    diff = redness * sat * 255.0

    # Normalize to 0-255 to ensure Otsu's thresholding has a full range to work with
    diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return diff.astype(np.uint8)


def adaptive_color_threshold(diff_map):
    # Otsu's thresholding dynamically finds the best threshold value
    ret, mask = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def clean_morphology(mask):
    # Use elliptical kernels for natural rounding around circular traffic signs
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Opening FIRST: Erosion followed by Dilation. This breaks thin bridges connecting background noise to the sign.
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    # Closing NEXT: Dilation followed by Erosion. An elliptical 5x5 kernel smoothly bridges gaps in the sign ring.
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel)
    return closing


def extract_and_mask(image, cleaned_mask):
    # Use 4-connectivity to prevent diagonal bridges from attaching background red objects (like awnings) to the sign
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=4)

    # If only background is found
    if num_labels <= 1:
        return np.zeros_like(image)

    # Find the label with the largest area (excluding the background which is label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create a new mask containing only the largest connected component
    largest_mask = np.uint8(labels == largest_label) * 255

    # Find external contours of the isolated sign component
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(largest_mask)
    if contours:
        # Draw all external contours filled (-1 thickness) to include the interior area of the sign naturally
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    else:
        filled_mask = largest_mask

    # Bitwise-AND the original image with the filled mask
    result = cv2.bitwise_and(image, image, mask=filled_mask)
    return result


def segment_red(img, debug=False):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blurred = apply_gaussian_blur(img)

    diff = get_red_diff_map(blurred)

    mask = adaptive_color_threshold(diff)

    cleaned = clean_morphology(mask)

    result = extract_and_mask(img, cleaned)

    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    _, final_mask = cv2.threshold(
        gray,
        1,
        255,
        cv2.THRESH_BINARY
    )

    if debug:
        debug_images = {
            "filtered": diff,
            "threshold": cleaned,
            "filled": final_mask
        }
        return final_mask, debug_images

    return final_mask


# ---------------------------------------------------------------------
# Blue segmentation
# ---------------------------------------------------------------------


def remove_small_components(binary_mask, min_area_ratio):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8  # include 8 directions
    )

    cleaned_mask = np.zeros_like(binary_mask)

    image_area = binary_mask.shape[0] * binary_mask.shape[1]  # shape[0]=image height , shape[1]=image width
    min_area = image_area * min_area_ratio

    for label in range(1, num_labels):  # 0 is background

        area = stats[label, cv2.CC_STAT_AREA]  # get the area of connected region

        if area >= min_area:
            cleaned_mask[labels == label] = 255

    return cleaned_mask


def enhance_img(image):

    image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    filtered = cv2.bilateralFilter(
        image_cvt,
        d=7,
        sigmaColor=40,
        sigmaSpace=40
    )

    filtered_cvt = cv2.cvtColor(filtered, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(filtered_cvt)

    hue_mask = cv2.inRange(
        h,
        99,
        125
    )

    # Adaptive thresholding on Saturation
    saturation_mask = cv2.adaptiveThreshold(
        s,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        91,
        2
    )

    value_mask = cv2.inRange(
        v,
        42,
        255
    )

    # Combine Hue, Saturation and Value masks
    blue_mask = cv2.bitwise_and(
        hue_mask,
        saturation_mask
    )

    blue_mask = cv2.bitwise_and(
        blue_mask,
        value_mask
    )

    blue_mask = remove_small_components(
        blue_mask,
        min_area_ratio=0.05
    )

    # Morphological closing:
    # connects nearby white regions and fills small black gaps
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (4, 4)
    )

    final_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=1
    )

    return blue_mask, filtered, final_mask


def get_filled_contour_mask(binary_image):

    contours, hierarchy = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros_like(binary_image)

    contour_mask = np.zeros_like(binary_image)

    # Select the largest contour as the traffic sign
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    hull = cv2.convexHull(largest_contour)

    cv2.drawContours(
        contour_mask,
        [hull],  # cv2.drawContours() expects a collection of contours
        -1,
        255,
        thickness=cv2.FILLED  # thickness: fill the contour
    )

    return contour_mask


def segment_blue(img, debug=False):

    blue_mask, filtered, final_mask = enhance_img(img)

    contour_mask = get_filled_contour_mask(final_mask)

    if debug:
        debug_images = {
            "filtered": filtered,
            "threshold": final_mask,
            "filled": contour_mask
        }
        return contour_mask, debug_images

    return contour_mask


# ---------------------------------------------------------------------
# Yellow segmentation
# ---------------------------------------------------------------------


def clean_dark_image(
    image,
    brightness_threshold=140,
    clahe_clip_limit=2.0,
    clahe_grid_size=(8, 8)
):
    cleaned_image = image.copy()

    hsv_image = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    height, width = v_channel.shape

    # Check the brightness near the image centre
    centre_region = v_channel[
        height // 4: 3 * height // 4,
        width // 4: 3 * width // 4
    ]

    mean_brightness = float(np.mean(centre_region))

    # Enhance dark images using CLAHE
    if mean_brightness < brightness_threshold:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size
        )

        enhanced_v = clahe.apply(v_channel)
        enhanced_hsv = cv2.merge(
            (h_channel, s_channel, enhanced_v)
        )

        cleaned_image = cv2.cvtColor(
            enhanced_hsv,
            cv2.COLOR_HSV2BGR
        )

    return cleaned_image


def get_valid_block_size(
    image,
    preferred_size=51
):
    block_size = min(
        preferred_size,
        min(image.shape[:2])
    )

    if block_size % 2 == 0:
        block_size -= 1

    if block_size < 3:
        raise ValueError(
            "The image is too small for adaptive thresholding."
        )

    return block_size


def create_combined_threshold(
    cleaned_image,
    lower_yellow=(5, 80, 30),
    upper_yellow=(40, 255, 255),
    adaptive_block_size=51,
    adaptive_c=7
):
    hsv_image = cv2.cvtColor(
        cleaned_image,
        cv2.COLOR_BGR2HSV
    )

    _, _, v_channel = cv2.split(hsv_image)

    lower_bound = np.array(
        lower_yellow,
        dtype=np.uint8
    )

    upper_bound = np.array(
        upper_yellow,
        dtype=np.uint8
    )

    # Extract yellow pixels using the HSV range
    yellow_mask = cv2.inRange(
        hsv_image,
        lower_bound,
        upper_bound
    )

    block_size = get_valid_block_size(
        v_channel,
        adaptive_block_size
    )

    # Apply adaptive thresholding for uneven lighting
    adaptive_image = cv2.adaptiveThreshold(
        v_channel,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        adaptive_c
    )

    # Combine the colour and threshold masks
    combined_threshold = cv2.bitwise_and(
        yellow_mask,
        adaptive_image
    )

    return combined_threshold


def apply_morphological_closing(
    combined_threshold,
    kernel_size=(3, 3),
    iterations=1
):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        kernel_size
    )

    # Connect broken regions and fill small gaps
    morphological_image = cv2.morphologyEx(
        combined_threshold,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations
    )

    return morphological_image


def select_best_contour(
    morphological_image,
    min_area_ratio=0.005,
    max_area_ratio=0.75
):
    height, width = morphological_image.shape
    image_area = height * width

    image_centre_x = width / 2
    image_centre_y = height / 2

    # Find the outer contours
    contours, _ = cv2.findContours(
        morphological_image.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_score = -1

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        # Remove contours that are too small or too large
        if not (
            image_area * min_area_ratio
            <= contour_area
            <= image_area * max_area_ratio
        ):
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        contour_centre_x = (
            moments["m10"] / moments["m00"]
        )

        contour_centre_y = (
            moments["m01"] / moments["m00"]
        )

        centre_distance = np.hypot(
            contour_centre_x - image_centre_x,
            contour_centre_y - image_centre_y
        )

        # Prefer a large contour near the image centre
        contour_score = (
            contour_area / (centre_distance + 1)
        )

        if contour_score > best_score:
            best_contour = contour
            best_score = contour_score

    return best_contour


def reconstruct_outer_contour(
    selected_contour,
    solidity_threshold=0.90,
    approximation_ratio=0.01
):
    if selected_contour is None:
        return None

    contour_area = cv2.contourArea(
        selected_contour
    )

    # Create a convex hull around the contour
    convex_hull = cv2.convexHull(
        selected_contour
    )

    hull_area = cv2.contourArea(
        convex_hull
    )

    if hull_area == 0:
        return selected_contour

    solidity = contour_area / hull_area

    # Reconstruct the boundary if the contour is incomplete
    if solidity < solidity_threshold:
        hull_perimeter = cv2.arcLength(
            convex_hull,
            True
        )

        return cv2.approxPolyDP(
            convex_hull,
            approximation_ratio * hull_perimeter,
            True
        )

    return selected_contour


def create_contour_outputs(
    original_image,
    selected_contour
):
    # Create an image for displaying the green contour
    contour_only_image = np.zeros_like(
        original_image
    )

    # Create the filled contour mask
    contour_mask = np.zeros(
        original_image.shape[:2],
        dtype=np.uint8
    )

    # Return empty results if no contour is found
    if selected_contour is None:

        segmented_original = np.zeros_like(
            original_image
        )

        return (
            contour_only_image,
            contour_mask,
            segmented_original
        )

    # Reconstruct the outer boundary
    outer_contour = reconstruct_outer_contour(
        selected_contour
    )

    # Draw the detected contour
    cv2.drawContours(
        contour_only_image,
        [outer_contour],
        -1,
        (0, 255, 0),
        2
    )

    # Fill the detected contour
    cv2.drawContours(
        contour_mask,
        [outer_contour],
        -1,
        255,
        cv2.FILLED
    )

    # Extract the traffic-sign region
    segmented_original = cv2.bitwise_and(
        original_image,
        original_image,
        mask=contour_mask
    )

    return (
        contour_only_image,
        contour_mask,
        segmented_original
    )


def segment_yellow(img, debug=False):

    cleaned = clean_dark_image(img)

    threshold = create_combined_threshold(cleaned)

    morph = apply_morphological_closing(threshold)

    contour = select_best_contour(morph)

    _, contour_mask, _ = create_contour_outputs(
        img,
        contour
    )

    if debug:
        debug_images = {
            "filtered": cleaned,
            "threshold": threshold,
            "filled": contour_mask
        }
        return contour_mask, debug_images

    return contour_mask


# ---------------------------------------------------------------------
# Shape feature extraction
#
# Moved here from pipeline/shapes.py (which now imports these back) because
# find_best_candidate below needs shape_score to compare red/blue/yellow
# candidates, and shapes.py already depended on this module - keeping the
# shared logic here avoids a circular import.
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# Multi-colour candidate scoring and contour selection
#
# Ported from @kahyikang's 03_automatic_colour_segmentation_for_dataset.ipynb.
# This module used to OR red|blue|yellow masks together and score contours
# found on that single merged mask, using only area x circularity x
# solidity, with a MIN_AREA of 3.0 pixels that barely filters anything.
# This instead finds each colour's own best candidate contour independently
# (on that colour's own cleaned mask, before it gets merged with the
# others) and only then picks a winner across colours - so two
# different-coloured blobs sitting next to each other in the same image
# can no longer merge into one bad contour. The score also adds two checks
# the merged-mask approach never made: hue_agreement_score (do the winning
# contour's actual pixels agree with the colour credited for it?) and
# colour_coverage (how much of the filled contour's interior is really
# that colour, versus background swept in by a loose mask?).
# ---------------------------------------------------------------------


def get_red_stage_masks(img):
    """Return (colour_mask, cleaned_mask) for red - the same masks
    segment_red computes internally, before it collapses them down to a
    single largest-component result.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blurred = apply_gaussian_blur(img)
    diff = get_red_diff_map(blurred)
    colour_mask = adaptive_color_threshold(diff)
    cleaned_mask = clean_morphology(colour_mask)
    return colour_mask, cleaned_mask


def get_blue_stage_masks(img):
    """Return (colour_mask, cleaned_mask) for blue - the same masks
    segment_blue computes internally, before it collapses them down to a
    single largest-contour result.
    """
    colour_mask, _, cleaned_mask = enhance_img(img)
    return colour_mask, cleaned_mask


def get_yellow_stage_masks(img):
    """Return (colour_mask, cleaned_mask) for yellow - the same masks
    segment_yellow computes internally, before it collapses them down to a
    single best-contour result.
    """
    cleaned_image = clean_dark_image(img)
    colour_mask = create_combined_threshold(cleaned_image)
    cleaned_mask = apply_morphological_closing(colour_mask)
    return colour_mask, cleaned_mask


COLOUR_STAGE_FUNCTIONS = {
    "red": get_red_stage_masks,
    "blue": get_blue_stage_masks,
    "yellow": get_yellow_stage_masks,
}


def hue_agreement_score(image_bgr, contour, colour):
    """Fraction-weighted agreement between the contour's pixel hues and the
    expected hue for `colour` ("red", "blue" or "yellow"). Pixels with low
    saturation/value are excluded, since hue is unreliable there (white,
    black and glare pixels carry no usable colour information).
    """
    contour_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, cv2.FILLED)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    valid = (contour_mask > 0) & (saturation >= 45) & (value >= 30)

    if not np.any(valid):
        return 0.0

    selected_hue = hue[valid]
    if colour == "red":
        distance = np.minimum(selected_hue, 180.0 - selected_hue)
        affinity = np.clip(1.0 - distance / 18.0, 0.0, 1.0)
    elif colour == "yellow":
        affinity = np.clip(1.0 - np.abs(selected_hue - 25.0) / 20.0, 0.0, 1.0)
    else:
        affinity = np.clip(1.0 - np.abs(selected_hue - 112.0) / 25.0, 0.0, 1.0)

    return float(np.mean(affinity))


def find_best_candidate(image_bgr, colour, colour_mask, cleaned_mask):
    """Score every contour on `cleaned_mask` and return the best one for
    this single colour (as {"colour", "contour", "score"}), or None.
    Candidates are filtered by a mask-relative area ratio - rather than a
    fixed MIN_AREA=3.0, which barely filters anything - and scored on
    area, distance from the image centre, solidity,
    circularity, whether the outline resembles a real sign shape,
    colour_coverage and hue_agreement_score.
    """
    height, width = cleaned_mask.shape
    image_area = float(height * width)
    image_diagonal = float(np.hypot(width, height))

    contours, _ = cv2.findContours(
        cleaned_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None

    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        if area < max(20.0, image_area * 0.002) or area_ratio > 0.85:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        centre_x = moments["m10"] / moments["m00"]
        centre_y = moments["m01"] / moments["m00"]
        centre_distance = np.hypot(centre_x - width / 2, centre_y - height / 2)

        solidity = float(area / hull_area)
        circularity = float(np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0, 1))
        area_score = float(min(area_ratio / 0.25, 1.0))
        centre_score = float(max(0.0, 1.0 - centre_distance / (0.5 * image_diagonal + 1e-6)))

        shape_features = extract_features(contour)
        shape = classify_shape(shape_features) if shape_features else "Unknown"
        shape_score = 1.0 if shape != "Unknown" else 0.25

        filled = np.zeros_like(cleaned_mask)
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
        filled_pixels = max(cv2.countNonZero(filled), 1)
        colour_pixels = cv2.countNonZero(cv2.bitwise_and(colour_mask, filled))
        colour_coverage = float(colour_pixels / filled_pixels)
        hue_score = hue_agreement_score(image_bgr, contour, colour)

        score = (
            0.25 * area_score
            + 0.17 * centre_score
            + 0.13 * solidity
            + 0.10 * circularity
            + 0.15 * shape_score
            + 0.08 * min(colour_coverage / 0.35, 1.0)
            + 0.12 * hue_score
        )

        if best is None or score > best["score"]:
            best = {"colour": colour, "contour": contour, "score": score}

    return best


def select_best_contour_multi_colour(image_bgr):
    """Pick one best traffic-sign contour by scoring red/blue/yellow
    candidates separately (find_best_candidate) and comparing across
    colours - instead of OR-ing their masks into one first and scoring
    contours found on that merged mask. The winner is passed through
    reconstruct_outer_contour to repair a broken/partial outline before
    being returned. Returns None if no colour produced a usable candidate.
    """
    candidates = []
    for colour, stage_function in COLOUR_STAGE_FUNCTIONS.items():
        colour_mask, cleaned_mask = stage_function(image_bgr)
        candidate = find_best_candidate(image_bgr, colour, colour_mask, cleaned_mask)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return None

    winner = max(candidates, key=lambda item: item["score"])
    return reconstruct_outer_contour(winner["contour"])

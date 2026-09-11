#!/usr/bin/env python
# coding: utf-8

# # Traffic-Sign Recognition Using SVM and HOG
# 
# **Support Vector Machine (SVM)**.
# 
# 1. Load the clean Chinese traffic-sign HOG training and public-test features.
# 2. Create a stratified training/validation split.
# 3. Compare several SVM parameter combinations using validation macro F1-score and accuracy.
# 4. Select the best SVM parameters automatically.
# 5. Retrain the selected SVM using the complete training set.
# 6. Display 10 training results from the selected best SVM.
# 7. Evaluate the final SVM once on the public dataset test set.
# 8. Display 10 testing results from the selected best SVM.
# 9. Load the already-extracted HOG features for the 84 final testing images and classify them without extracting HOG again.
# 10. Use the raw testing images only for segmentation and result visualisation.
# 11. Display all final testing results at the end of the notebook.
# 

# ## 1. Import libraries

# In[1]:


from pathlib import Path
from time import perf_counter
import json
import re
import warnings

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    train_test_split,
    ParameterSampler
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
EXPECTED_HOG_FEATURES = 1764
MAX_SECONDS_PER_IMAGE = 2.0

np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid")


# ## 2. Set project paths
# 

# In[2]:


# Place this notebook inside the project folder with the other notebooks.
DATA_DIR = Path("../data")
DATASET_DIR = DATA_DIR / "chinese_traffic_signs"
INPUT_DIR = DATA_DIR / "Inputs"
FEATURE_DIR = Path("../features")
RESULT_DIR = Path("../results")
MODEL_DIR = Path("../models")

LABEL_FILE = DATASET_DIR / "labels.csv"
TRAIN_FEATURE_FILE = FEATURE_DIR / "chinese_traffic_signs_hog_features_train.npz"
TEST_FEATURE_FILE = FEATURE_DIR / "chinese_traffic_signs_hog_features_test.npz"
INPUT_TEST_FEATURE_FILE = FEATURE_DIR / "assignment1_test_hog_features.npz"

SVM_VALIDATION_FILE = RESULT_DIR / "svm_parameter_validation_results.csv"
FINAL_MODEL_FILE = MODEL_DIR / "svm_hog_final_classifier.joblib"
FINAL_METADATA_FILE = MODEL_DIR / "svm_hog_final_classifier_metadata.json"
FINAL_RESULTS_FILE = RESULT_DIR / "svm_final_84_image_results.csv"
FINAL_DISPLAY_DIR = RESULT_DIR / "svm_final_detection_displays"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

required_paths = [
    DATASET_DIR,
    INPUT_DIR,
    LABEL_FILE,
    TRAIN_FEATURE_FILE,
    TEST_FEATURE_FILE,
    INPUT_TEST_FEATURE_FILE,
]
missing_paths = [path for path in required_paths if not path.exists()]

if missing_paths:
    missing_text = "\n".join(f"- {path.resolve()}" for path in missing_paths)
    raise FileNotFoundError(
        "The following required files or folders were not found:\n"
        f"{missing_text}\n\n"
        "Run the cleanup and HOG extraction notebooks first."
    )

print("All required project files were found.")


# ## 3. Load class names

# In[3]:


labels_df = pd.read_csv(LABEL_FILE)

if not {"ClassId", "Name"}.issubset(labels_df.columns):
    raise ValueError("labels.csv must contain ClassId and Name columns.")

label_map = dict(
    zip(
        labels_df["ClassId"].astype(int),
        labels_df["Name"].astype(str),
    )
)

display(labels_df.head(10))
print("Number of class names:", len(label_map))


# ## 4. Load the clean HOG training and public-test features

# In[4]:


train_data = np.load(TRAIN_FEATURE_FILE, allow_pickle=False)
test_data = np.load(TEST_FEATURE_FILE, allow_pickle=False)

X_train_full = train_data["X"].astype(np.float32, copy=False)
y_train_full = train_data["y"].astype(np.int64, copy=False)
train_filenames = train_data["filenames"].astype(str)

X_test = test_data["X"].astype(np.float32, copy=False)
y_test = test_data["y"].astype(np.int64, copy=False)
test_filenames = test_data["filenames"].astype(str)

if X_train_full.ndim != 2 or X_test.ndim != 2:
    raise ValueError("HOG feature matrices must be two-dimensional.")

if X_train_full.shape[1] != EXPECTED_HOG_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_HOG_FEATURES} HOG features, "
        f"but received {X_train_full.shape[1]}."
    )

if X_test.shape[1] != EXPECTED_HOG_FEATURES:
    raise ValueError("Public-test HOG dimensions do not match training dimensions.")

if not np.isfinite(X_train_full).all() or not np.isfinite(X_test).all():
    raise ValueError("NaN or infinite HOG values were found.")

print("Training features:", X_train_full.shape)
print("Public-test features:", X_test.shape)
print("Classes:", len(np.unique(y_train_full)))


# ## 5. Confirm that testing filenames are absent from model-development data

# In[5]:


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}

all_input_image_paths = sorted(
    path
    for path in INPUT_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)

input_filename_keys = {path.name.casefold() for path in all_input_image_paths}
train_filename_keys = {name.casefold() for name in train_filenames}
test_filename_keys = {name.casefold() for name in test_filenames}

testing_train_overlap = sorted(input_filename_keys & train_filename_keys)
testing_public_test_overlap = sorted(input_filename_keys & test_filename_keys)

print("Raw testing images:", len(all_input_image_paths))
print("Testing/train filename overlap:", len(testing_train_overlap))
print("Testing/public-test filename overlap:", len(testing_public_test_overlap))

if testing_train_overlap or testing_public_test_overlap:
    raise RuntimeError(
        "Data leakage detected. Testing input filenames are still present in "
        "the saved model-development features. Rerun the cleanup notebook and "
        "then regenerate the HOG feature files."
    )


# ## 6. Define the red, blue and yellow segmentation functions

# In[6]:


# Function to load an image and convert it from BGR to RGB
def load_image(filepath):
    img = cv2.imread(filepath)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Function to apply Gaussian Blur for noise reduction (using granular 3x3 kernel)
def apply_gaussian_blur(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Function to generate a normalized Red difference map using HSV color space
def get_red_diff_map(image):
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    H = hsv[:,:,0].astype(np.float32)
    S = hsv[:,:,1].astype(np.float32)

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

# Function to perform adaptive thresholding (Otsu's method) on the difference map
def adaptive_color_threshold(diff_map):
    # Otsu's thresholding dynamically finds the best threshold value
    ret, mask = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

# Function to apply Morphological Cleaning (Opening first to detach noise, Closing to fill gaps)
def clean_morphology(mask):
    # Use elliptical kernels for natural rounding around circular traffic signs
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Opening FIRST: Erosion followed by Dilation. This breaks thin bridges connecting background noise to the sign.
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    # Closing NEXT: Dilation followed by Erosion. An elliptical 5x5 kernel smoothly bridges gaps in the sign ring.
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel)
    return closing

# Function to apply the cleaned mask to the original image, filling internal area without distortion
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
    """
    Proposal-based red colour segmentation
    used for the Chinese Traffic Signs images.

    Pipeline:
    BGR -> RGB -> Gaussian Blur
    -> Red Difference Map -> Otsu Threshold
    -> Morphological Cleaning -> Largest Component
    """

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # Step 1: Gaussian blur
    blurred = apply_gaussian_blur(
        img_rgb
    )

    # Step 2: Proposal red difference map
    diff = get_red_diff_map(
        blurred
    )

    # Step 3: Otsu adaptive threshold
    mask = adaptive_color_threshold(
        diff
    )

    # Step 4: Morphological cleaning
    # Use small kernels to preserve traffic-sign details.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    # Opening: remove small isolated noise
    opening = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    # Closing: connect fragmented sign regions
    cleaned = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # Step 5: Keep the largest connected component
    result = extract_and_mask(
        img_rgb,
        cleaned
    )

    # Convert result back to a binary mask
    gray = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2GRAY
    )

    _, final_mask = cv2.threshold(
        gray,
        1,
        255,
        cv2.THRESH_BINARY
    )

    if debug:
        debug_images = {
            "red_difference": diff,
            "otsu_mask": mask,
            "cleaned_mask": cleaned,
            "final_mask": final_mask
        }

        return final_mask, debug_images

    return final_mask

def remove_small_components(binary_mask, min_area_ratio):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8 #include 8 directions
    )

    cleaned_mask = np.zeros_like(binary_mask)

    image_area = binary_mask.shape[0] * binary_mask.shape[1]   #shape[0]=image height , shape[1]=image width
    min_area = image_area * min_area_ratio

    for label in range(1, num_labels):  # 0 is background

        area = stats[label, cv2.CC_STAT_AREA] #get the area of connected region

        if area >= min_area:
            cleaned_mask[labels == label] = 255

    return cleaned_mask

def enhance_img( image):

    image_cvt=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    filtered = cv2.bilateralFilter(
        image_cvt,
        d=7,
        sigmaColor=40,
        sigmaSpace=40
    )


    filtered_cvt = cv2.cvtColor(filtered,cv2.COLOR_RGB2HSV)



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
        (4,4)
    )

    final_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=1
    )



    return blue_mask , filtered, final_mask




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
        [hull],  #cv2.drawContours() expects a collection of contours
        -1,
        255,
        thickness=cv2.FILLED  #thickness: fill the contour
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

# Enhance dark images using CLAHE
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


# Set a valid block size for adaptive thresholding
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


# Create the combined yellow mask
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


# Clean the mask using morphological closing
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


# Select the best traffic-sign contour
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


# Reconstruct an incomplete outer contour
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


# Create the contour and segmentation outputs
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

def find_best_contour(mask):
    """
    Select the best traffic-sign contour from a binary mask.

    Used for the Chinese Traffic Signs dataset.
    Candidate contours are filtered by a small minimum
    area and scored using area, circularity and solidity.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    best_contour = None
    best_score = 0

    # Use a small
    # absolute minimum contour area.
    MIN_AREA = 3.0

    for contour in contours:

        area = cv2.contourArea(contour)

        # Reject extremely small noise
        if area < MIN_AREA:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        # Circularity
        circularity = (
            4 * np.pi * area
            / (perimeter ** 2)
        )

        # Convex hull
        hull = cv2.convexHull(contour)

        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        # Solidity
        solidity = area / hull_area

        # Proposal-based contour score
        score = (
            area
            * circularity
            * solidity
        )

        if score > best_score:

            best_score = score
            best_contour = contour

    return best_contour


# ## 7. Prepare segmented images for result visualisation
# 

# In[7]:


def prepare_segmented_display(image):
    # Segmentation is performed only for visualisation.
    # Classification features are loaded from saved HOG .npz files.
    image_rgb = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    red_mask = segment_red(image_bgr)
    blue_mask = segment_blue(image_bgr)
    yellow_mask = segment_yellow(image_bgr)

    combined_mask = cv2.bitwise_or(red_mask, blue_mask)
    combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)

    best_contour = find_best_contour(combined_mask)

    if best_contour is not None:
        x, y, width, height = cv2.boundingRect(best_contour)
        width = max(1, width)
        height = max(1, height)

        filled_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        cv2.drawContours(
            filled_mask,
            [best_contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )

        segmented_full = cv2.bitwise_and(
            image_rgb,
            image_rgb,
            mask=filled_mask,
        )
        segmented_roi = segmented_full[y:y + height, x:x + width]
        segmentation_found = True
        bounding_box = (x, y, width, height)

    else:
        segmented_roi = image_rgb
        segmentation_found = False
        bounding_box = None

    if segmented_roi.size == 0:
        segmented_roi = image_rgb
        segmentation_found = False
        bounding_box = None

    return {
        "original": image_rgb,
        "segmented": segmented_roi,
        "segmentation_found": segmentation_found,
        "bounding_box": bounding_box,
        "combined_mask": combined_mask,
    }


# ## 8. Prepare the best-SVM image-result visualisation
# 

# In[8]:


# Locate all original Chinese traffic-sign images.
dataset_image_paths = sorted(
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

dataset_image_lookup = {
    path.name.casefold(): path
    for path in dataset_image_paths
}


def prepare_best_svm_visuals(
    features,
    labels,
    filenames,
    number_of_results=10,
    random_state=42,
):
    # Select only entries whose original image can be found.
    available_indices = np.array([
        index
        for index, filename in enumerate(filenames)
        if Path(str(filename)).name.casefold() in dataset_image_lookup
    ])

    if len(available_indices) == 0:
        raise RuntimeError("No original dataset images could be matched to filenames.")

    number_to_display = min(number_of_results, len(available_indices))
    random_generator = np.random.default_rng(random_state)
    selected_indices = random_generator.choice(
        available_indices,
        size=number_to_display,
        replace=False,
    )

    selected_predictions = final_model.predict(features[selected_indices])
    visual_results = []

    for data_index, prediction in zip(selected_indices, selected_predictions):
        filename = Path(str(filenames[data_index])).name
        image_path = dataset_image_lookup[filename.casefold()]

        with Image.open(image_path) as image:
            extraction = prepare_segmented_display(
                image.convert("RGB")
            )

        actual_class = int(labels[data_index])
        predicted_class = int(prediction)

        visual_results.append({
            "filename": filename,
            "original": extraction["original"],
            "segmented": extraction["segmented"],
            "actual_class": actual_class,
            "actual_name": label_map.get(
                actual_class,
                f"Class {actual_class}",
            ),
            "predicted_class": predicted_class,
            "predicted_name": label_map.get(
                predicted_class,
                f"Class {predicted_class}",
            ),
            "correct": predicted_class == actual_class,
            "segmentation_found": extraction["segmentation_found"],
        })

    return visual_results


def display_best_svm_visuals(visual_results, figure_title):
    total_results = len(visual_results)

    if total_results == 0:
        raise RuntimeError("No recognition results are available.")

    fig, axes = plt.subplots(
        total_results,
        2,
        figsize=(6, 2.2 * total_results),
        squeeze=False,
        facecolor="white",
    )

    for row_index, result in enumerate(visual_results):
        axes[row_index, 0].imshow(result["original"])
        axes[row_index, 0].set_title(
            f"Original - {result['filename']}",
            color="black",
            fontsize=8,
        )
        axes[row_index, 0].set_facecolor("white")
        axes[row_index, 0].axis("off")

        status = "Correct" if result["correct"] else "Wrong"
        colour = "green" if result["correct"] else "red"

        axes[row_index, 1].imshow(result["segmented"])
        axes[row_index, 1].set_title(
            f"{result['predicted_name']}\n"
            f"Class {result['predicted_class']} - {status}",
            color=colour,
            fontsize=8,
        )
        axes[row_index, 1].set_facecolor("white")
        axes[row_index, 1].axis("off")

    fig.suptitle(
        figure_title,
        color="black",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


print("Dataset images available for visualisation:", len(dataset_image_lookup))


# ## 9. Create a validation split and define SVM candidates

# In[9]:


def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


# The public test set is not used for parameter selection.
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

# These candidates test linear and non-linear decision boundaries.
svm_search_space = [
    {
        "kernel": ["linear"],
        "C": [0.01, 0.1, 0.3, 1, 3, 10, 30, 100],
        "class_weight": [None, "balanced"]
    },
    {
        "kernel": ["rbf"],
        "C": [0.1, 0.3, 1, 3, 10, 30, 100],
        "gamma": [
            "scale",
            0.0001,
            0.0003,
            0.001,
            0.003,
            0.01
        ],
        "class_weight": [None, "balanced"]
    }
]

SVM_CANDIDATES = list(
    ParameterSampler(
        svm_search_space,
        n_iter=12,
        random_state=RANDOM_STATE
    )
)

print("Parameter-selection training samples:", len(X_train))
print("Validation samples:", len(X_validation))
print("SVM candidates:", len(SVM_CANDIDATES))


# ## 10. Tune SVM and retrain the best SVM

# In[10]:


svm_validation_rows = []

for candidate_number, parameters in enumerate(SVM_CANDIDATES, start=1):
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            SVC(
                probability=False,
                cache_size=512,
                random_state=RANDOM_STATE,
                **parameters,
            ),
        ),
    ])

    training_start = perf_counter()
    svm_pipeline.fit(X_train, y_train)
    training_seconds = perf_counter() - training_start

    prediction_start = perf_counter()
    validation_predictions = svm_pipeline.predict(X_validation)
    prediction_seconds = perf_counter() - prediction_start

    scores = calculate_metrics(y_validation, validation_predictions)

    svm_validation_rows.append({
        "Candidate": candidate_number,
        "Kernel": parameters["kernel"],
        "C": parameters["C"],
        "Gamma": parameters.get("gamma", "not used"),
        "Class Weight": str(parameters["class_weight"]),
        **scores,
        "Training Time (s)": training_seconds,
        "Prediction Time per Image (ms)": (
            prediction_seconds / len(X_validation)
        ) * 1000,
        "Parameters": str(parameters),
    })


# In[11]:


svm_validation_df = pd.DataFrame(svm_validation_rows).sort_values(
    by=["Accuracy", "Macro F1", "Prediction Time per Image (ms)"],
    ascending=[False, False, True],
).reset_index(drop=True)

display(svm_validation_df.round(4))
svm_validation_df.to_csv(SVM_VALIDATION_FILE, index=False)


# In[12]:


best_candidate_number = int(svm_validation_df.iloc[0]["Candidate"])
best_parameters = SVM_CANDIDATES[best_candidate_number - 1]
best_classifier_name = "SVM"

print("Selected classifier: SVM")
print("Selected parameters:", best_parameters)
print("Validation accuracy:", f"{svm_validation_df.iloc[0]['Accuracy']:.4f}")
print("Validation macro F1:", f"{svm_validation_df.iloc[0]['Macro F1']:.4f}")


# In[13]:


# StandardScaler is fitted inside the pipeline using training data only.
final_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        SVC(
            probability=False,
            cache_size=512,
            random_state=RANDOM_STATE,
            **best_parameters,
        ),
    ),
])

training_start = perf_counter()
final_model.fit(X_train_full, y_train_full)
final_training_seconds = perf_counter() - training_start

print("Final SVM training completed.")
print("Full training samples:", len(X_train_full))
print("Training time:", f"{final_training_seconds:.4f} seconds")


# ## 11. Confusion matrix of training

# In[14]:


# Predict the complete training dataset
y_train_pred = final_model.predict(X_train_full)

# Obtain all classes in the training dataset
training_classes = np.unique(y_train_full)

# Generate the training confusion matrix
training_cm = confusion_matrix(
    y_train_full,
    y_train_pred,
    labels=training_classes,
)

# Display the confusion matrix
plt.figure(figsize=(18, 12))

sns.heatmap(
    training_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=training_classes,
    yticklabels=training_classes,
)

plt.title("Best SVM Confusion Matrix - Training Images")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.tight_layout()
plt.show()


# ## Display 10 training results from the best SVM
# 

# In[15]:


best_svm_training_visuals = prepare_best_svm_visuals(
    features=X_train_full,
    labels=y_train_full,
    filenames=train_filenames,
    number_of_results=10,
    random_state=42,
)

display_best_svm_visuals(
    best_svm_training_visuals,
    "Best SVM - 10 Training Results",
)


# ## 12. Evaluate the final SVM on the public dataset test set

# In[16]:


public_test_start = perf_counter()
y_test_pred = final_model.predict(X_test)
public_test_seconds = perf_counter() - public_test_start
public_test_ms_per_image = (public_test_seconds / len(X_test)) * 1000

public_test_scores = calculate_metrics(y_test, y_test_pred)

print("Final SVM Public-Dataset Test Results")
print("-------------------------------------")
print("Parameters       :", best_parameters)
for metric_name, metric_value in public_test_scores.items():
    print(f"{metric_name:17s}: {metric_value:.4f}")
print(f"Prediction time   : {public_test_seconds:.4f} seconds")
print(f"Time per image    : {public_test_ms_per_image:.4f} ms")


# In[17]:


all_classes = np.unique(np.concatenate([y_train_full, y_test]))

public_report_df = pd.DataFrame(
    classification_report(
        y_test,
        y_test_pred,
        labels=all_classes,
        target_names=[label_map.get(int(c), f"Class {c}") for c in all_classes],
        output_dict=True,
        zero_division=0,
    )
).transpose()

display(public_report_df.round(4))
public_report_df.to_csv(RESULT_DIR / "svm_final_public_test_classification_report.csv")


# ## 13. Display 10 testing results from the best SVM
# 

# In[18]:


best_svm_testing_visuals = prepare_best_svm_visuals(
    features=X_test,
    labels=y_test,
    filenames=test_filenames,
    number_of_results=10,
    random_state=43,
)

display_best_svm_visuals(
    best_svm_testing_visuals,
    "Best SVM - 10 Testing Results",
)


# ## 14. Load the already-extracted HOG features for the 84 final testing images
# 

# In[19]:


input_test_data = np.load(
    INPUT_TEST_FEATURE_FILE,
    allow_pickle=False,
)

required_test_keys = {"X", "image_paths"}
missing_test_keys = required_test_keys - set(input_test_data.files)

if missing_test_keys:
    raise KeyError(
        "The final testing feature file is missing keys: "
        + ", ".join(sorted(missing_test_keys))
    )

X_input_test = input_test_data["X"].astype(
    np.float32,
    copy=False,
)
input_test_saved_paths = input_test_data["image_paths"].astype(str)

if X_input_test.ndim != 2:
    raise ValueError("Final testing HOG features must be two-dimensional.")

if X_input_test.shape[1] != EXPECTED_HOG_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_HOG_FEATURES} HOG features, "
        f"but received {X_input_test.shape[1]}."
    )

if len(X_input_test) != len(input_test_saved_paths):
    raise ValueError(
        "The numbers of final testing feature vectors and image paths differ."
    )

if not np.isfinite(X_input_test).all():
    raise ValueError("NaN or infinite values were found in final testing features.")

if len(X_input_test) != 84:
    warnings.warn(
        f"Expected 84 final testing samples, but loaded {len(X_input_test)}."
    )


def saved_filename(saved_path):
    # Handles both Windows and POSIX paths stored inside the .npz file.
    return str(saved_path).replace("\\", "/").rsplit("/", 1)[-1]


all_input_image_paths = sorted(
    path
    for path in INPUT_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)

input_image_lookup = {}
for image_path in all_input_image_paths:
    input_image_lookup.setdefault(
        image_path.name.casefold(),
        [],
    ).append(image_path)

input_image_paths = []
missing_input_images = []

for saved_path in input_test_saved_paths:
    filename = saved_filename(saved_path)
    matches = input_image_lookup.get(filename.casefold(), [])

    if len(matches) == 1:
        input_image_paths.append(matches[0])
    elif len(matches) == 0:
        missing_input_images.append(filename)
    else:
        raise RuntimeError(
            f"More than one raw testing image is named {filename}."
        )

if missing_input_images:
    raise FileNotFoundError(
        "Raw testing images could not be matched to these saved features:\n"
        + "\n".join(f"- {name}" for name in missing_input_images)
    )

input_image_paths = np.asarray(input_image_paths, dtype=object)

print("Final testing HOG features:", X_input_test.shape)
print("Saved testing image paths:", len(input_test_saved_paths))
print("Matched raw images for visualisation:", len(input_image_paths))
print("HOG is loaded from:", INPUT_TEST_FEATURE_FILE.resolve())


# ## 15. Classify the 84 saved testing HOG feature vectors
# 

# In[20]:


def get_true_class_from_filename(filename):
    match = re.match(r"^(\d{3})", Path(filename).stem)
    if match is None:
        return None
    return int(match.group(1))


recognition_records = []
recognition_visuals = []
failed_images = []

for feature_vector, saved_path, image_path in zip(
    X_input_test,
    input_test_saved_paths,
    input_image_paths,
):
    filename = saved_filename(saved_path)

    try:
        # Only SVM classification is timed, matching notebook 05.
        classification_start = perf_counter()
        predicted_class = int(
            final_model.predict(feature_vector.reshape(1, -1))[0]
        )
        classification_seconds = perf_counter() - classification_start

        # The raw image is segmented only to prepare the visual result.
        with Image.open(image_path) as input_image:
            display_data = prepare_segmented_display(
                input_image.convert("RGB")
            )

        actual_class = get_true_class_from_filename(filename)
        actual_name = (
            label_map.get(actual_class, f"Class {actual_class}")
            if actual_class is not None
            else "Unknown"
        )
        predicted_name = label_map.get(
            predicted_class,
            f"Class {predicted_class}",
        )
        correct = (
            predicted_class == actual_class
            if actual_class is not None
            else None
        )

        recognition_records.append({
            "Filename": filename,
            "Actual Class": actual_class,
            "Actual Sign": actual_name,
            "Predicted Class": predicted_class,
            "Predicted Sign": predicted_name,
            "Correct": correct,
            "Segmentation Found": display_data["segmentation_found"],
            "Bounding Box": str(display_data["bounding_box"]),
            "Classification Time (s)": classification_seconds,
            "Below 2 Seconds": classification_seconds < MAX_SECONDS_PER_IMAGE,
        })

        recognition_visuals.append({
            "filename": filename,
            "original": display_data["original"],
            "segmented": display_data["segmented"],
            "actual_class": actual_class,
            "actual_name": actual_name,
            "predicted_class": predicted_class,
            "predicted_name": predicted_name,
            "correct": correct,
            "segmentation_found": display_data["segmentation_found"],
            "classification_seconds": classification_seconds,
        })

    except Exception as error:
        failed_images.append({
            "Filename": filename,
            "Error": repr(error),
        })

recognition_results_df = pd.DataFrame(recognition_records)
failed_images_df = pd.DataFrame(failed_images)

display(recognition_results_df)

if not failed_images_df.empty:
    print("Images that could not be processed:")
    display(failed_images_df)

recognition_results_df.to_csv(FINAL_RESULTS_FILE, index=False)
print("Saved final result table:", FINAL_RESULTS_FILE.resolve())


# ## 16. Calculate final results for all 84 testing images
# 

# In[21]:


evaluable_results_df = recognition_results_df.dropna(
    subset=["Actual Class", "Predicted Class"]
).copy()

if evaluable_results_df.empty:
    raise RuntimeError("No testing images have usable actual class labels.")

y_testing_true = evaluable_results_df["Actual Class"].astype(int).to_numpy()
y_testing_pred = evaluable_results_df["Predicted Class"].astype(int).to_numpy()

testing_scores = calculate_metrics(
    y_testing_true,
    y_testing_pred,
)

correct_images = int(np.sum(y_testing_true == y_testing_pred))
average_time = recognition_results_df["Classification Time (s)"].mean()
maximum_time = recognition_results_df["Classification Time (s)"].max()
minimum_time = recognition_results_df["Classification Time (s)"].min()
over_time_limit = int((~recognition_results_df["Below 2 Seconds"]).sum())
segmentation_successes = int(recognition_results_df["Segmentation Found"].sum())

final_summary_df = pd.DataFrame([{
    "Classifier": best_classifier_name,
    "Parameters": str(best_parameters),
    "Correct Images": correct_images,
    "Evaluated Images": len(evaluable_results_df),
    **testing_scores,
    "Segmentation Successes": segmentation_successes,
    "Processed Images": len(recognition_results_df),
    "Failed Images": len(failed_images_df),
    "Average Classification Time (s)": average_time,
    "Maximum Classification Time (s)": maximum_time,
    "Minimum Classification Time (s)": minimum_time,
    "Images Over 2 Seconds": over_time_limit,
}])

display(final_summary_df.round(4))

print("Final 84-Image Testing Results")
print("------------------------------")
print("Classifier:", best_classifier_name)
print("Parameters:", best_parameters)
print(f"Correct: {correct_images}/{len(evaluable_results_df)}")
for metric_name, metric_value in testing_scores.items():
    print(f"{metric_name:17s}: {metric_value:.4f}")
print(f"Segmentation found: {segmentation_successes}/{len(recognition_results_df)}")
print(f"Average classification time: {average_time:.6f} seconds/image")
print(f"Maximum classification time: {maximum_time:.6f} seconds")
print(f"Minimum classification time: {minimum_time:.6f} seconds")
print(f"All images below 2 seconds: {over_time_limit == 0}")


# ## 17. Confusion matrix of testing

# In[22]:


testing_classes = np.unique(y_testing_true)
testing_cm = confusion_matrix(
    y_testing_true,
    y_testing_pred,
    labels=testing_classes,
)

plt.figure(figsize=(14, 12))
sns.heatmap(
    testing_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=testing_classes,
    yticklabels=testing_classes,
)
plt.title(f"{best_classifier_name} Confusion Matrix - Testing Images")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.tight_layout()
plt.show()


# ## 18. Save the final model and metadata

# In[23]:


joblib.dump(final_model, FINAL_MODEL_FILE)

metadata = {
    "classifier": "SVM",
    "parameters": best_parameters,
    "parameter_selection": "20% stratified validation split",
    "selection_metric": "Macro F1, followed by accuracy and prediction time",
    "hog_features": EXPECTED_HOG_FEATURES,
    "hog_configuration": {
        "image_size": [32, 32],
        "orientations": 9,
        "pixels_per_cell": [4, 4],
        "cells_per_block": [2, 2],
        "block_norm": "L2-Hys",
        "transform_sqrt": True,
        "channel_axis": -1,
    },
    "training_samples": int(len(X_train_full)),
    "public_test_metrics": {
        name: float(value)
        for name, value in public_test_scores.items()
    },
    "final_testing_feature_file": str(INPUT_TEST_FEATURE_FILE),
    "final_testing_samples": int(len(X_input_test)),
    "testing_metrics": {
        name: float(value)
        for name, value in testing_scores.items()
    },
}

with FINAL_METADATA_FILE.open("w", encoding="utf-8") as stream:
    json.dump(metadata, stream, indent=2, ensure_ascii=False)

print("Saved final SVM model:", FINAL_MODEL_FILE.resolve())
print("Saved SVM metadata:", FINAL_METADATA_FILE.resolve())
print("Saved SVM validation table:", SVM_VALIDATION_FILE.resolve())


# ## 19. Display all final testing results

# In[24]:


def display_all_results():
    total_results = len(recognition_visuals)

    if total_results == 0:
        raise RuntimeError("No recognition results are available.")

    fig, axes = plt.subplots(
        total_results,
        2,
        figsize=(6, 2.2 * total_results),
        squeeze=False,
        facecolor="white",
    )

    for row_index, result in enumerate(recognition_visuals):
        # Original image
        axes[row_index, 0].imshow(result["original"])
        axes[row_index, 0].set_title(
            f"Original - {result['filename']}",
            color="black",
            fontsize=8,
        )
        axes[row_index, 0].set_facecolor("white")
        axes[row_index, 0].axis("off")

        # Recognition status
        status = (
            "Correct" if result["correct"] is True
            else "Wrong" if result["correct"] is False
            else "Unknown"
        )

        colour = (
            "green" if status == "Correct"
            else "red" if status == "Wrong"
            else "darkorange"
        )

        # Segmented image and prediction
        axes[row_index, 1].imshow(result["segmented"])
        axes[row_index, 1].set_title(
            f"{result['predicted_name']}\n"
            f"Class {result['predicted_class']} - {status}",
            color=colour,
            fontsize=8,
        )
        axes[row_index, 1].set_facecolor("white")
        axes[row_index, 1].axis("off")

    plt.tight_layout()
    plt.show()


display_all_results()


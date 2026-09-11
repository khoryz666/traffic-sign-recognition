#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# In[4]:


DATA_DIR = Path("../data")

DATASET_DIR = (
    DATA_DIR
    / "chinese_traffic_signs"
)

LABEL_FILE = (
    DATASET_DIR
    / "labels.csv"
)

labels_df = pd.read_csv(
    LABEL_FILE
)

display(
    labels_df.head(10)
)


# In[5]:


label_map = dict(
    zip(
        labels_df["ClassId"].astype(int),
        labels_df["Name"]
    )
)

print(label_map)


# In[6]:


FEATURE_DIR = Path("../features")

TRAIN_FILE = (
    FEATURE_DIR
    / "chinese_traffic_signs_hog_features_train.npz"
)

TEST_FILE = (
    FEATURE_DIR
    / "chinese_traffic_signs_hog_features_test.npz"
)


train_data = np.load(
    TRAIN_FILE
)

test_data = np.load(
    TEST_FILE
)


X_train_full = train_data["X"]
y_train_full = train_data["y"]
train_filenames = train_data["filenames"]

X_test = test_data["X"]
y_test = test_data["y"]
test_filenames = test_data["filenames"]


print(
    "Training features:",
    X_train_full.shape
)

print(
    "Training labels:",
    y_train_full.shape
)

print(
    "Training filenames:",
    train_filenames.shape
)

print(
    "Testing features:",
    X_test.shape
)

print(
    "Testing labels:",
    y_test.shape
)

print(
    "Testing filenames:",
    test_filenames.shape
)


# In[7]:


print(
    "Number of training samples:",
    len(X_train_full)
)

print(
    "Number of testing samples:",
    len(X_test)
)

print(
    "Number of classes:",
    len(np.unique(y_train_full))
)

print(
    "Classes:",
    np.unique(y_train_full)
)

print(
    "HOG features per image:",
    X_train_full.shape[1]
)


# In[8]:


print(
    "NaN in training:",
    np.isnan(X_train_full).any()
)

print(
    "NaN in testing:",
    np.isnan(X_test).any()
)

print(
    "Infinity in training:",
    np.isinf(X_train_full).any()
)

print(
    "Infinity in testing:",
    np.isinf(X_test).any()
)


# In[9]:


X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

print(
    "Training set:",
    X_train.shape
)

print(
    "Validation set:",
    X_val.shape
)

print(
    "Final test set:",
    X_test.shape
)


# In[10]:


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    "Feature scaling completed."
)


# In[11]:


C_values = [
    0.01,
    0.1,
    1,
    10,
    100
]

results = []

for C in C_values:

    logistic_model = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=5000,
        random_state=42
    )

    logistic_model.fit(
        X_train_scaled,
        y_train
    )

    y_val_pred = logistic_model.predict(
        X_val_scaled
    )

    accuracy = accuracy_score(
        y_val,
        y_val_pred
    )

    precision = precision_score(
        y_val,
        y_val_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        average="macro",
        zero_division=0
    )

    results.append({
        "C": C,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })


results_df = pd.DataFrame(
    results
)

display(
    results_df
)


# In[12]:


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    results_df["C"],
    results_df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    results_df["C"],
    results_df["F1-score"],
    marker="o",
    label="Macro F1-score"
)

plt.xscale("log")

plt.xlabel(
    "Regularization Parameter (C)"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Logistic Regression Performance "
    "for Different C Values"
)

plt.xticks(
    C_values,
    C_values
)

plt.legend()

plt.grid()

plt.show()


# In[13]:


best_row = results_df.loc[
    results_df["F1-score"].idxmax()
]

best_C = float(
    best_row["C"]
)

print(
    "Best C:",
    best_C
)

print(
    "Validation Accuracy:",
    round(
        best_row["Accuracy"],
        4
    )
)

print(
    "Validation Macro F1:",
    round(
        best_row["F1-score"],
        4
    )
)


# In[14]:


final_scaler = StandardScaler()

X_train_full_scaled = (
    final_scaler.fit_transform(
        X_train_full
    )
)

X_test_final_scaled = (
    final_scaler.transform(
        X_test
    )
)


# In[15]:


final_logistic = LogisticRegression(
    C=best_C,
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)

final_logistic.fit(
    X_train_full_scaled,
    y_train_full
)

y_train_pred = final_logistic.predict(
    X_train_full_scaled
)

print(
    f"Final Logistic Regression trained "
    f"with C = {best_C}"
)


# In[16]:


from PIL import Image

DATASET_DIR = Path(
    "../data/chinese_traffic_signs"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".ppm"
}

all_images = [
    p for p in DATASET_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower()
    in IMAGE_EXTENSIONS
]

image_lookup = {
    p.name: p
    for p in all_images
}

print(
    "Images found:",
    len(image_lookup)
)


# In[17]:


sample_count = min(
    10,
    len(y_train_full)
)

sample_indices = np.linspace(
    0,
    len(y_train_full) - 1,
    sample_count,
    dtype=int
)

fig, axes = plt.subplots(
    2,
    5,
    figsize=(20, 7)
)

axes = axes.flatten()


for ax, index in zip(
    axes,
    sample_indices
):

    filename = str(
        train_filenames[index]
    )

    actual = int(
        y_train_full[index]
    )

    predicted = int(
        y_train_pred[index]
    )

    image_path = image_lookup.get(
        filename
    )

    if image_path is not None:

        image = Image.open(
            image_path
        ).convert("RGB")

        ax.imshow(
            image
        )

    correct = (
        actual == predicted
    )

    ax.set_title(
        f"Actual: {label_map[actual]}\n"
        f"Predicted: {label_map[predicted]}\n"
        f"{'Correct' if correct else 'Wrong'}"
    )

    ax.axis("off")


plt.suptitle(
    "Logistic Regression "
    "Training Prediction Results",
    fontsize=16
)

plt.tight_layout()

plt.show()


# In[18]:


y_test_pred = final_logistic.predict(
    X_test_final_scaled
)


# In[19]:


test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_precision = precision_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)


print(
    "Logistic Regression Final Test Results"
)

print(
    "--------------------------------------"
)

print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1-score  : {test_f1:.4f}"
)


# In[20]:


cm = confusion_matrix(
    y_test,
    y_test_pred
)

fig, ax = plt.subplots(
    figsize=(16, 16)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False
)

plt.title(
    "Logistic Regression "
    "Confusion Matrix"
)

plt.tight_layout()

plt.show()


# In[21]:


sample_count = min(
    10,
    len(y_test)
)

sample_indices = np.linspace(
    0,
    len(y_test) - 1,
    sample_count,
    dtype=int
)

prediction_results = []

for index in sample_indices:

    actual = int(
        y_test[index]
    )

    predicted = int(
        y_test_pred[index]
    )

    prediction_results.append({
        "Test Index": index,
        "Actual Class": actual,
        "Predicted Class": predicted,
        "Correct":
            actual == predicted
    })


prediction_df = pd.DataFrame(
    prediction_results
)

display(
    prediction_df
)


# In[22]:


sample_count = min(
    10,
    len(y_test)
)

sample_indices = np.linspace(
    0,
    len(y_test) - 1,
    sample_count,
    dtype=int
)

fig, axes = plt.subplots(
    2,
    5,
    figsize=(20, 7)
)

axes = axes.flatten()


for ax, index in zip(
    axes,
    sample_indices
):

    filename = str(
        test_filenames[index]
    )

    actual = int(
        y_test[index]
    )

    predicted = int(
        y_test_pred[index]
    )

    image_path = image_lookup.get(
        filename
    )

    if image_path is not None:

        image = Image.open(
            image_path
        ).convert("RGB")

        ax.imshow(
            image
        )

    correct = (
        actual == predicted
    )

    ax.set_title(
        f"Actual: {label_map[actual]}\n"
        f"Predicted: {label_map[predicted]}\n"
        f"{'Correct' if correct else 'Wrong'}"
    )

    ax.axis("off")


plt.suptitle(
    "Logistic Regression Test "
    "Prediction Results",
    fontsize=16
)

plt.tight_layout()

plt.show()


# In[23]:


#def function to provide the segemented image with black background 
#and def function to display the result


# In[24]:


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


# In[25]:


def get_segmented_image(image_path):

    # Read original image
    image_bgr = cv2.imread(str(image_path))

    # Create red, blue and yellow masks
    red_mask = segment_red(image_bgr)
    blue_mask = segment_blue(image_bgr)
    yellow_mask = segment_yellow(image_bgr)

    # Combine masks
    combined_mask = cv2.bitwise_or(red_mask, blue_mask)
    combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)

    # Find traffic-sign contour
    contour = find_best_contour(combined_mask)

    # Empty mask
    filled_mask = np.zeros(
        image_bgr.shape[:2],
        dtype=np.uint8
    )

    if contour is not None:

        contour = reconstruct_outer_contour(contour)

        cv2.drawContours(
            filled_mask,
            [contour],
            -1,
            255,
            cv2.FILLED
        )

    # Black background outside sign
    segmented_bgr = cv2.bitwise_and(
        image_bgr,
        image_bgr,
        mask=filled_mask
    )

    # Convert for matplotlib
    original_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    segmented_rgb = cv2.cvtColor(
        segmented_bgr,
        cv2.COLOR_BGR2RGB
    )

    return original_rgb, segmented_rgb


# In[26]:


def display_prediction_results(
    filenames,
    y_true,
    y_pred,
    image_lookup,
    title
):

    sample_indices = np.linspace(
        0,
        len(filenames) - 1,
        10,
        dtype=int
    )

    fig, axes = plt.subplots(
        10,
        2,
        figsize=(10, 36)
    )

    for row, index in enumerate(sample_indices):

        filename = Path(
            str(filenames[index])
        ).name

        image_path = image_lookup.get(
            filename
        )

        actual = int(y_true[index])
        predicted = int(y_pred[index])

        if image_path is None:
            continue

        original, segmented = get_segmented_image(
            image_path
        )

        # Original image
        axes[row, 0].imshow(original)

        axes[row, 0].set_title(
            f"Original - {filename}"
        )

        axes[row, 0].axis("off")

        # Segmented image
        axes[row, 1].imshow(segmented)

        axes[row, 1].set_title(
            f"Actual: {label_map[actual]}\n"
            f"Predicted: {label_map[predicted]}\n"
            f"{'Correct' if actual == predicted else 'Wrong'}"
        )

        axes[row, 1].axis("off")

    plt.suptitle(
        title,
        fontsize=16,
        y=0.995
    )

    plt.subplots_adjust(
        top=0.96,
        bottom=0.02,
        hspace=0.65,
        wspace=0.25
    )

    plt.show()


# In[27]:


#Test on assignment 1 inputs


# In[28]:


assignment_data = np.load(
    "../features/"
    "assignment1_test_hog_features.npz"
)

X_assignment_test = (
    assignment_data["X"]
)

assignment_image_paths = (
    assignment_data["image_paths"]
)


print(
    "Assignment test features:",
    X_assignment_test.shape
)


# In[29]:


X_assignment_test_scaled = (
    final_scaler.transform(
        X_assignment_test
    )
)

print(
    "Scaled Assignment test shape:",
    X_assignment_test_scaled.shape
)


# In[30]:


import time

y_assignment_pred = []
classification_times = []

for i in range(
    len(X_assignment_test_scaled)
):

    # One image feature vector
    X_one = X_assignment_test_scaled[
        i:i + 1
    ]

    # Start timer
    start_time = time.perf_counter()

    # KNN classification
    prediction = final_logistic.predict(
        X_one
    )[0]

    # End timer
    end_time = time.perf_counter()

    runtime = (
        end_time - start_time
    )

    y_assignment_pred.append(
        prediction
    )

    classification_times.append(
        runtime
    )


# Convert to NumPy array
y_assignment_pred = np.array(
    y_assignment_pred
)

classification_times = np.array(
    classification_times
)



# In[31]:


print(
    "KNN Classification Runtime"
)

print(
    "--------------------------"
)

print(
    f"Average time per image : "
    f"{classification_times.mean():.6f} seconds"
)

print(
    f"Maximum time per image : "
    f"{classification_times.max():.6f} seconds"
)

print(
    f"Minimum time per image : "
    f"{classification_times.min():.6f} seconds"
)

print(
    f"All images below 2 sec : "
    f"{np.all(classification_times < 2)}"
)


# In[32]:


print(
    "Number of predictions:",
    len(y_assignment_pred)
)

print(
    "First 10 predictions:",
    y_assignment_pred[:10]
)


# In[33]:


def get_true_label_from_path(
    image_path
):

    filename = Path(
        str(image_path)
    ).name

    class_id = (
        Path(filename)
        .stem
        .split("_")[0]
    )

    return int(
        class_id
    )


# In[34]:


y_assignment_test = np.array(
    [
        get_true_label_from_path(
            path
        )
        for path
        in assignment_image_paths
    ]
)

print(
    "Actual labels:",
    y_assignment_test.shape
)

print(
    "First 10 actual labels:",
    y_assignment_test[:10]
)


# In[35]:


assignment_accuracy = accuracy_score(
    y_assignment_test,
    y_assignment_pred
)

assignment_precision = precision_score(
    y_assignment_test,
    y_assignment_pred,
    average="macro",
    zero_division=0
)

assignment_recall = recall_score(
    y_assignment_test,
    y_assignment_pred,
    average="macro",
    zero_division=0
)

assignment_f1 = f1_score(
    y_assignment_test,
    y_assignment_pred,
    average="macro",
    zero_division=0
)


print(
    "Logistic Regression Results "
    "on 84 Assignment Test Images"
)

print(
    "-----------------------------------------"
)

print(
    f"Accuracy  : "
    f"{assignment_accuracy:.4f}"
)

print(
    f"Precision : "
    f"{assignment_precision:.4f}"
)

print(
    f"Recall    : "
    f"{assignment_recall:.4f}"
)

print(
    f"F1-score  : "
    f"{assignment_f1:.4f}"
)


# In[36]:


assignment_results = []

for image_path, actual, predicted in zip(
    assignment_image_paths,
    y_assignment_test,
    y_assignment_pred
):

    actual = int(
        actual
    )

    predicted = int(
        predicted
    )

    assignment_results.append({

        "Filename":
            Path(
                str(image_path)
            ).name,

        "Actual Class":
            actual,

        "Actual Sign":
            label_map.get(
                actual,
                f"Class {actual}"
            ),

        "Predicted Class":
            predicted,

        "Predicted Sign":
            label_map.get(
                predicted,
                f"Class {predicted}"
            ),

        "Correct":
            actual == predicted
    })


assignment_results_df = pd.DataFrame(
    assignment_results
)

display(
    assignment_results_df
)


# In[37]:


INPUT_DIR = Path(
    "../data/Inputs"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".ppm"
}

assignment_images = [
    p for p in INPUT_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower()
    in IMAGE_EXTENSIONS
]

assignment_image_lookup = {
    p.name: p
    for p in assignment_images
}

print(
    "Assignment images found:",
    len(assignment_images)
)


# In[38]:


sample_count = min(
    10,
    len(assignment_image_paths)
)

sample_indices = np.linspace(
    0,
    len(assignment_image_paths) - 1,
    sample_count,
    dtype=int
)

fig, axes = plt.subplots(
    2,
    5,
    figsize=(20, 7)
)

axes = axes.flatten()


for ax, index in zip(
    axes,
    sample_indices
):

    filename = Path(
        str(
            assignment_image_paths[index]
        )
    ).name

    actual = int(
        y_assignment_test[index]
    )

    predicted = int(
        y_assignment_pred[index]
    )

    image_path = (
        assignment_image_lookup.get(
            filename
        )
    )

    if image_path is not None:

        image = Image.open(
            image_path
        ).convert("RGB")

        ax.imshow(
            image
        )

    correct = (
        actual == predicted
    )

    ax.set_title(
        f"Actual: {label_map[actual]}\n"
        f"Predicted: {label_map[predicted]}\n"
        f"{'Correct' if correct else 'Wrong'}"
    )

    ax.axis("off")


plt.suptitle(
    "Logistic Regression Prediction Results "
    "on Assignment Test Images",
    fontsize=16
)

plt.tight_layout()

plt.show()


# In[39]:


display_prediction_results(
    assignment_image_paths,
    y_assignment_test,
    y_assignment_pred,
    assignment_image_lookup,
    f"Logistic Regression Assignment Test Results (C={best_C})"
)


# In[40]:


cm = confusion_matrix(
    y_assignment_test,
    y_assignment_pred
)

fig, ax = plt.subplots(
    figsize=(16, 16)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False
)

plt.title(
    "Logistic Regression Confusion Matrix "
    "on Assignment Test Set"
)

plt.tight_layout()

plt.show()


# In[41]:


# Display ALL assignment test results

total_results = len(assignment_image_paths)

fig, axes = plt.subplots(
    total_results,
    2,
    figsize=(10, 2.8 * total_results),
    squeeze=False
)

for row, index in enumerate(
    range(total_results)
):

    filename = Path(
        str(assignment_image_paths[index])
    ).name

    actual = int(
        y_assignment_test[index]
    )

    predicted = int(
        y_assignment_pred[index]
    )

    correct = (
        actual == predicted
    )

    image_path = assignment_image_lookup.get(
        filename
    )

    if image_path is None:
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
        continue

    # Get original and segmented image
    original, segmented = get_segmented_image(
        image_path
    )

    # -------------------------
    # Original image
    # -------------------------
    axes[row, 0].imshow(
        original
    )

    axes[row, 0].set_title(
        f"Original - {filename}",
        fontsize=9
    )

    axes[row, 0].axis(
        "off"
    )

    # -------------------------
    # Segmented image
    # -------------------------
    axes[row, 1].imshow(
        segmented
    )

    axes[row, 1].set_title(
        f"Actual: {label_map.get(actual, actual)}\n"
        f"Predicted: {label_map.get(predicted, predicted)}\n"
        f"{'Correct' if correct else 'Wrong'}",
        fontsize=9
    )

    axes[row, 1].axis(
        "off"
    )


plt.suptitle(
    f"All Logistic Regression Assignment Test Results (C={best_C})",
    fontsize=16,
    y=0.999
)

plt.subplots_adjust(
    top=0.995,
    bottom=0.005,
    hspace=0.6,
    wspace=0.25
)

plt.show()


# In[ ]:





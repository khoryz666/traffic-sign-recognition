#!/usr/bin/env python
# coding: utf-8

# ### 1.Import Libraries

# ### Shared Segmentation Functions
# 
# This notebook imports the integrated color segmentation functions
# (`segment_red()`, `segment_blue()`, and `segment_yellow()`)
# developed by the color segmentation modules.

# In[ ]:


get_ipython().run_line_magic('run', '"Color Segmentation.ipynb"')


# In[ ]:


#Import the libraries
import os     #auto processes every image in subfolders
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### 2.Image Preprocessing

# In[ ]:


def preprocess_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blurred = cv2.GaussianBlur(img_rgb, (5,5), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    return img_rgb, hsv


# ### 3.Find Best Contour

# In[ ]:


def find_best_contour(mask):

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_score = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)
        print("Area:", area)

        if area < 1000:
            print("Rejected: area too small")
            continue

        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        hull = cv2.convexHull(cnt)

        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = area / hull_area

        circularity = 4*np.pi*area/(perimeter*perimeter)

        score = area * circularity * solidity

        if score > best_score:
            best_score = score
            best_contour = cnt

    return best_contour


# ### 4.Feature Extraction

# In[ ]:


def extract_features(contour):

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return None

    circularity = 4*np.pi*area/(perimeter*perimeter)

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


# ### 5.Shape Classification

# In[ ]:


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


# ### 6. Draw Result

# In[ ]:


def draw_result(img_rgb, contour, shape, bounding_box):

    result = img_rgb.copy()

    x, y, w, h = bounding_box

    cv2.drawContours(
        result,
        [contour],
        -1,
        (0,255,0),
        2
    )

    cv2.putText(
        result,
        shape,
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )

    return result


# ### 7.Main Program

# In[ ]:


input_folder = "Inputs"

for root, dirs, files in os.walk(input_folder):

    for filename in sorted(files):

        image_path = os.path.join(root, filename)

        if os.path.isdir(image_path):
            continue

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = cv2.imread(image_path)

        if img is None:
            continue

        #preprocessing
        img_rgb, hsv = preprocess_image(img)

        #color segmentation
        # mask = segment_traffic_sign(img)
        folder = os.path.basename(root)

        if folder == "Blue Signs":
            mask, debug = segment_blue(img, debug=True)

        elif folder == "Red signs":
            mask, debug = segment_red(img, debug=True)

        elif folder == "Yellow Signs":
            mask, debug = segment_yellow(img, debug=True)
        else:
            continue

        #contour detection
        best_contour = find_best_contour(mask)

        if best_contour is None:
            print(f"{filename}: No traffic sign detected.")
            continue

        #feature extraction
        features = extract_features(best_contour)

        #shape classification
        shape = classify_shape(features)

        print(f"Folder (Actual): {folder}")
        print(f"Image: {filename}")
        print(f"Predicted: {shape}")
        print(f"Vertices: {features['vertices']}")
        print(f"Circularity: {features['circularity']:.3f}")
        print(f"Aspect Ratio: {features['aspect_ratio']:.3f}")
        print("-" * 50)

        #draw final result
        result = draw_result(
            img_rgb,
            best_contour,
            shape,
            features["bounding_box"]
        )

        # display results
        fig, ax = plt.subplots(1,5, figsize=(18,5))

        ax[0].imshow(img_rgb)
        ax[0].set_title("Original")

        ax[1].imshow(debug["filtered"], cmap="gray")
        ax[1].set_title("Filtered")

        ax[2].imshow(debug["threshold"], cmap="gray")
        ax[2].set_title("Adaptive Threshold")

        ax[3].imshow(debug["filled"], cmap="gray")
        ax[3].set_title("Filled Contour")

        ax[4].imshow(result)
        ax[4].set_title(shape)

        for a in ax:
            a.axis("off")
        plt.show()


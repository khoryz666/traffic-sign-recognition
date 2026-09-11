#!/usr/bin/env python
# coding: utf-8

# # Remove Input-Test Images from the Chinese Traffic-Sign Training Dataset
# 
# This notebook prevents data leakage by finding images whose **exact filenames** occur in both:
# 
# - the Chinese traffic-sign dataset used for training; and
# - the 84 assignment input images used for final testing.
# 
# Matching training images are moved to a timestamped backup folder outside the dataset, and their rows are removed from `annotations.csv`. The operation is recoverable because the files are moved rather than permanently deleted.
# After duplicate removal, the notebook also synchronizes `annotations.csv` with the remaining image folder, removes repeated annotation filenames, and verifies that there is exactly one annotation row per image.
# 

# ## Important workflow
# 
# 1. Run all cells once with `ACTION = "preview"`.
# 2. Check the match table and affected class counts.
# 3. Change `ACTION` to `"move"` and enter the required confirmation phrase.
# 4. Run all cells again. The annotation-image synchronization runs after duplicate removal.
# 5. Rerun `03_hog_features_chinese.ipynb`.
# 6. Rerun every classifier notebook.
# 
# Previously generated HOG training and public-test feature files are archived by default because they still contain the old dataset split.

# ## 1. Import libraries

# In[1]:


from datetime import datetime
from pathlib import Path
import shutil
import warnings

import pandas as pd


# ## 2. Configure paths and action

# In[2]:


# Place this notebook in the same project folder as the HOG and classifier notebooks.
DATA_DIR = Path("../data")
DATASET_DIR = DATA_DIR / "chinese_traffic_signs"
INPUT_DIR = DATA_DIR / "Inputs"

FEATURE_DIR = Path("../features")
RESULT_DIR = Path("../results")
BACKUP_ROOT = DATA_DIR / "removed_input_duplicates"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
EXPECTED_INPUT_COUNT = 84

# Safety setting:
# - "preview": report matches only; no dataset files are changed.
# - "move": move matched training images and update annotations.csv.
ACTION = "move"

# Required only when ACTION = "move".
CONFIRMATION = "REMOVE INPUT DUPLICATES"
REQUIRED_CONFIRMATION = "REMOVE INPUT DUPLICATES"

# Recommended: archive old HOG train/test files so they cannot be reused accidentally.
ARCHIVE_EXISTING_HOG_FEATURES = True

RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("Dataset folder:", DATASET_DIR.resolve())
print("Input folder  :", INPUT_DIR.resolve())
print("Action        :", ACTION)


# ## 3. Validate the dataset path

# In[3]:


if not DATASET_DIR.exists() or not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        "Chinese traffic-sign dataset folder was not found:\n"
        f"{DATASET_DIR.resolve()}"
    )

if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
    raise FileNotFoundError(
        "Extracted assignment Inputs folder was not found:\n"
        f"{INPUT_DIR.resolve()}\n\n"
        "Place the extracted Inputs folder inside the project's data folder."
    )

if DATASET_DIR.resolve() == INPUT_DIR.resolve():
    raise ValueError("The dataset folder and input folder must not be the same folder.")

try:
    INPUT_DIR.resolve().relative_to(DATASET_DIR.resolve())
except ValueError:
    pass
else:
    raise ValueError(
        "The Inputs folder must be outside the training dataset folder."
    )

print("Dataset path validation passed.")


# ## 4. Locate and read annotations.csv

# In[4]:


required_annotation_columns = {"file_name", "category"}

annotation_candidates = []
preferred_annotation_file = DATASET_DIR / "annotations.csv"

if preferred_annotation_file.exists():
    annotation_candidates.append(preferred_annotation_file)

annotation_candidates.extend(
    path
    for path in sorted(DATASET_DIR.rglob("*.csv"))
    if path != preferred_annotation_file
)

ANNOTATIONS_FILE = None
annotations_df = None

for candidate in annotation_candidates:
    try:
        candidate_df = pd.read_csv(candidate)
    except Exception:
        continue

    if required_annotation_columns.issubset(candidate_df.columns):
        ANNOTATIONS_FILE = candidate
        annotations_df = candidate_df.copy()
        break

if ANNOTATIONS_FILE is None:
    raise FileNotFoundError(
        "No compatible annotation CSV was found. Expected columns: "
        f"{sorted(required_annotation_columns)}"
    )

annotations_df["file_name"] = (
    annotations_df["file_name"]
    .astype(str)
    .str.strip()
)

print("Annotation file:", ANNOTATIONS_FILE.resolve())
print("Annotation rows:", len(annotations_df))
print("Classes:", annotations_df["category"].nunique())


# ## 5. Read the assignment input filenames

# In[5]:


input_image_paths = sorted(
    path
    for path in INPUT_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)

input_filenames = sorted(
    {path.name for path in input_image_paths},
    key=str.casefold,
)

if not input_filenames:
    raise RuntimeError(
        "No supported image files were found inside the Inputs folder:\n"
        f"{INPUT_DIR.resolve()}"
    )

if len(input_filenames) != EXPECTED_INPUT_COUNT:
    warnings.warn(
        f"Expected {EXPECTED_INPUT_COUNT} unique input image filenames, "
        f"but found {len(input_filenames)}. Check the input source before continuing."
    )

input_keys = {filename.casefold() for filename in input_filenames}

print("Input folder:", INPUT_DIR.resolve())
print("Input image files found:", len(input_image_paths))
print("Unique input image filenames:", len(input_filenames))
display(pd.DataFrame({"Input Filename": input_filenames}))


# ## 6. Find exact filename matches in the training dataset
# 
# Matching is case-insensitive because the project is normally run on Windows. The complete filename, including suffix and extension, must match. For example, `020_0002_j.png` does not match `020_0002.png`.

# In[6]:


dataset_image_paths = sorted(
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)

matching_dataset_paths = [
    path
    for path in dataset_image_paths
    if path.name.casefold() in input_keys
]

matched_input_keys = {path.name.casefold() for path in matching_dataset_paths}
unmatched_input_filenames = [
    filename
    for filename in input_filenames
    if filename.casefold() not in matched_input_keys
]

matching_files_df = pd.DataFrame([
    {
        "Filename": path.name,
        "Dataset Relative Path": str(path.relative_to(DATASET_DIR)),
        "Dataset Full Path": str(path.resolve()),
    }
    for path in matching_dataset_paths
])

print("Dataset images found:", len(dataset_image_paths))
print("Matched dataset files:", len(matching_dataset_paths))
print("Unique matched input filenames:", len(matched_input_keys))
print("Input filenames with no dataset match:", len(unmatched_input_filenames))

display(matching_files_df)

if unmatched_input_filenames:
    display(pd.DataFrame({
        "Input Filename Without Dataset Match": unmatched_input_filenames
    }))

matching_files_df.to_csv(
    RESULT_DIR / "input_training_filename_matches_preview.csv",
    index=False,
)


# ## 7. Find matching annotation rows and affected classes

# In[7]:


annotation_match_mask = (
    annotations_df["file_name"]
    .str.casefold()
    .isin(input_keys)
)

matching_annotations_df = annotations_df.loc[annotation_match_mask].copy()
remaining_annotations_df = annotations_df.loc[~annotation_match_mask].copy()

print("Matching image files:", len(matching_dataset_paths))
print("Matching annotation rows:", len(matching_annotations_df))
print("Remaining annotation rows:", len(remaining_annotations_df))

if len(matching_annotations_df) > 0:
    affected_classes_df = (
        matching_annotations_df
        .groupby("category")
        .size()
        .rename("Rows Removed")
        .reset_index()
        .sort_values("category")
    )
    display(affected_classes_df)
else:
    affected_classes_df = pd.DataFrame(columns=["category", "Rows Removed"])
    print("No annotation rows match the input filenames.")

remaining_class_counts = (
    remaining_annotations_df["category"]
    .value_counts()
    .sort_index()
)

empty_classes = sorted(
    set(annotations_df["category"].unique())
    - set(remaining_annotations_df["category"].unique())
)

print("Classes with zero remaining annotation rows:", empty_classes)

if empty_classes:
    warnings.warn(
        "The cleanup would remove every annotation from one or more classes. "
        "Do not continue until the dataset is reviewed."
    )


# ## 8. Review the cleanup summary

# In[8]:


summary_df = pd.DataFrame([{
    "Input image filenames": len(input_filenames),
    "Dataset images before cleanup": len(dataset_image_paths),
    "Matched dataset files": len(matching_dataset_paths),
    "Matched annotation rows": len(matching_annotations_df),
    "Annotation rows before cleanup": len(annotations_df),
    "Annotation rows after cleanup": len(remaining_annotations_df),
    "Classes after cleanup": remaining_annotations_df["category"].nunique(),
}])

display(summary_df)

if ACTION == "preview":
    print("\nPREVIEW ONLY: no training images or annotation rows have been changed.")
    print('To apply the cleanup, set ACTION = "move" and set:')
    print(f'CONFIRMATION = "{REQUIRED_CONFIRMATION}"')


# ## 9. Apply the cleanup
# 
# This cell only changes files when both conditions are satisfied:
# 
# ```python
# ACTION = "move"
# CONFIRMATION = "REMOVE INPUT DUPLICATES"
# ```
# 
# The matching images, original annotation CSV, old HOG feature files, and audit information are stored under a timestamped folder in `../data/removed_input_duplicates/`.

# In[9]:


cleanup_applied = False
cleanup_backup_dir = None
moved_image_pairs = []
moved_feature_pairs = []
annotation_replaced = False

if ACTION not in {"preview", "move"}:
    raise ValueError('ACTION must be either "preview" or "move".')

if ACTION == "preview":
    print("Cleanup was not applied because ACTION is set to preview.")

else:
    if CONFIRMATION != REQUIRED_CONFIRMATION:
        raise PermissionError(
            "Cleanup was not applied. Set CONFIRMATION exactly to:\n"
            f'CONFIRMATION = "{REQUIRED_CONFIRMATION}"'
        )

    if empty_classes:
        raise RuntimeError(
            "Cleanup stopped because it would leave at least one class without data: "
            f"{empty_classes}"
        )

    if not matching_dataset_paths and not matching_annotations_df.empty:
        warnings.warn(
            "Matching annotation rows exist, but no corresponding dataset images were found. "
            "The rows will still be removed from annotations.csv."
        )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cleanup_backup_dir = BACKUP_ROOT / run_timestamp
    backup_images_dir = cleanup_backup_dir / "images"
    backup_metadata_dir = cleanup_backup_dir / "metadata"
    backup_features_dir = cleanup_backup_dir / "stale_hog_features"

    backup_images_dir.mkdir(parents=True, exist_ok=False)
    backup_metadata_dir.mkdir(parents=True, exist_ok=True)

    annotation_backup_file = backup_metadata_dir / ANNOTATIONS_FILE.name
    shutil.copy2(ANNOTATIONS_FILE, annotation_backup_file)

    try:
        # Move each matched image while preserving its relative dataset path.
        for source_path in matching_dataset_paths:
            relative_path = source_path.relative_to(DATASET_DIR)
            destination_path = backup_images_dir / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            if destination_path.exists():
                raise FileExistsError(
                    f"Backup destination already exists: {destination_path}"
                )

            shutil.move(str(source_path), str(destination_path))
            moved_image_pairs.append((source_path, destination_path))

        # Write the filtered annotation table to a temporary file, then replace
        # annotations.csv atomically.
        temporary_annotation_file = (
            ANNOTATIONS_FILE.parent
            / f"{ANNOTATIONS_FILE.name}.cleanup_temporary"
        )
        remaining_annotations_df.to_csv(
            temporary_annotation_file,
            index=False,
        )
        temporary_annotation_file.replace(ANNOTATIONS_FILE)
        annotation_replaced = True

        # Archive stale HOG train/test features to force feature re-extraction.
        if ARCHIVE_EXISTING_HOG_FEATURES:
            stale_hog_files = [
                FEATURE_DIR / "chinese_traffic_signs_hog_features_train.npz",
                FEATURE_DIR / "chinese_traffic_signs_hog_features_test.npz",
            ]

            for feature_path in stale_hog_files:
                if not feature_path.exists():
                    continue

                backup_features_dir.mkdir(parents=True, exist_ok=True)
                destination_path = backup_features_dir / feature_path.name

                if destination_path.exists():
                    raise FileExistsError(
                        f"Feature backup already exists: {destination_path}"
                    )

                shutil.move(str(feature_path), str(destination_path))
                moved_feature_pairs.append((feature_path, destination_path))

        matching_files_df.to_csv(
            backup_metadata_dir / "moved_training_images.csv",
            index=False,
        )
        matching_annotations_df.to_csv(
            backup_metadata_dir / "removed_annotation_rows.csv",
            index=False,
        )
        summary_df.to_csv(
            backup_metadata_dir / "cleanup_summary.csv",
            index=False,
        )

        cleanup_applied = True

    except Exception:
        # Roll back changes if any part of the operation fails.
        if annotation_replaced and annotation_backup_file.exists():
            shutil.copy2(annotation_backup_file, ANNOTATIONS_FILE)

        for original_path, backup_path in reversed(moved_feature_pairs):
            if backup_path.exists() and not original_path.exists():
                original_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(backup_path), str(original_path))

        for original_path, backup_path in reversed(moved_image_pairs):
            if backup_path.exists() and not original_path.exists():
                original_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(backup_path), str(original_path))

        raise

    print("Cleanup completed successfully.")
    print("Training images moved:", len(moved_image_pairs))
    print("Annotation rows removed:", len(matching_annotations_df))
    print("Old HOG feature files archived:", len(moved_feature_pairs))
    print("Backup folder:", cleanup_backup_dir.resolve())


# ## 10. Verify that no exact input filenames remain in the dataset

# In[10]:


current_dataset_image_paths = sorted(
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)

remaining_image_overlap = [
    path
    for path in current_dataset_image_paths
    if path.name.casefold() in input_keys
]

current_annotations_df = pd.read_csv(ANNOTATIONS_FILE)
current_annotation_overlap = current_annotations_df[
    current_annotations_df["file_name"]
    .astype(str)
    .str.strip()
    .str.casefold()
    .isin(input_keys)
]

verification_df = pd.DataFrame([{
    "Dataset images currently present": len(current_dataset_image_paths),
    "Remaining matching image files": len(remaining_image_overlap),
    "Current annotation rows": len(current_annotations_df),
    "Remaining matching annotation rows": len(current_annotation_overlap),
}])

display(verification_df)

if cleanup_applied:
    if remaining_image_overlap:
        raise RuntimeError(
            "Verification failed: matching input filenames remain in the dataset."
        )

    if len(current_annotation_overlap) > 0:
        raise RuntimeError(
            "Verification failed: matching input filenames remain in annotations.csv."
        )

    print("Verification passed: no exact input filenames remain in the training dataset.")
else:
    print("Preview verification complete. No cleanup was applied.")


# ## 11. Synchronize annotations.csv with the remaining image folder
# 
# This process runs after the duplicate-removal verification. It first removes annotation rows whose `file_name` cannot be found among the remaining dataset images. It then removes repeated `file_name` entries so that each image has exactly one annotation row. When a filename occurs more than once, the first row is retained.
# 
# In preview mode, the notebook reports the rows that would be removed. In move mode, it backs them up and removes them from `annotations.csv`.
# 
# The matching is case-insensitive and uses the complete filename, including its extension.
# 

# In[11]:


# Rescan the dataset after input-duplicate removal.
synchronization_image_paths = sorted(
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

synchronization_image_keys = {
    path.name.casefold()
    for path in synchronization_image_paths
}

annotations_before_sync_df = pd.read_csv(ANNOTATIONS_FILE)

if "file_name" not in annotations_before_sync_df.columns:
    raise ValueError("annotations.csv must contain the file_name column.")

annotations_before_sync_df["file_name"] = (
    annotations_before_sync_df["file_name"]
    .astype(str)
    .str.strip()
)

annotation_filename_keys = (
    annotations_before_sync_df["file_name"]
    .str.casefold()
)

# False means that the annotation row has no corresponding image file.
annotation_image_exists_mask = annotation_filename_keys.isin(
    synchronization_image_keys
)

orphan_annotation_rows_df = annotations_before_sync_df.loc[
    ~annotation_image_exists_mask
].copy()

image_matched_annotations_df = annotations_before_sync_df.loc[
    annotation_image_exists_mask
].copy()

# Keep one annotation row per image filename. The first occurrence is retained.
duplicate_annotation_filename_mask = (
    image_matched_annotations_df["file_name"]
    .str.casefold()
    .duplicated(keep="first")
)

duplicate_annotation_rows_df = image_matched_annotations_df.loc[
    duplicate_annotation_filename_mask
].copy()

synchronized_annotations_df = image_matched_annotations_df.loc[
    ~duplicate_annotation_filename_mask
].copy()

annotation_sync_summary_df = pd.DataFrame([{
    "Image files after duplicate cleanup": len(synchronization_image_paths),
    "Unique image filenames": len(synchronization_image_keys),
    "Annotation rows before synchronization": len(annotations_before_sync_df),
    "Annotation rows without an image": len(orphan_annotation_rows_df),
    "Rows after image matching": len(image_matched_annotations_df),
    "Repeated filename rows": len(duplicate_annotation_rows_df),
    "Final annotation rows": len(synchronized_annotations_df),
    "Final unique annotation filenames": (
        synchronized_annotations_df["file_name"].str.casefold().nunique()
    ),
}])

display(annotation_sync_summary_df)

if not orphan_annotation_rows_df.empty:
    print("Annotation rows whose image file was not found:")
    display(orphan_annotation_rows_df)
else:
    print("Every annotation row currently has a corresponding image file.")

if not duplicate_annotation_rows_df.empty:
    print("Repeated annotation filename rows that will be removed:")
    display(duplicate_annotation_rows_df)
else:
    print("No repeated annotation filenames were found.")

annotation_rows_removed_by_sync = 0

if ACTION == "preview":
    print(
        "PREVIEW ONLY: orphan and repeated-filename annotation rows were "
        "identified but annotations.csv was not changed."
    )

elif ACTION == "move":
    if not cleanup_applied or cleanup_backup_dir is None:
        raise RuntimeError(
            "Annotation synchronization requires the duplicate cleanup to "
            "finish successfully first."
        )

    synchronization_backup_dir = cleanup_backup_dir / "metadata"
    synchronization_backup_dir.mkdir(parents=True, exist_ok=True)

    orphan_annotation_rows_df.to_csv(
        synchronization_backup_dir / "annotation_rows_without_images.csv",
        index=False,
    )

    duplicate_annotation_rows_df.to_csv(
        synchronization_backup_dir / "duplicate_annotation_filename_rows.csv",
        index=False,
    )

    total_annotation_rows_to_remove = (
        len(orphan_annotation_rows_df)
        + len(duplicate_annotation_rows_df)
    )

    if total_annotation_rows_to_remove == 0:
        print("No annotation rows needed to be removed.")

    else:
        annotations_before_sync_backup = (
            synchronization_backup_dir
            / "annotations_before_image_synchronization.csv"
        )
        shutil.copy2(
            ANNOTATIONS_FILE,
            annotations_before_sync_backup,
        )

        temporary_sync_file = (
            ANNOTATIONS_FILE.parent
            / f"{ANNOTATIONS_FILE.name}.sync_temporary"
        )

        try:
            synchronized_annotations_df.to_csv(
                temporary_sync_file,
                index=False,
            )
            temporary_sync_file.replace(ANNOTATIONS_FILE)
            annotation_rows_removed_by_sync = total_annotation_rows_to_remove

        except PermissionError as error:
            if temporary_sync_file.exists():
                temporary_sync_file.unlink()

            if annotations_before_sync_backup.exists():
                shutil.copy2(
                    annotations_before_sync_backup,
                    ANNOTATIONS_FILE,
                )
            raise PermissionError(
                "Cannot update annotations.csv. Close it in Excel, VS Code "
                "CSV Preview, or another application, and pause OneDrive "
                "synchronization before running this cell again."
            ) from error

        except Exception:
            if temporary_sync_file.exists():
                temporary_sync_file.unlink()

            if annotations_before_sync_backup.exists():
                shutil.copy2(
                    annotations_before_sync_backup,
                    ANNOTATIONS_FILE,
                )
            raise

        print(
            "Annotation synchronization and deduplication completed. Rows removed:",
            annotation_rows_removed_by_sync,
        )


# ## 12. Verify the final image and annotation match
# 
# This verification requires exactly one annotation row for every remaining image. It checks the total lengths, duplicate filenames, and exact filename sets.
# 

# In[12]:


final_image_paths = sorted(
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

final_image_keys = {
    path.name.casefold()
    for path in final_image_paths
}

final_annotations_df = pd.read_csv(ANNOTATIONS_FILE)
final_annotation_keys_series = (
    final_annotations_df["file_name"]
    .astype(str)
    .str.strip()
    .str.casefold()
)
final_annotation_keys = set(final_annotation_keys_series)

annotation_filenames_without_images = sorted(
    final_annotation_keys - final_image_keys
)
image_filenames_without_annotations = sorted(
    final_image_keys - final_annotation_keys
)

unique_image_count = len(final_image_keys)
unique_annotation_filename_count = len(final_annotation_keys)
unique_filename_lengths_match = (
    unique_image_count == unique_annotation_filename_count
)
exact_filename_sets_match = (
    final_image_keys == final_annotation_keys
)

duplicate_annotation_row_count = (
    len(final_annotations_df)
    - unique_annotation_filename_count
)

duplicate_image_filename_count = (
    len(final_image_paths)
    - unique_image_count
)

total_lengths_match = (
    len(final_image_paths) == len(final_annotations_df)
)

final_match_df = pd.DataFrame([{
    "Image files": len(final_image_paths),
    "Unique image filenames": unique_image_count,
    "Annotation rows": len(final_annotations_df),
    "Unique annotation filenames": unique_annotation_filename_count,
    "Duplicate annotation filename rows": duplicate_annotation_row_count,
    "Duplicate image filenames in folders": duplicate_image_filename_count,
    "Annotation filenames without images": len(annotation_filenames_without_images),
    "Image filenames without annotations": len(image_filenames_without_annotations),
    "Unique filename lengths match": unique_filename_lengths_match,
    "Total image and annotation lengths match": total_lengths_match,
    "Exact filename sets match": exact_filename_sets_match,
}])

display(final_match_df)

if annotation_filenames_without_images:
    display(pd.DataFrame({
        "Annotation Filename Without Image": (
            annotation_filenames_without_images
        )
    }))

if image_filenames_without_annotations:
    display(pd.DataFrame({
        "Image Filename Without Annotation": (
            image_filenames_without_annotations
        )
    }))

if ACTION == "move":
    if annotation_filenames_without_images:
        raise RuntimeError(
            "Final verification failed: annotation filenames without images remain."
        )

    if image_filenames_without_annotations:
        raise RuntimeError(
            "Final verification failed: some images have no annotation row."
        )

    if duplicate_annotation_row_count > 0:
        raise RuntimeError(
            "Final verification failed: duplicate annotation filenames remain."
        )

    if duplicate_image_filename_count > 0:
        raise RuntimeError(
            "Final verification failed: duplicate image filenames remain."
        )

    if not total_lengths_match:
        raise RuntimeError(
            "Final verification failed: image and annotation lengths differ."
        )

    if not unique_filename_lengths_match:
        raise RuntimeError(
            "Final verification failed: unique filename lengths do not match."
        )

    if not exact_filename_sets_match:
        raise RuntimeError(
            "Final verification failed: image and annotation filename sets differ."
        )

    print("Final verification passed.")
    print(
        "Image files and annotation rows match:",
        len(final_image_paths),
    )
else:
    print("Preview verification complete; annotations.csv was not changed.")


# ## 13. Next steps

# In[13]:


if cleanup_applied:
    print("Next steps:")
    print("1. Rerun 03_hog_features_chinese.ipynb from the beginning.")
    print("2. Confirm new HOG train and test feature files are generated.")
    print("3. Rerun the KNN, Logistic Regression, SVM, and Random Forest notebooks.")
    print("4. Report the new results; do not reuse results generated before cleanup.")
else:
    print("Review the preview tables before enabling ACTION = 'move'.")


# ## Restoring the removed files if necessary
# 
# Every cleanup run creates a timestamped backup under:
# 
# ```text
# ../data/removed_input_duplicates/
# ```
# 
# The `images` subfolder preserves each file's original path relative to the Chinese dataset. The `metadata` subfolder contains the original annotation CSV and audit tables. Old HOG feature files, when found, are kept in `stale_hog_features`.
# 
# Do not move the backup folder inside `chinese_traffic_signs`, because the HOG notebook searches that dataset recursively.
# When annotation synchronization removes rows, the metadata folder also contains `annotations_before_image_synchronization.csv` and `annotation_rows_without_images.csv`. Removed repeated filename rows are listed in `duplicate_annotation_filename_rows.csv`.
# 

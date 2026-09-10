"""Dataset annotation loading, cleaning and train/held-out splitting.

Replaces the per-notebook copies of this logic (previously pasted into both
02-feature-extraction notebooks) and replaces the old
02_remove_input_duplicates_and_sync_annotations.ipynb's (now
00-dataset/02_clean_annotations.ipynb) file-moving approach: the 84 held-out
assignment-test images are never
relocated out of data/chinese_traffic_signs/ - they're excluded from the
training pool purely by filtering annotations, via split_held_out().
"""

from pathlib import Path

import pandas as pd
from PIL import Image

from pipeline import paths

REQUIRED_ANNOTATION_COLUMNS = {
    "file_name",
    "width",
    "height",
    "x1",
    "y1",
    "x2",
    "y2",
    "category",
}


def list_dataset_images(dataset_dir=paths.DATASET_DIR):
    return sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in paths.IMAGE_EXTENSIONS
    )


def find_annotations_file(dataset_dir=paths.DATASET_DIR):
    """Locate annotations.csv (or another compatible CSV) under dataset_dir."""
    preferred = dataset_dir / "annotations.csv"
    candidates = [preferred] if preferred.exists() else []
    candidates += [p for p in sorted(dataset_dir.rglob("*.csv")) if p != preferred]

    for candidate in candidates:
        try:
            df = pd.read_csv(candidate)
        except Exception:
            continue

        if REQUIRED_ANNOTATION_COLUMNS.issubset(df.columns):
            return candidate, df

    raise FileNotFoundError(
        "No compatible annotation CSV was found under "
        f"{dataset_dir.resolve()}. Expected columns: "
        f"{sorted(REQUIRED_ANNOTATION_COLUMNS)}"
    )


def read_raw_annotations(dataset_dir=paths.DATASET_DIR):
    """Read the annotation CSV as-is, with no filtering or deduplication."""
    annotations_file, df = find_annotations_file(dataset_dir)
    df = df.copy()
    df["file_name"] = df["file_name"].astype(str).str.strip()
    return annotations_file, df


def clean_annotations(df, image_lookup):
    """Remove orphan rows (no matching image file) and duplicate file_name
    rows (case-insensitive, keeping the first occurrence).

    Returns (cleaned_df, orphan_count, duplicate_count). Pure dataframe
    filtering - never touches files on disk, so it's safe to call every run.
    """
    df = df.copy()
    df["file_name"] = df["file_name"].astype(str).str.strip()

    has_image = df["file_name"].isin(image_lookup)
    orphan_count = int((~has_image).sum())
    matched = df.loc[has_image].copy()

    is_duplicate = matched["file_name"].str.casefold().duplicated(keep="first")
    duplicate_count = int(is_duplicate.sum())

    cleaned = matched.loc[~is_duplicate].reset_index(drop=True)
    return cleaned, orphan_count, duplicate_count


def load_annotations(dataset_dir=paths.DATASET_DIR):
    """Load a ready-to-use annotation table: orphan/duplicate rows removed,
    category cast to int. Returns (annotations_df, image_lookup)."""
    _, raw_df = read_raw_annotations(dataset_dir)
    image_lookup = {p.name: p for p in list_dataset_images(dataset_dir)}

    cleaned_df, _, _ = clean_annotations(raw_df, image_lookup)

    if len(cleaned_df) == 0:
        raise RuntimeError(
            "The annotation file was found, but none of its rows matched "
            "an image file on disk."
        )

    cleaned_df["category"] = cleaned_df["category"].astype(int)
    return cleaned_df, image_lookup


def held_out_filenames(test_image_list_file=paths.TEST_IMAGE_LIST_FILE):
    """The 84 assignment-test filenames that must never be trained on."""
    return {
        line.strip()
        for line in test_image_list_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def split_held_out(annotations_df, held_out=None):
    """Split an annotation table into (train_pool_df, held_out_df) by
    filename, case-insensitively. Images are never moved - both frames
    point at the same data/chinese_traffic_signs/ folder."""
    if held_out is None:
        held_out = held_out_filenames()

    held_out_keys = {name.casefold() for name in held_out}
    is_held_out = annotations_df["file_name"].str.casefold().isin(held_out_keys)

    train_pool_df = annotations_df.loc[~is_held_out].reset_index(drop=True)
    held_out_df = annotations_df.loc[is_held_out].reset_index(drop=True)
    return train_pool_df, held_out_df


def assert_no_leakage(*filename_groups):
    """Raise if any two groups of filenames (each an iterable of names)
    share a filename, case-insensitively. Replaces the ad hoc leakage checks
    previously duplicated in the SVM notebook."""
    keyed_groups = [{str(name).casefold() for name in group} for group in filename_groups]

    for i in range(len(keyed_groups)):
        for j in range(i + 1, len(keyed_groups)):
            overlap = keyed_groups[i] & keyed_groups[j]
            if overlap:
                raise RuntimeError(
                    f"Data leakage detected: {len(overlap)} filenames appear "
                    f"in more than one split, e.g. {sorted(overlap)[:5]}"
                )


class ChineseTrafficSignDataset:
    """Minimal (image, label) dataset wrapper over an annotation table."""

    def __init__(self, annotation_table, image_lookup):
        self.annotations = annotation_table.reset_index(drop=True).copy()
        self.image_lookup = image_lookup

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]

        image_path = self.image_lookup[row["file_name"]]

        image = Image.open(image_path).convert("RGB")

        label = int(row["category"])

        return image, label

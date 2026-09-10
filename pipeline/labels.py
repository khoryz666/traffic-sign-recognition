"""Class-id -> display-name lookup for the classifier notebooks.

The Kaggle "chinese-traffic-signs" dataset ships only images/ and
annotations.csv - no labels.csv (ClassId/Name) file like GTSRB-style
datasets. load_label_map() honors a real labels.csv if one is present
(dataset_dir/labels.csv), and otherwise falls back to the numeric category
id as the display label, rather than crashing.
"""

import pandas as pd

from pipeline import paths


def load_label_map(dataset_dir=paths.DATASET_DIR, label_file=None):
    if label_file is None:
        label_file = dataset_dir / "labels.csv"

    if label_file.exists():
        labels_df = pd.read_csv(label_file)

        if not {"ClassId", "Name"}.issubset(labels_df.columns):
            raise ValueError("labels.csv must contain ClassId and Name columns.")

        return dict(
            zip(
                labels_df["ClassId"].astype(int),
                labels_df["Name"].astype(str),
            )
        )

    from pipeline.annotations import load_annotations

    annotations_df, _ = load_annotations(dataset_dir)
    class_ids = sorted(annotations_df["category"].unique())
    return {int(class_id): str(int(class_id)) for class_id in class_ids}

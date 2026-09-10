"""Central path constants for the pipeline.

Every notebook lives one directory below the repo root (00-dataset/,
01-roi-segmentation/, 02-feature-extraction/, 03-classifier/), so these
relative paths resolve the same way regardless of which stage imports them,
as long as the notebook's working directory is its own folder (Jupyter's
default).
"""

from pathlib import Path

DATA_DIR = Path("../data")
DATASET_DIR = DATA_DIR / "chinese_traffic_signs"
TEST_IMAGE_LIST_FILE = DATA_DIR / "test-image-list.txt"
ANNOTATIONS_FILE = DATASET_DIR / "annotations.csv"
LABEL_FILE = DATASET_DIR / "labels.csv"

FEATURE_DIR = Path("../features")
RESULT_DIR = Path("../results")
MODEL_DIR = Path("../models")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}

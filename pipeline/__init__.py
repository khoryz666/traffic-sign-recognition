"""Shared code for the traffic-sign-recognition notebook pipeline.

Each numbered stage directory (00-deps-and-dataset, 01-roi-segmentation,
02-feature-extraction, 03-classifier) imports from this package instead of
redefining the same functions notebook-by-notebook. Notebooks add the repo
root to sys.path before importing, e.g.:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("..").resolve()))

    from pipeline import paths, annotations, segmentation, preprocessing, features, labels
"""

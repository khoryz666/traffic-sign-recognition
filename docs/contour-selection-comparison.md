# Contour-Selection Change: Before vs After

Comparison of pipeline results before and after replacing the OR-merged
red/blue/yellow mask contour selection (`area x circularity x solidity`
only) with `pipeline.segmentation.select_best_contour_multi_colour`,
ported from [@kahyikang](https://github.com/kahyikang)'s
`03_automatic_colour_segmentation_for_dataset.ipynb` (each colour's own
best candidate contour scored independently - area, centre distance,
solidity, circularity, shape, colour coverage, hue agreement - then
compared across colours).

Commits: `d6c3cbf` (pipeline refactor) through `6f7c9e4` (docs update).

## 01 — ROI Segmentation (84 held-out images)

| Metric | Before | After |
| :--- | :-: | :-: |
| Traffic-sign contour detected | 84 / 84 | 84 / 84 |
| Circle | 36 | 45 |
| Triangle | 18 | 21 |
| Unknown | 16 | 8 |
| Rectangle | 6 | 6 |
| Octagon | 6 | 2 |
| Square | 2 | 2 |

**Unknown shapes dropped by half (16 → 8)** — the previous merged-mask
approach was more likely to hand shape classification a broken or
cross-colour-contaminated contour it couldn't confidently classify.

## 02 — Feature Extraction

| Metric | Before | After |
| :--- | :-: | :-: |
| HOG / HSV dimensions | 1,764 / 960 | 1,764 / 960 (unchanged) |
| Train extracted | 4,730 / 4,731 | 4,719 / 4,731 |
| Test extracted | 1,180 / 1,183 | 1,177 / 1,183 |
| Held-out extracted | 84 / 84 | 84 / 84 |
| **Task 1 recognition rate (HOG)** | 100% (84/84) | 100% (84/84) |
| **Task 1 recognition rate (HSV)** | 100% (84/84) | 100% (84/84) |

The Task 1 recognition rate (the graded metric, measured only on the 84
held-out images) is unchanged. On the larger train/test pool, extraction
got marginally **stricter**: 11 more train images and 3 more test images
now fail to produce a contour, because the new per-colour scoring's
relative-area, colour-coverage and hue-agreement checks reject a few weak
candidates the old loose union-mask approach used to accept.

## 03 — Classifiers

| Classifier | Parameters (before → after) | Test acc | Test macro F1 | Held-out acc | Held-out macro F1 |
| :--- | :--- | :-: | :-: | :-: | :-: |
| KNN | before: k=1, manhattan, uniform | 0.9839 | 0.9691 | 0.9881 | 0.9901 |
| KNN | after: k=1, euclidean, uniform | **0.9881** | **0.9874** | **1.0000** | **1.0000** |
| Logistic Regression | before: C=1.0 | 0.9856 | 0.9747 | 0.9881 | 0.9901 |
| Logistic Regression | after: C=0.1 | **0.9898** | **0.9933** | **1.0000** | **1.0000** |
| SVM | before: linear, C=10 | 0.9839 | 0.9726 | 0.9881 | 0.9901 |
| SVM | after: rbf, C=30, gamma=0.0001, balanced | **0.9881** | **0.9893** | **1.0000** | **1.0000** |

**All three classifiers now reach 100% accuracy and 100% macro-F1 on the
84-image held-out set**, up from 98.81% / ~99.0% before. Test-split
accuracy and macro-F1 also improved for every classifier. The tuned
hyperparameters shifted too (SVM: linear → rbf kernel; KNN: manhattan →
euclidean distance) — the grid searches re-ran against the cleaner
features and landed on different optima.

## Summary

Cleaner, per-colour contour candidates trade a small amount of raw
extraction coverage on the general train/test pool (~0.3% more images
dropped) for a real jump in downstream classification quality: fewer
ambiguous ("Unknown") shapes, and every classifier now perfectly
classifies the blind 84-image assignment test set.

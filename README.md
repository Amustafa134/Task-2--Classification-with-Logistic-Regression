# Task 2: Classification with Logistic Regression

## Description
Build a classifier to predict a categorical outcome — in this case, the
**variety of a dry bean** — and compare Logistic Regression against Random
Forest and SVM.

## Dataset
- **File:** `Dry_Bean_Dataset.xlsx`
- **Source:** [UCI Machine Learning Repository - Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Rows / Columns:** 13,611 rows (13,543 after cleaning), 16 numeric features + 1 target
- **Target variable:** `Class` (7 bean varieties: Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira)
- **Features:** 16 geometric measurements extracted from bean images (Area,
  Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity,
  ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness,
  ShapeFactor1-4)

> Note: classes are imbalanced — Dermason has 3,546 samples vs. Bombay's 522.
> Macro-averaged precision/recall/F1 are used so each class counts equally,
> regardless of size.

## Steps Performed
1. **Exploration** — checked shape, dtypes, class balance, missing values
   (none found), and duplicate rows (68 found).
2. **Cleaning**
   - Removed 68 exact duplicate rows (13,611 → 13,543) before splitting, to
     prevent train/test leakage.
   - Encoded the text `Class` column into numeric labels with `LabelEncoder`.
3. **Train/test split** — 80/20, **stratified** by class (`stratify=y`) to
   preserve each bean variety's proportion in both sets, given the class
   imbalance.
4. **Feature scaling** — `StandardScaler` fit on training data only, then
   applied to test data (avoids data leakage).
5. **Model training** — trained three classifiers on the same split:
   - Logistic Regression
   - Random Forest
   - SVM (RBF kernel)
6. **Evaluation** — accuracy, precision, recall, F1 (macro-averaged), a
   confusion matrix, and one-vs-rest ROC curves (with AUC) for Logistic
   Regression.
7. **Visualization** — confusion matrix heatmap, ROC curves per bean class,
   and a grouped bar chart comparing all three models across all four metrics.

## Results

| Model                | Accuracy | Precision | Recall | F1-score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression    | 0.9192   | 0.9307    | 0.9300 | 0.9302   |
| Random Forest           | 0.9192   | 0.9322    | 0.9295 | 0.9308   |
| SVM (RBF kernel)         | 0.9210   | 0.9342    | 0.9317 | 0.9328   |

### Key findings
- **All three models perform almost identically** (~92% accuracy), with SVM
  very slightly ahead. This mirrors the same pattern seen on the Iris
  dataset, now confirmed on a dataset 90x larger with 7 classes instead of 3.
- **Bombay is classified perfectly** (0 errors) — its measurements are large
  enough to be unambiguous, even though it's the smallest class in the data.
- **The main source of error is confusion between Dermason and Sira**
  (53 + 41 = 94 misclassifications), followed by smaller confusion between
  Barbunya and Cali. These are visually/geometrically similar bean types.
- **Conclusion:** the ~8% error rate reflects a genuine limit in how well
  these 16 shape features can separate certain bean varieties — not a
  weakness in any specific algorithm. A more complex model does not resolve
  the ambiguity because the overlap exists in the data itself.

### ROC curve results (Logistic Regression, one-vs-rest)
All 7 classes achieve AUC of 0.99-1.00. Bombay, Horoz, and Seker reach a
perfect AUC of 1.00. Dermason and Sira — the two classes most confused with
each other in the confusion matrix — have the (comparatively) lowest AUC at
0.99, with Sira's curve visibly the least sharp of the seven. This shows the
model still ranks Sira/Dermason probabilities well overall; the
misclassifications happen in a narrow zone where the two classes'
measurements genuinely overlap, rather than reflecting a weak model.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place `Dry_Bean_Dataset.xlsx` in the same folder as `main.py`
3. Run: `python main.py`

## Tools
Python, pandas, numpy, scikit-learn, matplotlib, openpyxl (for reading .xlsx)
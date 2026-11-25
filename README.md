# ✈️ Flight Ticket Price Prediction

Predicting airline ticket prices from airline, route, timing, seat class and booking window. This is a regression problem built for the **MLP Sept 2026 Kaggle assignment (KA1)**.

| | |
|---|---|
| **Final model** | Average of tuned LightGBM and XGBoost |
| **Cross-validated R²** | 0.983 (5-fold) |
| **RMSE / MAE** | ≈ 2,930 / ≈ 1,560 |
| **Kaggle score** | 0.983 |

📓 The full analysis is in [`assignment-1.ipynb`](assignment-1.ipynb).

---

## The data

- **train.csv**: 40,000 rows, 10 features plus the `price` target
- **test.csv**: 10,000 rows to predict
- Features: `airline`, `flight`, `source`, `destination`, `departure`, `arrival`, `stops`, `class`, `duration`, `days_left`

Five columns have 6–12% missing values in both train and test, so dropping rows was not an option.

## Highlights

### 🔍 Recovering corrupted flight codes
About 5,900 flight codes looked like `6.00E-269` or `0.00E+00`. These turned out to be **Indigo** codes (`6E-269`) that a spreadsheet had read as scientific notation. Every code prefix maps to exactly one airline, so all **4,613 missing airlines were recovered exactly**, with no guessing.

### 🧹 Schedule-aware imputation
Flights run on fixed schedules, so missing `departure`, `stops` and `duration` values are filled from what the *same flight* usually does, falling back to broader groups when needed. This is a custom scikit-learn transformer (`FlightCleaner`) that is refit inside every CV fold, so there is no leakage.

| Column | Lookup accuracy | Naive baseline |
|---|---|---|
| departure | 90.0% correct | 23.7% |
| stops | 94.8% correct | 83.1% |
| duration | 2.82 h avg error | 5.76 h |

### 📊 Two pricing systems, not one
- A typical **Business** fare costs ~9× a typical **Economy** fare. Only Vistara and Air India sell Business.
- **Economy** pricing is driven by *timing*: fares roughly double about two weeks before departure.
- **Business** pricing is driven by the *specific flight and its duration*. Booking window barely matters.

## Model comparison

Every model was evaluated with the same 5-fold cross-validation, and the full pipeline was refit in each fold.

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **LightGBM (tuned) + XGBoost** | **0.9834** | **2,927** | **1,556** |
| LightGBM (tuned) | 0.9829 | 2,970 | 1,606 |
| XGBoost | 0.9828 | 2,976 | 1,585 |
| Random Forest | 0.9803 | 3,192 | 1,667 |
| Histogram Gradient Boosting | 0.9794 | 3,257 | 1,838 |
| Decision Tree | 0.9730 | 3,735 | 1,883 |
| K-Nearest Neighbours | 0.9540 | 4,873 | 2,870 |
| Linear / Ridge Regression | 0.9110 | 6,775 | 4,584 |
| Mean price (baseline) | 0.0000 | 22,715 | 19,758 |

### What actually moved the score
| Step (LightGBM, same settings) | R² |
|---|---|
| Raw columns, no flight code | 0.9709 |
| + flight code | 0.9813 |
| + repaired flight codes | 0.9817 |
| + schedule gaps filled | 0.9827 |

- Linear → tree ensembles was the biggest jump (0.91 → 0.98).
- Hyperparameter tuning on Ridge, KNN, Decision Tree and LightGBM barely helped (+0.0002 at best). Blending two models helped more.
- A log-price target did **worse** on every metric, because large misses on Business fares dominate R² and RMSE.

## Honest limits

- The overall R² is flattered by the easy Economy/Business split. **Within class**, R² is about **0.73 for Economy** and **0.88 for Business**.
- Most Indigo flight numbers were lost to the corruption, so individual Indigo flights can't be told apart.
- There is no seat-availability or exact-date data, which likely explains much of the remaining error.

## Notebook structure

1. First look at the data
2. Data quality: missing values, damaged flight codes, duplicates, outliers
3. Exploratory data analysis
4. Cleaning and feature engineering
5. Preprocessing for each model family
6. Validation setup
7. Model comparison
8. Effect of cleaning and target choices
9. Hyperparameter tuning on four models
10. Error analysis
11. Feature importance (overall and per class)
12. Final model and `submission.csv`
13. Conclusions

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn lightgbm xgboost
```

Put `train.csv`, `test.csv` and `sample_submission.csv` from the competition next to the notebook (or run it on Kaggle), then run all cells. The output is `submission.csv`.

## Tech stack

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · Matplotlib

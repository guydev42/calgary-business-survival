# Calgary business survival analyzer and location recommender

## Problem statement

Thousands of new businesses register in Calgary each year, yet many close within a few years. Understanding which factors drive longevity and which communities offer the best conditions can help entrepreneurs, economic development agencies, and city planners. This project analyzes 22,000+ business licence records using survival analysis and classification to predict business outcomes and recommend locations.

## Approach

- Fetched business licence and civic census data from Calgary Open Data
- Computed Kaplan-Meier survival curves segmented by business type
- Fitted a Cox Proportional-Hazards model to identify closure risk factors
- Trained Random Forest and XGBoost classifiers for survived-vs-closed prediction
- Built a composite location scoring function (survival 45%, competition 30%, diversity 25%)

## Key results

| Metric | Value |
|--------|-------|
| Cox concordance index | ~0.68 |
| XGBoost accuracy | ~0.80 |
| XGBoost AUC-ROC | ~0.86 |

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
project_09_business_survival_recommender/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── notebooks/
│   └── 01_eda.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py
    └── model.py
```

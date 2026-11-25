# Datasets

## Included Datasets

| Dataset | Rows | Cols | Task | Target | Source |
|---|---|---|---|---|---|
| demo.csv | 150 | 5 | Classification (3 classes) | target | Iris-based demo |
| example_dataset.csv | 150 | 5 | Classification (3 classes) | target | Course template |
| adult.csv | 32,561 | 15 | Classification (2 classes) | income | UCI ML Repository |
| WineQuality.csv | 1,599 | 12 | Regression | quality | UCI ML Repository |
| dataset_facebook.csv | 500 | 19 | Regression | Total Interactions | UCI ML Repository |
| Life Expectancy Data.csv | 2,938 | 22 | Regression | Life expectancy | WHO / Kaggle |
| letter-recognition.csv | 20,000 | 17 | Classification (16 classes) | lettr | UCI ML Repository |
| creditcard (1).csv | 284,807 | 31 | Classification (2 classes, 577:1 imbalance) | Class | Kaggle |
| Online_Retail.csv | 541,909 | 8 | — | — | Not used in testing |

## Notes

- `demo.csv` and `example_dataset.csv` are small datasets for testing that the code runs.
- Only `demo.csv` and `example_dataset.csv` are committed; download the other datasets from the sources listed above.
- All datasets are CSV format with headers.
- The agent supports `--target auto` for automatic target column detection.

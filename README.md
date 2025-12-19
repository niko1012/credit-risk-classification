# Credit Risk Classification Project

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)

## 1. Research Question
**"Which classification model performs best for predicting loan defaults on the German Credit dataset: Logistic Regression, Random Forest, or XGBoost?"**

In the banking sector, the cost of missing a default (False Negative) is much higher than rejecting a good customer (False Positive). Therefore, this project aims to build a model that maximizes **Recall** and **ROC-AUC** to effectively identify high-risk clients, while managing the significant class imbalance in the data (70% Good / 30% Bad).

## 2. Project Structure
The project follows a modular data science structure:

```text
credit-risk-classification/
├── data/                   # Contains the dataset (downloaded automatically)
├── results/                # Generated plots (.png) and metrics (.csv, .txt)
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── data_loader.py      # Automated data downloading and cleaning
│   ├── evaluation.py       # Metrics, Confusion Matrix, and ROC Curves
│   ├── models.py           # Definition of ML models (LR, RF, XGBoost, SMOTE)
│   └── preprocessing.py    # Stratified splitting, Encoding, Scaling
├── tests/                  # Unit tests (Coverage > 90%)
├── .gitignore              # Files to ignore in Git
├── LICENSE                 # MIT License
├── main.py                 # Main entry point for the pipeline
├── PROPOSAL.md             # Initial project proposal
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── setup.py                # Package setup file
```

## 3. Setup & Installation
This project is designed to be reproducible. Follow these steps to set up the environment:

1. Clone the repository:

        ```bash
        git clone [https://github.com/niko1012/credit-risk-classification.git](https://github.com/niko1012/credit-risk-classification.git)
        cd credit-risk-classification
        ```

2. Install dependencies: It is recommended to use a virtual environment.

        ```bash 
        pip install -r requirements.txt
        ```

3. Install the project in editable mode: (Required for imports in tests to work correctly)

        ```bash
        pip install -e .
        ```

## 4. Usage

1. Run the Main Pipeline

To run the full workflow (Load Data -> Preprocess -> Train -> Evaluate):

        ```bash
        python main.py
        ```

**Expected Output:** The script will download the data, train 4 models, print classification reports to the terminal, and save all figures/metrics to the results/ folder.

2. Run Tests

To verify the code integrity and check coverage:

        ```bash
        pytest --cov=src
        ```

## 5. Results & Findings
I compared four approaches, focusing on ROC-AUC (overall performance) and Recall (ability to detect defaults).

The Logistic Regression (Balanced) was the best performer with a ROC-AUC of 0.803, offering the best balance between recall and precision.

The Logistic Regression (SMOTE) showed very similar performance to the balanced weights approach, achieving a ROC-AUC of 0.801.

The Random Forest model reached a ROC-AUC of 0.790; while it had high accuracy, it suffered from poor recall and missed many defaults.

The XGBoost model achieved a ROC-AUC of 0.780, showing signs of overfitting despite the use of regularization parameters.

**Key Takeaway**
Contrary to expectations, the simpler Logistic Regression outperformed complex ensemble methods (Random Forest, XGBoost) on this small dataset (1000 observations).
    - Random Forest achieved the highest accuracy (~77%) but failed to detect the minority class (Recall ~0.35), making it risky for a bank.
    - Logistic Regression achieved the best Recall (~0.80), successfully identifying 80% of the potential defaulters, which is the primary business objective.

## 6. Requirements
- Python 3.10+
- pandas
- numpy
- scikit-learn==1.3.2
- xgboost
- imbalanced-learn==0.11.0
- matplotlib
- seaborn
- requests
- types-requests
- pytest
- pytest-cov
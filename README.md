# Credit Risk Classification Project

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)

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

## 3. Setup & Installation
This project is designed to be reproducible. Follow these steps to set up the environment:

### 1. Clone the repository:

        ```python
        git clone [https://github.com/niko1012/credit-risk-classification.git](https://github.com/niko1012/credit-risk-classification.git)
        cd credit-risk-classification
        ```

### 2. Install dependencies: It is recommended to use a virtual environment.

        ```python
        pip install -r requirements.txt
        ```

### 3. Install the project in editable mode: (Required for imports in tests to work correctly)

        ```python
        pip install -e .
        ```

## 4. Usage

**Run the Main Pipeline**

To run the full workflow (Load Data -> Preprocess -> Train -> Evaluate):

        ```python
        python main.py
        ```

**Expected Output:** The script will download the data, train 4 models, print classification reports to the terminal, and save all figures/metrics to the results/ folder.

**Run Tests**

To verify the code integrity and check coverage (current coverage: 92%):

        ```python
        pytest --cov=src
        ```

## 5. Results & Findings
We compared four approaches. The evaluation focused on ROC-AUC (overall performance) and Recall (ability to detect defaults).

+--------------------------------+---------+--------------------------------------------------------------+
| Model                          | ROC-AUC | Key Observation                                              |
+--------------------------------+---------+--------------------------------------------------------------+
| Logistic Regression (Balanced) | 0.803   | Best Performer. Best balance between recall and precision.   |
| Logistic Regression (SMOTE)    | 0.801   | Very similar performance to the balanced weights approach.   |
| Random Forest                  | 0.790   | High accuracy but poor recall (missed many defaults).        |
| XGBoost                        | 0.780   | Signs of overfitting despite regularization parameters.      |
+--------------------------------+---------+--------------------------------------------------------------+

**Key Takeaway**
Contrary to expectations, the simpler Logistic Regression outperformed complex ensemble methods (Random Forest, XGBoost) on this small dataset (1000 observations).
    - Random Forest achieved the highest accuracy (~77%) but failed to detect the minority class (Recall ~0.35), making it risky for a bank.
    - Logistic Regression achieved the best Recall (~0.80), successfully identifying 80% of the potential defaulters, which is the primary business objective.

## 6. Requirements
- Python 3.10+
- pandas
- numpy
- scikit-learn
- xgboost
- imbalanced-learn
- matplotlib
- seaborn
- requests
- types-requests
- pytest
- pytest-cov
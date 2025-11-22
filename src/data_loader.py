import pandas as pd
import os

def load_german_credit_data(save_path: str = "data/german_credit_data.csv") -> pd.DataFrame:
    
    # Official UCI dataset URL 
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    # Column names (as the raw file does not have a header)
    columns = [
        "Status_checking_account", "Duration_in_month", "Credit_history", "Purpose",
        "Credit_amount", "Savings_account", "Present_employment_since",
        "Installment_rate", "Personal_status_sex", "Other_debtors",
        "Present_residence_since", "Property", "Age", "Other_installment_plans",
        "Housing", "Number_existing_credits", "Job", "Number_people_liable",
        "Telephone", "Foreign_worker", "Target"
    ]

    print(f"Downloading data from {url}...")
    
    try:
        # Read data 
        df = pd.read_csv(url, sep=' ', header=None, names=columns)

        # The target is 1 (Good) or 2 (Bad). 
        # We transform it to 0 (Good/No Default) and 1 (Bad/Default) for standard classification.
        df['Target'] = df['Target'].map({1: 0, 2: 1})

        # Create the data directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save as a CSV
        df.to_csv(save_path, index=False)
        print(f"Data successfully saved to {save_path}")

        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e

if __name__ == "__main__":
    # Simple test to check if the function works when run directly
    df = load_german_credit_data()
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("First 5 rows preview:")
    print(df.head())
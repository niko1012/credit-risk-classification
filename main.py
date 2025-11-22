from src.data_loader import load_german_credit_data
from src.preprocessing import preprocess_and_split

def main():
    
    print("--- Starting Credit Risk Project ---")
    
    # 1. Data Loading
    try:
        print("\n[Step 1] Loading Data...")
        df = load_german_credit_data()
        print("Data loaded successfully.")
        
        # 2. Preprocessing & Splitting
        print("\n[Step 2] Preprocessing & Splitting...")
        X_train, X_test, y_train, y_test = preprocess_and_split(df)
        
        # Verification of the split (Requirement: Stratified Split)
        print("\nVerification of Class Distribution in Train Set:")
        print(y_train.value_counts(normalize=True))
        
        print("Verification of Class Distribution in Test Set:")
        print(y_test.value_counts(normalize=True))
        
    except Exception as e:
        print(f"Critical error in main pipeline: {e}")

if __name__ == "__main__":
    main()
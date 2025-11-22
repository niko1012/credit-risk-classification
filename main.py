from src.data_loader import load_german_credit_data
from src.preprocessing import preprocess_and_split
from src.models import train_models
from src.evaluation import evaluate_models 

def main():
    
    print("--- Starting Credit Risk Project ---")
    
    # 1. Data Loading
    try:
        print("\n[Step 1] Loading Data...")
        df = load_german_credit_data()
        
        # 2. Preprocessing & Splitting
        print("\n[Step 2] Preprocessing & Splitting...")
        X_train, X_test, y_train, y_test = preprocess_and_split(df)
        
        # 3. Model Training
        print("\n[Step 3] Training Models...")
        models = train_models(X_train, y_train)
        
        # 4. Evaluation
        print("\n[Step 4] Evaluation...")
        results_df = evaluate_models(models, X_test, y_test)
        
        print("\n--- Final Results Summary ---")
        print(results_df)
        print("\nProject Pipeline Completed Successfully.")
        
    except Exception as e:
        print(f"Critical error in main pipeline: {e}")

if __name__ == "__main__":
    main()
from src.data_loader import load_german_credit_data

def main():
    """
    Main execution function for the Credit Risk Classification project.
    Orchestrates data loading, processing, and model training.
    """
    print("--- Starting Credit Risk Project ---")
    
    # 1. Data Loading
    try:
        print("\n[Step 1] Loading Data...")
        df = load_german_credit_data()
        
        # Quick check of class imbalance (Requirement from Francesco)
        print("Data loaded successfully.")
        print(f"Total observations: {len(df)}")
        
        # Calculate class distribution
        class_counts = df['Target'].value_counts(normalize=True)
        print("\nTarget Distribution (0=No Default, 1=Default):")
        print(class_counts)
        
    except Exception as e:
        print(f"Critical error in main pipeline: {e}")

if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_and_split(df: pd.DataFrame, target_col: str = 'Target', test_size: float = 0.2, random_state: int = 42):
    """
    Performs data preprocessing and splits the dataset into training and testing sets
    
    Steps:
    1. Separates the target variable (y) from the features (X)
    2. Encodes categorical variables using One-Hot Encoding (pd.get_dummies)
    3. Performs a Stratified Train-Test Split to maintain class distribution
    """
    print("\n[Preprocessing] Starting data preparation...")

    # 1. Separate Features and Target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Encode Categorical Variables
    # Machine Learning models require numerical input 
    # pd.get_dummies converts text columns into binary columns (0/1)
    # drop_first=True avoids multicollinearity (dummy variable trap)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    print(f"Encoded features: {X.shape[1]} original -> {X_encoded.shape[1]} expanded features")

    # 3. Stratified Split
    # 'stratify=y' ensures that both Train and Test sets have the same 
    # proportion of defaults (30%) as the original dataset
    print(f"Splitting data (Test size: {test_size}, Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Simple test logic
    # We create a dummy dataframe to test the function
    data = {
        'Age': [25, 30, 35, 40, 45],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Male'],
        'Target': [0, 0, 1, 0, 1]
    }
    df_dummy = pd.DataFrame(data)
    preprocess_and_split(df_dummy)
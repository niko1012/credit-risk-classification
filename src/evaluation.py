import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def evaluate_models(models: dict, X_test, y_test):
    """
    Evaluates trained models using metrics suited for imbalanced data
    Generates classification reports, confusion matrices, and ROC Curves
    """
    print("\n[Evaluation] Starting model evaluation...")
    
    results = []

    # Setup for ROC Curve plot (we will plot all models on the same figure)
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"\n--- Evaluating {name} ---")
        
        # Predictions
        y_pred = model.predict(X_test)
        # Probability of class 1 (Default) is needed for ROC-AUC
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 1. Classification Report (Precision, Recall, F1)
        print(classification_report(y_test, y_pred))
        
        # 2. ROC-AUC Score
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {auc:.4f}")
        
        # Store result for summary
        results.append({'Model': name, 'ROC-AUC': auc})

        # 3. Add to ROC Curve Plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

        # 4. Confusion Matrix Plot (Saved separately)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        # Save confusion matrix image
        plt.savefig(f"results/confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close() 

    # Finalize and Save ROC Plot (Figure 1)
    plt.figure(1) 
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess") # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/roc_curves_comparison.png")
    print("\nEvaluation plots saved to 'results/' folder.")
    
    return pd.DataFrame(results)
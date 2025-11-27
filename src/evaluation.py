import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def evaluate_models(models: dict, X_test, y_test):
    """
    Evaluates trained models using metrics suited for imbalanced data.
    Generates classification reports, confusion matrices, ROC Curves,
    and saves a full textual report.
    """
    print("\n[Evaluation] Starting model evaluation...")
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    results = []
    
    # String buffer to hold the full text report (exactly like terminal output)
    full_report_text = "--- Detailed Classification Report ---\n\n"

    # Setup for ROC Curve plot
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"\n--- Evaluating {name} ---")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # 1. Classification Report
        # We capture it as a string to save it, and print it to terminal
        report_str = classification_report(y_test, y_pred)
        print(report_str)
        
        # 2. ROC-AUC Score
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {auc:.4f}")
        
        # --- Add to Text Report ---
        full_report_text += f"--- Evaluating {name} ---\n"
        full_report_text += report_str
        full_report_text += f"\nROC-AUC Score: {auc:.4f}\n"
        full_report_text += "-"*60 + "\n\n"
        # --------------------------

        # Store result for CSV summary
        results.append({'Model': name, 'ROC-AUC': auc})

        # 3. Add to ROC Curve Plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

        # 4. Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"results/confusion_matrix_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
        plt.close() 

    # Finalize and Save ROC Plot
    plt.figure(1) 
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/roc_curves_comparison.png")
    
    # --- Save Outputs ---
    
    # 1. Save the CSV summary (Just the scores)
    pd.DataFrame(results).to_csv("results/metrics_summary.csv", index=False)
    
    # 2. Save the FULL textual report (The tables)
    with open("results/evaluation_report.txt", "w") as f:
        f.write(full_report_text)
        
    print("\n[Output] evaluation_report.txt and metrics_summary.csv saved to 'results/' folder.")
    print("[Output] Plots saved to 'results/' folder.")
    
    return pd.DataFrame(results)
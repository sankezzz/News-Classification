import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support
)
from src.config import (
    VECTORIZER_PATH, MODEL_PATH, 
    PROCESSED_TEST_PATH, LABELS
)

def run_advanced_evaluation(test_path, results_dir='results/'):
    # 1. Setup
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    tfidf = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    
    test_df = pd.read_csv(test_path)
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')
    
    X_test = tfidf.transform(test_df['cleaned_text'])
    y_test = test_df['Class Index']
    
    # 2. Predictions
    y_pred = model.predict(X_test)
    target_names = [LABELS[i] for i in sorted(LABELS.keys())]
    
    # 3. Save Text Metrics
    report_path = os.path.join(results_dir, 'advanced_metrics.txt')
    with open(report_path, 'w') as f:
        f.write("ADVANCED MODEL PERFORMANCE REPORT\n")
        f.write("="*40 + "\n")
        f.write(f"Overall Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n\n")
        f.write(classification_report(y_test, y_pred, target_names=target_names))
    
    # 4. Generate Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix: Predicted vs Actual')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # 5. Generate Metrics Bar Chart
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    metrics_df = pd.DataFrame({
        'Category': target_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }).melt(id_vars='Category', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_df, x='Category', y='Score', hue='Metric')
    plt.ylim(0, 1.1)
    plt.title('Performance Metrics by Category')
    
    metrics_plot_path = os.path.join(results_dir, 'metrics_comparison.png')
    plt.savefig(metrics_plot_path)
    plt.close()

    print(f"--- Advanced Evaluation Complete ---")
    print(f"Text report: {report_path}")
    print(f"Visuals saved: {cm_path} and {metrics_plot_path}")

if __name__ == "__main__":
    run_advanced_evaluation(PROCESSED_TEST_PATH)
import joblib
import os
import pandas as pd
from sklearn.metrics import classification_report
from src.config import (
    VECTORIZER_PATH, MODEL_PATH, 
    PROCESSED_TEST_PATH, LABELS
)

def load_resources():
    """Loads the saved vectorizer and model using config paths."""
    tfidf = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return tfidf, model

def predict_custom_text(text, tfidf, model):
    """Predicts the category of a single string of text."""
    vectorized_text = tfidf.transform([text])
    prediction = model.predict(vectorized_text)[0]
    
    return LABELS.get(prediction, "Unknown")

def run_evaluation(test_path, results_dir='results/'):
    """Runs evaluation and saves the report to a text file."""
    tfidf, model = load_resources()
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    test_df = pd.read_csv(test_path)
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')
    
    X_test = tfidf.transform(test_df['cleaned_text'])
    y_test = test_df['Class Index']
    
    y_pred = model.predict(X_test)
    
    target_names = [LABELS[i] for i in sorted(LABELS.keys())]
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    print("\n--- Model Evaluation Report ---")
    print(report)
    
    # Save to text file
    report_path = os.path.join(results_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("AG News Classification - SVM Model Evaluation\n")
        f.write("="*45 + "\n")
        f.write(report)
    
    print(f"✅ Full report saved to: {report_path}")

if __name__ == "__main__":
    # 1. Run full evaluation and save results
    run_evaluation(PROCESSED_TEST_PATH)
    
    # 2. Quick test with custom input
    tfidf_obj, model_obj = load_resources()
    sample_news = "Nvidia announces new AI chips to compete with AMD in the data center market."
    category = predict_custom_text(sample_news, tfidf_obj, model_obj)
    
    print(f"\nSample Input: '{sample_news}'")
    print(f"Predicted Category: {category}")
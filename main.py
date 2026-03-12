import os
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Importing from your project modules
from src.config import (
    RAW_TRAIN_PATH, RAW_TEST_PATH, 
    PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH,
    MODEL_DIR, LABELS
)
from src.preprocessing import preprocess_dataset
from src.feat_engg import extract_features
from src.train import train_model

def run_pipeline():
    print("--- 🚀 Starting News Classification Pipeline ---")

    # 1.Preprocesing
    if not os.path.exists(PROCESSED_TRAIN_PATH):
        print(f"\n[Step 1] Preprocessing Raw Data...")
        preprocess_dataset(RAW_TRAIN_PATH, PROCESSED_TRAIN_PATH)
        preprocess_dataset(RAW_TEST_PATH, PROCESSED_TEST_PATH)
    else:
        print(f"\n[Step 1] Processed data found at {PROCESSED_TRAIN_PATH}. Skipping.")

    # 2.Feature Engineering
    print(f"\n[Step 2] Extracting TF-IDF Features...")
    X_train, X_test, y_train, y_test = extract_features(
        PROCESSED_TRAIN_PATH, 
        PROCESSED_TEST_PATH, 
        MODEL_DIR
    )

    # 3.Training
    print(f"\n[Step 3] Training the Linear SVM...")
    model = train_model(X_train, y_train, MODEL_DIR)

    # 4.Evaluation
    print(f"\n[Step 4] Evaluating Model Performance...")
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    
    # We pull the category names directly from our config LABELS
    target_names = [LABELS[i] for i in sorted(LABELS.keys())]
    
    report = classification_report(
        y_test, 
        predictions, 
        target_names=target_names
    )
    
    print("=" * 40)
    print(f" FINAL ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 40)
    print("Detailed Classification Report:")
    print(report)
    print(" Pipeline Execution Complete ")

if __name__ == "__main__":
    run_pipeline()
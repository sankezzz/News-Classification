import os
from src.config import (
    RAW_TRAIN_PATH, RAW_TEST_PATH, 
    PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH,
    MODEL_DIR
)
from src.preprocessing import preprocess_dataset
from src.feat_engg import extract_features
from src.train import train_model
from src.evaluate import run_full_evaluation

def run_pipeline():
    print("Starting Classification Pipeline ")

    # 1. Preprocessing
    if not os.path.exists(PROCESSED_TRAIN_PATH):
        print("\n[Step 1] Preprocessing Raw Data...")
        preprocess_dataset(RAW_TRAIN_PATH, PROCESSED_TRAIN_PATH)
        preprocess_dataset(RAW_TEST_PATH, PROCESSED_TEST_PATH)
    else:
        print("\n[Step 1] Processed data already exists.")

    # 2.Feature ngineering
    print("\n[Step 2] Extracting Features...")
    X_train, X_test, y_train, y_test = extract_features(
        PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, MODEL_DIR
    )

    # 3.Training
    print("\n[Step 3] Training SVM Model...")
    train_model(X_train, y_train, MODEL_DIR)

    # 4. Eval
    print("\n[Step 4] Running Evaluation & Generating Visuals...")
    run_full_evaluation(PROCESSED_TEST_PATH)

    print("\n--- Pipeline Success! Check the 'results/' folder. ---")

if __name__ == "__main__":
    run_pipeline()
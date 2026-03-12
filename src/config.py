import os

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data Paths
RAW_TRAIN_PATH = os.path.join(DATA_DIR, 'raw', 'train.csv')
RAW_TEST_PATH = os.path.join(DATA_DIR, 'raw', 'test.csv')

PROCESSED_TRAIN_PATH = os.path.join(DATA_DIR, 'processed', 'train_cleaned.csv')
PROCESSED_TEST_PATH = os.path.join(DATA_DIR, 'processed', 'test_cleaned.csv')

# Model Paths
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.pkl')

# Hyperparameters
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
SVM_C = 1.0
RANDOM_STATE = 42

# Label Mapping
LABELS = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
} 
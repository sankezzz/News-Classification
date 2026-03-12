import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def extract_features(train_path, test_path, model_dir='models/'):
    """
    Converts cleaned text into TF-IDF vectors and saves the vectorizer.
    """
    # 1. Load the processed data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)



    # Handle any potential NaNs from preprocessing
    train_df['cleaned_text'] = train_df['cleaned_text'].fillna('')
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')

    print("Vectorizing text data...")
    
    # 2. Initialize TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    # 3. Fit on train, transform both
    X_train = tfidf.fit_transform(train_df['cleaned_text'])
    X_test = tfidf.transform(test_df['cleaned_text'])
    
    y_train = train_df['Class Index']
    y_test = test_df['Class Index']

    # 4. Save the vectorizer for future use (inference)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    joblib.dump(tfidf, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    print(f"Vectorizer saved to {model_dir}")

    return X_train, X_test, y_train, y_test


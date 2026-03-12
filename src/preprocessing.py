import pandas as pd
import re
import string
from nltk.corpus import stopwords
import nltk

# Ensure you have the stopwords downloaded
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Basic text cleaning for News data.
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and special characters
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    
    # 3. Remove numbers (optional, but often helpful for news)
    text = re.sub(r'\d+', '', text)
    
    # 4. Remove stopwords
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    
    return text

def preprocess_dataset(input_path, output_path):
    """
    Reads the raw CSV, cleans the text, and saves to the processed folder.
    """
    # AG News usually has 'Class Index', 'Title', 'Description'
    df = pd.read_csv(input_path)
    
    # Combine Title and Description for better context
    df['full_text'] = df['Title'] + " " + df['Description']
    
    print(f"Cleaning text for {input_path}...")
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # We only need the label and the cleaned text
    processed_df = df[['Class Index', 'cleaned_text']]
    
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_dataset('data/raw/train.csv', 'data/processed/train_cleaned.csv')
    preprocess_dataset('data/raw/test.csv', 'data/processed/test_cleaned.csv')
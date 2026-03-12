import joblib
import os
from sklearn.svm import LinearSVC
from feat_engg import extract_features

def train_model(X_train, y_train, model_dir='models/'):
    """
    Trains a Linear SVM classifier 
    Here i am using a Class of sklearn instead of the svm fuction becuase we have a large dataset 
    """
    print("Initializing Linear SVM...")
    
    model = LinearSVC(C=1.0, random_state=42, max_iter=1000)

    print("Training started (this might take a few moments)...")
    model.fit(X_train, y_train)
    
    model_path = os.path.join(model_dir, 'svm_model.pkl')
    joblib.dump(model, model_path)
    
    print(f"Model trained and saved to {model_path}")
    return model

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = extract_features(
        'data/processed/train_cleaned.csv', 
        'data/processed/test_cleaned.csv'
    )
    
    train_model(X_train, y_train)
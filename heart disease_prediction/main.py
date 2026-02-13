import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(filepath):
    """Loads the dataset from the CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def preprocess_data(df):
    """Preprocesses the data: encoding and scaling."""
    # Define features and target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Identify categorical and numerical columns
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numerical_cols = [col for col in X.columns if X[col].dtype != 'object']

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return X, y, preprocessor

def train_and_evaluate(X, y, preprocessor):
    """Trains multiple models and evaluates them."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True)
    }

    results = {}
    models_trained = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        clf.fit(X_train, y_train)
        models_trained[name] = clf
        
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        # plt.show()

    return results, models_trained

def predict_user_input(model, user_data):
    """Predicts heart disease for a single user input."""
    columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    input_df = pd.DataFrame([user_data], columns=columns)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    return prediction, probability

def main():
    filepath = 'd:/heart disease_prediction/hearts.csv'
    df = load_data(filepath)
    
    if df is not None:
        # Basic EDA
        print("\nDataset Info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())

        X, y, preprocessor = preprocess_data(df)
        results, trained_models = train_and_evaluate(X, y, preprocessor)
        
        print("\nFinal Accuracy Summary:")
        for name, acc in results.items():
            print(f"{name}: {acc:.4f}")

        # === User Prediction Request ===
        # Input: 43,F,TA,100,223,0,Normal,142,N,0,Up
        print("\n--- Testing Specific User Input ---")
        user_input_values = [43, 'F', 'TA', 100, 223, 0, 'Normal', 142, 'N', 0, 'Up']
        
        # Use the best model (Random Forest)
        rf_pipeline = trained_models['Random Forest']
        
        pred, prob = predict_user_input(rf_pipeline, user_input_values)
        
        result_str = "Heart Disease Detected ⚠️" if pred == 1 else "Normal (No Heart Disease) ✅"
        print(f"\nUser Input: {user_input_values}")
        print(f"Prediction: {result_str}")
        print(f"Confidence: {max(prob)*100:.2f}%")

if __name__ == "__main__":
    main()

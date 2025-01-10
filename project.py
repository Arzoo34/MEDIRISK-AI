# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import chi2_contingency


# Step 1: Load and preprocess data
def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv("C:/Users/akanksha dhoundiyal/Desktop/mini project/diabetes.csv")

    # Replace 0s with NaN and fill missing values with median
    columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_to_clean:
        data[col].replace(0, np.nan, inplace=True)
        data[col].fillna(data[col].median(), inplace=True)

    # Split features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    return data, X, y


# Step 2: Exploratory Data Analysis
def perform_eda(data):
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    # Distribution of target variable
    sns.countplot(x='Outcome', data=data)
    plt.title('Distribution of Outcome')
    plt.show()

    # Chi-square test example (e.g., between Pregnancies and Outcome)
    contingency_table = pd.crosstab(data['Pregnancies'], data['Outcome'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-Square Test: chi2={chi2}, p-value={p}")


# Step 3: Train and evaluate Naive Bayes model
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Main function
def main():
    # Load data
    filepath = 'diabetes.csv'  # Ensure this file is in the same directory as the script
    data, X, y = load_and_preprocess_data(filepath)

    # Perform EDA
    perform_eda(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate model
    train_and_evaluate_model(X_train, X_test, y_train, y_test)


# Run the script
if __name__ == "__main__":
    main()

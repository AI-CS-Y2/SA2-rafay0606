import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load the dataset
file_path = 'used_cars_UK.csv'
df = pd.read_csv(file_path)

# Preprocessing
# Remove unnecessary columns
df = df.drop(columns=['Unnamed: 0'])

# Handle missing values
df = df.dropna(subset=['Mileage(miles)', 'Registration_Year', 'Price', 'Fuel type'])

# Bin Registration_Year into intervals
bins = [1999, 2010, 2015, 2020, 2025]  # Define year bins
labels = ['2000-2010', '2011-2015', '2016-2020', '2021-2025']  # Labels for bins
df['Year_Bin'] = pd.cut(df['Registration_Year'], bins=bins, labels=labels)

# Encode the Year_Bin as a categorical variable
label_encoder = LabelEncoder()
df['Year_Bin'] = label_encoder.fit_transform(df['Year_Bin'])

# Define features and target
X = df[['Mileage(miles)', 'Price']]  # Features
y = df['Year_Bin']  # Target

# Split the data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN Classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_knn = knn.predict(X_test_scaled)

# Initialize the logistic regression classifier
log_reg = LogisticRegression(max_iter=1000)

# Train the classifier
log_reg.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_log = log_reg.predict(X_test_scaled)

# Calculate accuracy and F1 score for both models
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')

accuracy_log = accuracy_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log, average='weighted')

# Print the scores
print(f'KNN Accuracy: {accuracy_knn * 100:.2f}%')
print(f'KNN F1 Score: {f1_knn:.2f}')
print(f'Logistic Regression Accuracy: {accuracy_log * 100:.2f}%')
print(f'Logistic Regression F1 Score: {f1_log:.2f}')

# Plot the comparison
metrics = ['Accuracy (%)', 'F1 Score']
knn_scores = [accuracy_knn * 100, f1_knn]
log_scores = [accuracy_log * 100, f1_log]

x = np.arange(len(metrics))  # Label locations
width = 0.35  # Bar width

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, knn_scores, width, label='KNN')
bars2 = ax.bar(x + width/2, log_scores, width, label='Logistic Regression')

# Add labels, title, and legend
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of KNN and Logistic Regression')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Add value labels to the bars
for bars in [bars1, bars2]:
    ax.bar_label(bars, fmt='%.2f')

plt.tight_layout()
plt.show()

# User input for prediction
try:
    mileage = float(input("Enter the mileage (in miles): "))
    price = float(input("Enter the price (in currency): "))
    new_data = pd.DataFrame([[mileage, price]], columns=['Mileage(miles)', 'Price'])
    new_data_scaled = scaler.transform(new_data)

    # Predictions
    knn_prediction = knn.predict(new_data_scaled)
    log_prediction = log_reg.predict(new_data_scaled)

    print(f'KNN Predicted Registration Year for your car: {label_encoder.inverse_transform(knn_prediction)[0]}')
    print(f'Logistic Regression Predicted Registration Year for your car: {label_encoder.inverse_transform(log_prediction)[0]}')

except ValueError:
    print("Invalid input. Please enter numeric values for mileage and price.")

# ============================================
# Construction Delay Prediction Using ANN
# Researcher: Sudarshan
# Civil Engineering — Project Management
# Dataset: CONSTADEX - Construction Management
# ============================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load Dataset
df = pd.read_csv('../data/TRAIN_csv', index_col=0)

# Step 3: Explore the Data
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn Names:", df.columns.tolist())
print("\nDelay Distribution:")
print(df['DELAYED'].value_counts())

# Step 4: Define Features & Target
# Features: built_area, modul_price, weeks_duration,
#           DETACHED, COLLECTIVE, COMMERCIAL, OTHERS
# Target: DELAYED (1 = delayed, 0 = not delayed)
X = df.drop('DELAYED', axis=1)
y = df['DELAYED']

# Step 5: Split Data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Scale the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Build ANN Model
# Architecture: 2 hidden layers (64 and 32 neurons)
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42
)

# Step 8: Train the Model
print("\nTraining ANN model...")
model.fit(X_train, y_train)
print("Training complete!")

# Step 9: Evaluate the Model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Not Delayed', 'Delayed']
))

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load existing data
try:
    data = pd.read_csv('usage_data.csv', names=['age', 'condition'])
except FileNotFoundError:
    # Create default data if none exists
    data = pd.DataFrame({"age": [10, 20, 30, 40, 50], "condition": ["usable", "usable", "throw", "throw", "throw"]})

# Prepare training data
X = data[['age']]
y = data['condition']

# Train a decision tree classifier
model = DecisionTreeClassifier(max_depth=5)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model trained and saved successfully!")

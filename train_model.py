from pycaret.classification import setup, compare_models
from pycaret.datasets import get_data
import joblib

# Load dataset
iris = get_data('iris')
print("--- Dataset Sample ---")
print(iris.head())

# Setup PyCaret experiment
exp = setup(data=iris, target='species', session_id=42, verbose=False, html=False)

# Compare and select best performing model
best_model = compare_models()
print("\n--- Best Performing Model ---")
print(best_model)

# Persist trained model
joblib.dump(best_model, 'iris_model.pkl')
print("\n[SUCCESS] Model saved as iris_model.pkl")

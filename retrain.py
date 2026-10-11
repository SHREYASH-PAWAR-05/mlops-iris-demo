from pycaret.classification import setup, compare_models, finalize_model, save_model
from pycaret.datasets import get_data
import joblib

iris = get_data('iris')
exp = setup(iris, target='species', session_id=42, verbose=False, html=False)
model = compare_models()
final = finalize_model(model)
save_model(final, 'iris_model')

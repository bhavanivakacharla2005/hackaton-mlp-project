# PantryPulse

PantryPulse is a small food-waste risk demo built with a multilayer perceptron (MLP). Enter an item's type, age, quantity, household size, and typical weekly usage to estimate whether it may spoil before it is used.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Model and data

The app calls `load_model()` the first time a user submits the form. That function trains a scikit-learn MLP classifier on a deterministic synthetic dataset, and Streamlit caches the trained model for subsequent use. The labels are generated from simple shelf-life and consumption assumptions so the demo works without downloads or private household data. This is a hackathon prototype, not a food-safety tool; predictions should not be used to decide whether food is safe to eat.
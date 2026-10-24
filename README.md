 # PantryPulse 🥗

### ML-Powered Food Waste Risk Prediction

PantryPulse is a machine learning project that estimates whether a food item may spoil before a household uses it. It uses a Multilayer Perceptron (MLP) classifier with a Streamlit interface to demonstrate food waste risk prediction.

## 🚀 Features

- Predicts estimated food spoilage risk.
- Supports seven food categories.
- Accepts days since purchase, quantity, household size, and typical weekly consumption.
- Uses a scikit-learn MLP classification model.
- Provides an interactive web interface built with Streamlit.
- Caches the trained model for subsequent predictions.

## 🛠️ Technologies Used

- Python
- Streamlit
- scikit-learn
- NumPy
- pandas

## 📁 Project Structure

```
hackaton-mlp-project/
├── app.py
├── mlp_model.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```
git clone https://github.com/bhavanivakacharla2005/hackaton-mlp-project.git
cd hackaton-mlp-project
```

### 2. Create a virtual environment

```
python -m venv .venv
```

Activate it:

**Windows**

```
.venv\Scripts\activate
```

**Linux / macOS / Codespaces**

```
source .venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the application

```
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`. In GitHub Codespaces, open the forwarded port 8501.

## 🧠 How It Works

1. The user selects a food category.
2. The user enters the purchase age, quantity, household size, and typical weekly consumption.
3. The application loads the model when the prediction workflow is first used.
4. The model processes the supplied inputs and estimates the spoilage risk.
5. The application displays the prediction.

## 🧪 Testing Status

The application has been opened and a prediction has been obtained manually in the development environment. A separate automated test suite has not been configured, and comprehensive runtime testing has not been completed.

## ⚠️ Limitations

- The model uses synthetic training data rather than a verified real-world food spoilage dataset.
- Predictions are estimates and have not been validated for real-world accuracy.
- The application is an educational prototype, not a food safety tool.
- Do not use its predictions to decide whether food is safe to consume.

## 🔮 Future Improvements

- Train and evaluate the model using a reliable real-world dataset.
- Add automated unit and integration tests.
- Improve model evaluation using suitable classification metrics.
- Add visual summaries of food waste risk.

## 👩‍💻 Author

**Bhavani Vakacharla**

MCA Student | Machine Learning | Python | Streamlit

GitHub: [bhavanivakacharla2005](https://github.com/bhavanivakacharla2005)

---

*Developed as a machine learning project demonstrating food waste risk estimation.*

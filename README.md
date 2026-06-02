# Heart Disease Prediction

This repository contains an end-to-end classical machine learning pipeline designed to predict the likelihood of heart disease based on clinical parameters. The project encompasses exploratory data analysis, robust data preprocessing, the training of multiple classification models, and an interactive web deployment. 

## 📁 Project Structure

The repository is organized to separate the exploratory workflow, serialized models, and application code:

* **`app.py`**: The main application script used to launch the interactive web interface.
* **`Heart-Disease-Predictor.ipynb`**: A Jupyter Notebook containing the core workflow, including data exploration, feature engineering, and model training.
* **`data/`**: Contains the structured dataset used for the project, specifically `heart.csv`.
* **`artifacts/`**: Stores the serialized machine learning models, preprocessing objects, and performance data.
    * `RandomForest.pkl`: Trained Random Forest classifier.
    * `SVM.pkl`: Trained Support Vector Machine model.
    * `LogisticRegression.pkl`: Trained Logistic Regression model.
    * `DecisionTree.pkl`: Trained Decision Tree classifier.
    * `Preprocessor.pkl`: The data transformation pipeline used to scale and encode features.
    * `metrics.json`: Contains the evaluation metrics and performance results of the trained models.
* **`requirements.txt`**: Lists all necessary Python dependencies to run the project locally.

## 🛠️ Technology Stack

This project utilizes a standard data science and machine learning ecosystem:
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Data Visualization:** `plotly`
* **Deployment & UI:** `streamlit`

## 🚀 Installation and Local Setup

### Clone the Repository

```bash
git clone https://github.com/Priyesh-DS-Code/Heart-Disease-Predictor.git
cd Heart-Disease-Predictor
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will start at:

```text
http://localhost:8501
```

---

## 📈 Project Workflow

1. User enters health-related information.
2. Input data is preprocessed using the trained preprocessor.
3. All machine learning models generate predictions.
4. Predictions are displayed in a comparison table.
5. The SVM model provides the final recommendation.
6. Model accuracy score is displayed for better transparency.

---

## 🔮 Future Enhancements

- Disease Probability Prediction
- Cloud Deployment (AWS)

---

## 👨‍💻 Author

**Priyesh Kumar Kashyap**

B.Tech Computer Science Student  
Machine Learning & Data Scientist Enthusiast

---
  

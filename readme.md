# Medical Insurance Premium Prediction

## Objective

This project aims to provide an **AI-driven web application** for predicting medical insurance premiums based on user input. The model leverages machine learning to estimate insurance charges, helping users and insurers make informed decisions.

---

## Features

- **User-friendly Streamlit web interface**
- Predicts insurance premium based on personal and lifestyle details
- Utilizes a Gradient Boosting Regressor for accurate predictions
- Handles both numerical and categorical data with preprocessing

---

## Model Details

- **Algorithm:** Gradient Boosting Regressor (`sklearn.ensemble.GradientBoostingRegressor`)
- **Preprocessing:**
  - Numerical features scaled with `StandardScaler`
  - Categorical features encoded with `OneHotEncoder`
- **Pipeline:** Combines preprocessing and regression in a single workflow
- **Training Data:** `insurance_dataset.csv` (must include all required columns)

### Input Parameters

| Parameter              | Type  | Description              | Example Values            |
| ---------------------- | ----- | ------------------------ | ------------------------- |
| age                    | int   | Age of the applicant     | 25                        |
| gender                 | str   | Gender                   | "male", "female"          |
| bmi                    | float | Body Mass Index          | 24.5                      |
| children               | int   | Number of children       | 2                         |
| smoker                 | str   | Smoking status           | "yes", "no"               |
| region                 | str   | Residential region       | "southwest", "northeast"  |
| medical_history        | str   | Medical history          | "None", "Diabetes"        |
| family_medical_history | str   | Family medical history   | "Heart disease"           |
| exercise_frequency     | str   | Exercise frequency       | "Never", "Frequently"     |
| occupation             | str   | Occupation type          | "White collar", "Student" |
| coverage_level         | str   | Insurance coverage level | "Basic", "Premium"        |

---

## How to Use

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/medical-insurance.git
cd medical-insurance
```

### 2. Install Requirements

```sh
pip install -r requirements.txt
```

### 3. Train the Model (if needed)

Ensure `insurance_dataset.csv` is present in the project directory.

```sh
python model.py
```

This will generate `trained_model.pkl`.

### 4. Run the Streamlit App

```sh
streamlit run Frontend.py
```

---

## Deployment

You can deploy this app for free using [Streamlit Community Cloud](https://share.streamlit.io/):

1. Push your code and `trained_model.pkl` to a public GitHub repository.
2. Go to Streamlit Community Cloud and connect your repo.
3. Set the main file as `Frontend.py`.
4. Deploy and share your app!

---

## File Structure

```
medical-insurance/
│
├── Frontend.py           # Streamlit web app
├── model.py              # Model training and serialization
├── insurance_dataset.csv # Training data (not included in repo)
├── trained_model.pkl     # Saved trained model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Notes

- Ensure the `scikit-learn` version in `requirements.txt` matches the version used for training the model to avoid pickle errors.
- The app expects the model file (`trained_model.pkl`) to be present in the root directory.

---

## License

This project is for educational purposes. Please check the dataset license before using for commercial purposes.

---

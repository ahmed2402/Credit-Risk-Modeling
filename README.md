# Credit Risk Modeling & Loan Approval Prediction System

A comprehensive machine learning project that predicts loan approval decisions using various algorithms and provides an interactive web interface for real-time predictions.

## 🎯 Project Overview

This project implements a credit risk assessment system that evaluates loan applications and predicts the likelihood of approval or rejection. The system uses multiple machine learning models trained on historical loan data to make informed predictions based on applicant demographics, financial information, and loan characteristics.

## ✨ Key Features

- **Interactive Web Application**: Streamlit-based interface for real-time loan predictions
- **Multiple ML Models**: Comparison of 10+ algorithms including XGBoost, Random Forest, SVM, and more
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Data Preprocessing Pipeline**: Robust feature engineering and data cleaning
- **Model Comparison**: Performance evaluation across multiple algorithms
- **Dual Prediction Interface**: Both XGBoost and PyCaret model predictions available

## 📁 Project Structure

```
Credit-Risk-Modeling/
├── app.py                          # Streamlit web application
├── eda.ipynb                       # Exploratory Data Analysis notebook
├── model.ipynb                     # Model training and evaluation notebook
├── README.md                       # Project documentation
├── datasets/                       # Data files
│   ├── loan_data.csv              # Original dataset (45,000 records)
│   ├── cleaned.csv                # Preprocessed dataset
│   ├── loan2.csv                  # Secondary processed dataset
│   ├── X.csv                      # Feature matrix
│   └── y.csv                      # Target variable
└── models/                        # Trained model artifacts
    ├── best_model.pkl             # PyCaret best model
    ├── xgboost_model.pkl          # XGBoost model
    ├── scaler.pkl                 # StandardScaler for feature normalization
    └── quantile_transformer.pkl   # QuantileTransformer for income feature
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies
Install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm catboost
pip install joblib seaborn matplotlib scipy pycaret
```

Or create a `requirements.txt` file:

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.2.0
joblib>=1.3.0
seaborn>=0.12.0
matplotlib>=3.6.0
scipy>=1.10.0
pycaret>=3.0.0
```

Then install:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Web Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Personal Details**: Enter applicant's age, gender, and education level
2. **Financial Information**: Input income, employment experience, and credit score
3. **Loan Details**: Specify loan amount, interest rate, and purpose
4. **Prediction**: Choose between XGBoost or PyCaret models for prediction

### Jupyter Notebooks

#### Exploratory Data Analysis (`eda.ipynb`)
- Data overview and statistics
- Missing value analysis
- Feature distributions and correlations
- Outlier detection and treatment
- Data preprocessing and feature engineering

#### Model Training (`model.ipynb`)
- Model comparison across 10+ algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - XGBoost
  - LightGBM
  - CatBoost
  - AdaBoost
  - Gradient Boosting
- Hyperparameter tuning
- Performance evaluation
- Model selection and saving

## 📊 Dataset Information

**Source Dataset**: 45,000 loan records with 14 features

### Features:
- **Personal Information**:
  - `person_age`: Age of the applicant
  - `person_gender`: Gender (Male/Female)
  - `person_education`: Education level (High School to Doctorate)
  - `person_income`: Annual income
  - `person_emp_exp`: Employment experience in years
  - `person_home_ownership`: Home ownership status (RENT/OWN/MORTGAGE/OTHER)

- **Financial Information**:
  - `credit_score`: Credit score (300-850)
  - `cb_person_cred_hist_length`: Credit history length in years
  - `previous_loan_defaults_on_file`: Previous default history (Yes/No)

- **Loan Information**:
  - `loan_amnt`: Loan amount requested
  - `loan_intent`: Purpose of loan (PERSONAL/EDUCATION/MEDICAL/VENTURE/HOMEIMPROVEMENT/DEBTCONSOLIDATION)
  - `loan_int_rate`: Interest rate
  - `loan_percent_income`: Loan amount as percentage of income

- **Target Variable**:
  - `loan_status`: Loan approval status (0=Rejected, 1=Approved)

## 🤖 Models & Performance

The project implements and compares multiple machine learning algorithms:

### Available Models:
1. **XGBoost**: Gradient boosting algorithm optimized for performance
2. **PyCaret Best Model**: Automatically selected best-performing model
3. **Additional Models** (trained in notebooks):
   - Random Forest
   - Logistic Regression
   - SVM
   - Neural Networks
   - And more...

### Model Features:
- **Preprocessing Pipeline**: Automatic scaling and transformation
- **Feature Engineering**: One-hot encoding for categorical variables
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Cross-Validation**: Robust model evaluation

## 🎛️ Data Preprocessing

### Feature Engineering:
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler for continuous features
- **Transformation**: QuantileTransformer for income normalization
- **Feature Selection**: Optimized feature set for model performance

### Data Pipeline:
1. Data cleaning and missing value handling
2. Outlier detection and treatment
3. Feature encoding and transformation
4. Train-test split (80-20)
5. Model training and validation

## 📈 Model Evaluation

The models are evaluated using standard classification metrics:
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive prediction accuracy
- **Recall**: Ability to find all positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

## 🔮 Future Enhancements

- [ ] Add more sophisticated feature engineering
- [ ] Implement ensemble methods
- [ ] Add model interpretability features (SHAP values)
- [ ] Create REST API for model serving
- [ ] Add automated model retraining pipeline
- [ ] Implement A/B testing framework
- [ ] Add real-time model monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or suggestions regarding this project, please feel free to reach out!

---

**Note**: This project is for educational and demonstration purposes. In a production environment, additional validation, security measures, and regulatory compliance would be required for financial applications.
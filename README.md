## Confirmed Cases of Compulsory Notification Diseases - Machine Learning Analysis
This project focuses on predicting the number of confirmed cases of compulsory notification diseases using machine learning techniques. The analysis includes data pre-processing, feature selection, and the training of models using the XGBoost algorithm. The goal is to optimize the model for accurate predictions, aiding in public health decision-making.

Table of Contents
Project Overview
Requirements
Installation
Data
Modeling Approach
Model Performance
Usage
Contributing
License
Project Overview
This repository contains a machine learning model for predicting compulsory notification diseases in various municipalities. It uses the XGBoost algorithm, a powerful gradient boosting framework that provides efficient and accurate results. The project includes feature selection, hyperparameter tuning, and model evaluation to improve the predictive power.

Requirements
To run the code in this project, you need the following Python libraries, which can be installed using the requirements.txt file:

Python 3.8+
numpy
pandas
scikit-learn
xgboost
joblib
matplotlib
seaborn
You can install the required libraries using:

bash
Copiar código
pip install -r requirements.txt
Installation
Clone this repository:

bash
Copiar código
git clone https://github.com/CesarRonai/Casos-confirmados-de-doen-as-de-notifica-o-compuls-ria.git
Navigate to the project directory:

bash
Copiar código
cd Casos-confirmados-de-doen-as-de-notifica-o-compuls-ria
Install the required dependencies:

bash
Copiar código
pip install -r requirements.txt
Data
The dataset includes information about compulsory notification diseases in different municipalities over multiple years. It contains the following features:

Municipality: The location where the cases were reported.
Year: The year in which the data was collected.
Confirmed cases: The number of confirmed cases of the diseases.
Source: The source of the data.
Modeling Approach
The project utilizes the XGBoost algorithm for prediction, which is well-suited for structured/tabular data and handles missing values effectively. The following steps were undertaken during the model development process:

Data Preprocessing: Cleaning and transforming the data, handling missing values, and encoding categorical variables.
Feature Selection: Selecting the most relevant features using SelectKBest with f_regression.
Model Training: Training the XGBoost model with default and optimized hyperparameters.
Model Evaluation: Using metrics such as Mean Squared Error (MSE) and R² (Coefficient of Determination) to evaluate model performance.
Model Performance
The model was evaluated on unseen test data. The optimized model achieved the following performance:

SE (MSE): 3445.53
R²: 0.6707
These results indicate that the model can explain about 67% of the variance in the data, with a mean squared error of 3445.53.

Usage
You can use the trained model to make predictions on new data. Follow these steps:

1. Load the Model
Use the following code to load the pre-trained XGBoost model:

python
Copiar código
import joblib

# Load the saved XGBoost model
model = joblib.load('modelo_xgboost_otimizado.pkl')

# Example of how to use the model for prediction
# Assuming X_test is the new dataset for which predictions are needed
y_pred = model.predict(X_test)
2. Save the Model
If you retrain the model or make improvements, you can save it using:

python
Copiar código
import joblib

# Save the optimized model to a file
joblib.dump(model, 'modelo_xgboost_otimizado.pkl')
Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request with your improvements.

Fork the project.
Create your feature branch (git checkout -b feature/AmazingFeature).
Commit your changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

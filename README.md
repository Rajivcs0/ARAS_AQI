# ARAS_AQI

**Overview**
This project predicts PM2.5 concentration using AI & Machine Learning algorithms based on a dataset containing timestamped air quality data. The dataset includes features such as Timestamp, Year, Month, Day, Hour, and PM2.5 concentration levels. The model is developed and executed in Google Colab, Visual Studio Code, and deployed using Streamlit.

**Features**
Data preprocessing and feature engineering
Machine learning model training and evaluation
Deployment using Streamlit for real-time PM2.5 prediction

**Dataset**
The dataset contains the following columns:
Timestamp: Date and time of air quality measurement
Year: Extracted year from timestamp
Month: Extracted month from timestamp
Day: Extracted day from timestamp
Hour: Extracted hour from timestamp
PM2.5: PM2.5 concentration (target variable)

**Technologies Used**
Google Colab: Model training and experimentation
Visual Studio Code: Local development and code structuring
Streamlit: Web-based application for PM2.5 prediction

**Python Libraries:**
Pandas, NumPy (Data Processing)
Scikit-learn (Machine Learning)
Matplotlib, Seaborn (Data Visualization)
Streamlit (Web App Development)

**Installation**

**Clone the repository:**
https://github.com/Rajivcs0/ARAS_AQI.git

**Run the Streamlit app:**
streamlit run app.py

**Model Training**
Load and preprocess the dataset.
Feature extraction (convert timestamp into year, month, day, hour).
Train Machine Learning models such as:
Linear Regression
Random Forest Regressor
Evaluate models using RMSE, R² scores.
Save the trained model using Pickle or Joblib.

**Running the App**
streamlit run app.py

**App Features**
User input for Year, Month, Day, Hour
Real-time PM2.5 prediction

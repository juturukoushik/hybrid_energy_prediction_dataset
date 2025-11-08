HYBRID ENERGY PREDICTION PROJECT

Project Overview:

This project focuses on predicting the amount of energy generated from multiple renewable sources like solar and wind using Artificial Intelligence (AI) and Machine Learning (ML). The main goal is to help in managing and planning renewable energy usage more efficiently.

Objective:

To develop a model that learns from past solar and wind energy production data and accurately predicts future hybrid energy generation.

Week 1 Milestone (30%):

In this phase, the dataset was collected from Kaggle and preprocessed. The data was cleaned, missing values were removed, and datetime features were standardized. Time-based features such as hour, day, month, and weekday were created. A visualization of energy production over time was generated, and the cleaned dataset was saved for future model training.

Week 2 Milestone (50%):

In this phase, the processed data was split into training and testing sets. A Random Forest Regressor model was trained using features like hour, day, month, and energy sources (solar and wind). The model predicted energy production and compared the predicted values with actual results. Performance was evaluated using MAE, RMSE, and R² scores to measure accuracy. The trained model was saved for future use.

Dataset Details:

Dataset: intermittent-renewables-production-france.csv
Source: Kaggle
Description: The dataset contains hourly solar and wind energy production data. It helps identify time-based patterns and trends to predict total hybrid energy output.

AI and ML Involvement:

This project applies both AI and ML concepts.
AI is used for intelligent prediction and decision-making, while ML techniques (Random Forest Regressor) are used to train the model from historical data and make accurate predictions.

Tools and Technologies:-

Python
Google Colab
Pandas, NumPy, Matplotlib
Scikit-learn
Joblib

Conclusion:

Up to Week 2, the project successfully completed data collection, preprocessing, model training, prediction, and evaluation. The model can now predict hybrid renewable energy output accurately and serves as a base for further improvement in upcoming phases.

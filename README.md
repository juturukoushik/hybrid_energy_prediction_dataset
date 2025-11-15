HYBRID ENERGY PREDICTION PROJECT

Predicting hybrid (solar + wind) renewable energy generation using AI & ML.
Clear, reproducible steps for data, preprocessing, models (RF, XGBoost, LSTM), dashboard (Streamlit), evaluation, and deliverables.

📌 Project Summary

This project predicts the amount of electricity produced by hybrid renewable sources (solar + wind) using machine learning and deep learning. The goal is to help planners and grid operators estimate short-term generation to improve scheduling, storage, and demand planning.

What you get:

Cleaned and engineered dataset

Tabular ML models: Random Forest & XGBoost

Sequence model: LSTM (captures time dependencies)

A Streamlit dashboard for live prediction demos

Evaluation metrics, visualizations, and final deliverables

📁 Dataset

File: intermittent-renewables-production-france.csv
Source: Kaggle (used intermittently-renewables production dataset)
Description: Hourly production values for renewable sources (Solar, Wind) and related time features (date/hour). Contains columns like:

Date and Hour, Date, StartHour, EndHour

Source (Solar / Wind)

Production (MW)

engineered fields: DayOfYear, dayName, monthName, etc.

🧹 What we did so far (Milestones)
Week 1 — 30% (Completed)

Collected dataset from Kaggle.

Cleaned and preprocessed: handled missing values, parsed datetimes, removed rows with missing Production (or filled as needed).

Created time-based features: Hour, Day, Month, DayOfWeek.

Pivoted Source to obtain Solar, Wind, and Total_Production.

Saved cleaned dataset: /content/cleaned_hybrid_energy.csv.

Simple visualization: energy production time series.

Week 2 — 50% (Completed)

Feature engineering for ML & LSTM: lag_1, lag_3, rolling_3h, rolling_6h, solar_hour, wind_month, cyclic features (hour_sin, hour_cos, month_sin, month_cos).

Split data: train/test (e.g., 80/20).

Trained models: Random Forest & XGBoost (tabular), LSTM (sequential).

Saved models and scalers: rf_model.pkl, xgb_model.pkl, lstm_model.keras, standard_scaler.pkl, seq_feat_scaler.pkl, seq_target_scaler.pkl.

Built initial Streamlit app skeleton.

Week 3 — Final (To finish / Completed actions)

Performed full EDA, advanced feature engineering and analysis.

Tested advanced models and compared metrics (MAE, RMSE, R²).

Created visualizations: actual vs predicted, residuals, scatter, rolling average, feature importances.

Finalized Streamlit app and prepared downloadable assets.

Documented project and uploaded artifacts.

🔬 Model details & why these algorithms

Random Forest (RF): Robust to outliers, handles non-linear relationships. Good baseline for tabular problems.

XGBoost (XGB): Gradient boosting for strong performance on structured data.

LSTM: Sequence-based model capturing temporal dependencies — useful when past production influences the future.

Scalers:

standard_scaler.pkl for tabular models (RF/XGB).

seq_feat_scaler.pkl for LSTM input features.

seq_target_scaler.pkl for LSTM output inverse-transform.

🛠 How to reproduce (Colab & local)
A. In Google Colab (recommended for training)

Upload intermittent-renewables-production-france.csv to Colab or mount Drive.

Run preprocessing cell (create Datetime, drop missing Production, pivot, time features).

Feature engineering (lags, rolling, cyclic features).

Split data and train:

Train RF & XGB with scikit-learn / xgboost.

Train LSTM with tensorflow.keras (windowing, create sequences).

Save models & scalers:

import joblib
rf.fit(X_train_scaled, y_train)
joblib.dump(rf, '/content/rf_model.pkl')
joblib.dump(standard_scaler, '/content/standard_scaler.pkl')
# For LSTM scalers:
joblib.dump(seq_feat_scaler, '/content/seq_feat_scaler.pkl')
joblib.dump(seq_target_scaler, '/content/seq_target_scaler.pkl')
# Save keras model in .keras format:
lstm.save('/content/lstm_model.keras')


Save predictions CSVs and visualizations to /content/.

Download the model/scaler files to your local machine or upload to Google Drive.

B. Run Streamlit app locally (Windows)

Make sure Python (3.8+) is installed.

Create a venv and install dependencies:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# requirements.txt should include:
# streamlit, joblib, numpy, pandas, scikit-learn, xgboost, tensorflow, matplotlib, seaborn


Place these files into the same folder (streamlit_app):

app.py
rf_model.pkl
xgb_model.pkl
lstm_model.keras
standard_scaler.pkl
seq_feat_scaler.pkl
seq_target_scaler.pkl


Run app:

py -m streamlit run app.py


Open http://localhost:8501 in browser.

📎 app.py — Quick notes (key points)

It loads RF/XGB models and standard_scaler.pkl for tabular predictions.

Loads seq_feat_scaler.pkl (16 features) and seq_target_scaler.pkl (1 target) for LSTM.

If an old/wrong scaler (seq_minmax_scaler.pkl) exists, delete it — it causes feature-size mismatch.

The LSTM input is created as a repeating sequence of current features (naïve) for demo; you can improve by using real historical windows.

📊 Visualizations & evaluation

Metrics computed per model: MAE, RMSE, R².

Visuals produced:

Actual vs Predicted time series (first N samples)

Residual histograms (for error distribution)

Scatter Actual vs Predicted (with R² shown)

24-hour rolling average comparison (smoothing)

RandomForest feature importance

Saved into /plots and also merged predictions csv: merged_predictions_all_models.csv.

✅ What to include in your submission

Hybrid_Energy_Prediction.ipynb (Colab notebook with preprocessing, training, evaluation).

merged_predictions_all_models.csv (test predictions).

final_summary.txt (short project conclusion & metrics).

plots/ PNGs for visuals.

streamlit_app/ folder or Google Drive zipped models link (if models are large).

hybrid_energy_app.md or streamlit_run_log.md (execution log + clickable Drive link).

README.md (this file).

📥 How to provide large model files (GitHub best practices)

If model files exceed 100 MB, either:

Zip them (GitHub allows up to 100 MB per file), or

Upload to Google Drive and provide a clickable link in README.

Example link snippet:

## Download Trained Models
Due to GitHub file limits, trained models are in Google Drive:
[Download models (ZIP)](https://drive.google.com/drive/folders/your-folder-id)

📝 Final summary (example you can paste to submissions)

Hybrid Energy Prediction — Final Summary: This project builds predictive models (Random Forest, XGBoost, LSTM) to estimate hybrid (solar + wind) generation. Data cleaning, feature engineering (time features, lags, rolling means, cyclic encodings), training, and evaluation were completed. A Streamlit dashboard provides an interactive demo. Visualizations and evaluation metrics (MAE, RMSE, R²) are included. Models and scalers were saved and linked for reproducibility.

🔧 Troubleshooting & tips

If Streamlit raises: X has 16 features, but MinMaxScaler is expecting 1 — delete seq_minmax_scaler.pkl and ensure seq_feat_scaler.pkl & seq_target_scaler.pkl exist. The two scalers must be separate.

If running locally, ensure dependencies match Colab environment (TensorFlow and scikit-learn versions can affect saved model loading).

If LSTM predictions are off: check sequence windowing alignment and inverse-scaling logic.

🧭 Next steps & improvements (future work)

Use exogenous variables: weather forecasts (irradiance, wind speed, temperature).

Improve LSTM by training on multi-step forecasting and using attention or transformer-based time-series models.

Add probabilistic forecasting (prediction intervals) using quantile regression or Bayesian approaches.

Deploy Streamlit to a cloud service (Streamlit Cloud, Heroku, or VPS) and automate model updates.

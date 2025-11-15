# app.py (debug-friendly, uses seq_feat_scaler + seq_target_scaler)
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os, traceback

st.set_page_config(page_title='Hybrid Energy Predictor', layout='centered')
st.title('Hybrid Energy Predictor — Wind + Solar (Debug Mode)')

@st.cache_resource
def load_resources():
    resources = {}
    # RF
    try:
        resources['rf'] = joblib.load('rf_model.pkl')
    except Exception as e:
        resources['rf'] = None
        resources['rf_err'] = str(e)
    # XGB
    try:
        resources['xgb'] = joblib.load('xgb_model.pkl')
    except Exception as e:
        resources['xgb'] = None
        resources['xgb_err'] = str(e)
    # Standard scaler
    try:
        resources['scaler'] = joblib.load('standard_scaler.pkl')
    except Exception as e:
        resources['scaler'] = None
        resources['scaler_err'] = str(e)
    # LSTM model (.keras or .h5)
    try:
        if os.path.exists('lstm_model.keras'):
            resources['lstm'] = load_model('lstm_model.keras')
        elif os.path.exists('lstm_model.h5'):
            resources['lstm'] = load_model('lstm_model.h5')
        else:
            resources['lstm'] = None
    except Exception as e:
        resources['lstm'] = None
        resources['lstm_err'] = str(e)
    # SEPARATE scalers for LSTM (features and target)
    try:
        resources['seq_feat_scaler'] = joblib.load('seq_feat_scaler.pkl')
    except Exception as e:
        resources['seq_feat_scaler'] = None
        resources['seq_feat_scaler_err'] = str(e)
    try:
        resources['seq_target_scaler'] = joblib.load('seq_target_scaler.pkl')
    except Exception as e:
        resources['seq_target_scaler'] = None
        resources['seq_target_scaler_err'] = str(e)
    # Old wrong scaler - detect if present
    if os.path.exists('seq_minmax_scaler.pkl'):
        resources['old_seq_scaler_present'] = True
    else:
        resources['old_seq_scaler_present'] = False
    return resources

res = load_resources()

# Show loaded resource status for debugging
st.subheader("Loaded resources (debug)")
col1, col2 = st.columns(2)
with col1:
    st.write("Models:")
    st.write(f"RandomForest: {'Loaded' if res.get('rf') is not None else 'Missing'}")
    if res.get('rf') is None and res.get('rf_err'): st.write(" RF error:", res.get('rf_err'))
    st.write(f"XGBoost: {'Loaded' if res.get('xgb') is not None else 'Missing'}")
    if res.get('xgb') is None and res.get('xgb_err'): st.write(" XGB error:", res.get('xgb_err'))
    st.write(f"LSTM: {'Loaded' if res.get('lstm') is not None else 'Missing'}")
    if res.get('lstm') is None and res.get('lstm_err'): st.write(" LSTM error:", res.get('lstm_err'))
with col2:
    st.write("Scalers:")
    st.write(f"standard_scaler.pkl: {'Loaded' if res.get('scaler') is not None else 'Missing'}")
    if res.get('scaler') is None and res.get('scaler_err'): st.write(" scaler err:", res.get('scaler_err'))
    st.write(f"seq_feat_scaler.pkl: {'Loaded' if res.get('seq_feat_scaler') is not None else 'Missing'}")
    if res.get('seq_feat_scaler') is None and res.get('seq_feat_scaler_err'): st.write(" feat scaler err:", res.get('seq_feat_scaler_err')[:200])
    st.write(f"seq_target_scaler.pkl: {'Loaded' if res.get('seq_target_scaler') is not None else 'Missing'}")
    if res.get('seq_target_scaler') is None and res.get('seq_target_scaler_err'): st.write(" targ scaler err:", res.get('seq_target_scaler_err')[:200])
    if res.get('old_seq_scaler_present'): st.error("Old scaler 'seq_minmax_scaler.pkl' is present — REMOVE/RENAME it to avoid conflicts.")

# If scalers exist, show n_features_in_ (helpful)
def n_features_of(s):
    try:
        return getattr(s, 'n_features_in_', None)
    except Exception:
        return None

if res.get('seq_feat_scaler') is not None:
    st.write("seq_feat_scaler.n_features_in_:", n_features_of(res['seq_feat_scaler']))
if res.get('seq_target_scaler') is not None:
    st.write("seq_target_scaler.n_features_in_:", n_features_of(res['seq_target_scaler']))

# Sidebar inputs
st.sidebar.header('Input Parameters')
hour = st.sidebar.slider('Hour (0-23)', 0, 23, 12)
day = st.sidebar.slider('Day (1-31)', 1, 31, 1)
month = st.sidebar.slider('Month (1-12)', 1, 12, 1)
dow = st.sidebar.slider('DayOfWeek (0=Mon)', 0, 6, 2)
wind = st.sidebar.number_input('Wind production (MW)', value=100.0, step=1.0, format="%.2f")
solar = st.sidebar.number_input('Solar production (MW)', value=200.0, step=1.0, format="%.2f")

# Feature creation same as training
lag_1 = wind + solar
lag_3 = wind + solar
rolling_3h = wind + solar
rolling_6h = wind + solar
solar_hour = solar * hour
wind_month = wind * month
hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)
month_sin = np.sin(2*np.pi*(month-1)/12)
month_cos = np.cos(2*np.pi*(month-1)/12)

vals = [hour, day, month, dow, wind, solar, lag_1, lag_3, rolling_3h, rolling_6h,
        solar_hour, wind_month, hour_sin, hour_cos, month_sin, month_cos]
X = np.array(vals).reshape(1,-1)

st.write("### Input summary")
st.write(f"Hour: {hour}, Day: {day}, Month: {month}, DOW: {dow}")
st.write(f"Wind: {wind:.2f} MW, Solar: {solar:.2f} MW")

# Prepare X_scaled for RF/XGB
X_scaled = X
if res.get('scaler') is not None:
    try:
        X_scaled = res['scaler'].transform(X)
    except Exception as e:
        st.warning(f"standard scaler transform failed: {e}")
        X_scaled = X

# RF/XGB predictions
rf_pred, xgb_pred = None, None
if res.get('rf') is not None:
    try:
        rf_pred = float(res['rf'].predict(X_scaled)[0])
    except Exception as e:
        st.warning("RF predict failed: " + str(e))
if res.get('xgb') is not None:
    try:
        xgb_pred = float(res['xgb'].predict(X_scaled)[0])
    except Exception as e:
        st.warning("XGB predict failed: " + str(e))

# LSTM prediction: verify scalers, shapes
lstm_pred = None
if res.get('lstm') is None:
    st.info("LSTM model not loaded; skipping LSTM prediction.")
else:
    # ensure both scalers loaded
    if res.get('seq_feat_scaler') is None or res.get('seq_target_scaler') is None:
        st.warning("LSTM model loaded but seq_feat_scaler.pkl or seq_target_scaler.pkl missing.")
    else:
        # check n_features_in_ for seq_feat_scaler
        feat_n = n_features_of(res['seq_feat_scaler'])
        if feat_n is not None and feat_n != X.shape[1]:
            st.error(f"seq_feat_scaler expects {feat_n} features but current input has {X.shape[1]}. LSTM disabled.")
        else:
            try:
                X_feat_scaled = res['seq_feat_scaler'].transform(X)  # should be (1,16)
                WINDOW = 24
                seq = np.tile(X_feat_scaled, (WINDOW,1)).reshape(1,WINDOW,X.shape[1])
                lstm_scaled = res['lstm'].predict(seq)
                # inverse transform using target scaler (1-dim)
                lstm_pred = float(res['seq_target_scaler'].inverse_transform(lstm_scaled.reshape(-1,1))[0][0])
            except Exception as e:
                st.error("LSTM prediction error: " + str(e))
                st.error(traceback.format_exc())

st.write('---')
st.write('## Predictions (MW)')
if rf_pred is not None:
    st.success(f"Random Forest: {rf_pred:.2f} MW")
else:
    st.warning("Random Forest not available.")

if xgb_pred is not None:
    st.success(f"XGBoost: {xgb_pred:.2f} MW")
else:
    st.warning("XGBoost not available.")

if lstm_pred is not None:
    st.success(f"LSTM (sequence): {lstm_pred:.2f} MW")
else:
    st.info("LSTM prediction not available.")

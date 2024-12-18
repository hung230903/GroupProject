
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load train and test data
train_data = pd.read_csv('data_train.csv')
test_data = pd.read_csv('data_test.csv')

# Feature engineering: Combine SSID and MAC address
train_data['SSID_MAC'] = train_data['SSID'] + '_' + train_data['MAC address']
test_data['SSID_MAC'] = test_data['SSID'] + '_' + test_data['MAC address']

# OneHotEncoding for SSID_MAC
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_train = encoder.fit_transform(train_data[['SSID_MAC']])
encoded_test = encoder.transform(test_data[['SSID_MAC']])

# Prepare features and targets
X_train = np.hstack((encoded_train, train_data[['Signal Strength (dBm)']]))
X_test = np.hstack((encoded_test, test_data[['Signal Strength (dBm)']]))

y_train_x = train_data['X Position'].values
y_train_y = train_data['Y Position'].values
y_test_x = test_data['X Position'].values
y_test_y = test_data['Y Position'].values

# Train Random Forest models
rf_model_x = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_x.fit(X_train, y_train_x)

rf_model_y = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_y.fit(X_train, y_train_y)

# Make predictions
rf_pred_x = rf_model_x.predict(X_test)
rf_pred_y = rf_model_y.predict(X_test)

# Function to smooth predictions using a moving average
def smooth_predictions(predictions, window_size=3):
    '''
    Smooth predictions using a moving average filter.
    :param predictions: Array of predicted values.
    :param window_size: Size of the smoothing window (default is 3).
    :return: Smoothed predictions.
    '''
    return np.convolve(predictions, np.ones(window_size)/window_size, mode='same')

# Apply smoothing to predictions
window_size = 5  # Adjustable window size
rf_pred_x_smoothed = smooth_predictions(rf_pred_x, window_size=window_size)
rf_pred_y_smoothed = smooth_predictions(rf_pred_y, window_size=window_size)

# Calculate errors
mae_x = mean_absolute_error(y_test_x, rf_pred_x_smoothed)
mae_y = mean_absolute_error(y_test_y, rf_pred_y_smoothed)
rmse_x = mean_squared_error(y_test_x, rf_pred_x_smoothed, squared=False)
rmse_y = mean_squared_error(y_test_y, rf_pred_y_smoothed, squared=False)

# Save models and encoder
joblib.dump(rf_model_x, 'rf_model_x.pkl')
joblib.dump(rf_model_y, 'rf_model_y.pkl')
joblib.dump(encoder, 'encoder.pkl') 

# Print errors
print(f"MAE after smoothing: X={mae_x}, Y={mae_y}")
print(f"RMSE after smoothing: X={rmse_x}, Y={rmse_y}")

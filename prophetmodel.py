import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Define additional metrics
def mean_absolute_deviation(y_true, y_pred):
    """Calculate Mean Absolute Deviation (as MAE for forecasting)."""
    return np.mean(np.abs(y_true - y_pred))

def anomaly_correlation_coefficient(y_true, y_pred, trend=None):
    """Calculate ACC based on anomalies (deviations from mean or trend)."""
    if trend is None:
        y_anomalies = y_true - np.mean(y_true)
        p_anomalies = y_pred - np.mean(y_pred)
    else:
        y_anomalies = y_true - trend
        p_anomalies = y_pred - trend
    return pearsonr(y_anomalies, p_anomalies)[0]

def index_of_agreement(y_true, y_pred):
    """Calculate Index of Agreement (IA)."""
    y_mean = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((np.abs(y_pred - y_mean) + np.abs(y_true - y_mean)) ** 2)
    return 1 - numerator / denominator if denominator != 0 else np.nan

# Load and preprocess data
df = pd.read_csv("Cow_Calf.csv", index_col=[0], parse_dates=[0])
df = df.dropna()  # Remove missing values

# Scale regressors
scaler = StandardScaler()
regressors = ['Net_Gas_Price', 'Corn_Price', 'CPI', 'Exchange_Rate_JPY_USD']
df[regressors] = scaler.fit_transform(df[regressors])

# Train-test split
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].reset_index().rename(columns={'Date': 'ds', 'Gross_Revenue': 'y'})
test = df.iloc[split_idx:].reset_index().rename(columns={'Date': 'ds', 'Gross_Revenue': 'y'})

# Define grid search parameters
param_grid = {
    'changepoint_prior_scale': [0.01, 0.05, 0.1],
    'seasonality_prior_scale': [1.0, 5.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Grid search
best_params = None
best_rmse = float('inf')
results = []

for params in ParameterGrid(param_grid):
    model = Prophet(**params, yearly_seasonality=True)
    
    # Add the regressors
    for reg in regressors:
        model.add_regressor(reg, prior_scale=0.5)
        train[reg] = df[reg].iloc[:split_idx].values
        test[reg] = df[reg].iloc[split_idx:].values
    
    model.fit(train)
    fcst = model.predict(test)
    y_true = test['y'].values
    y_pred = fcst['yhat'].values
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mad = mean_absolute_deviation(y_true, y_pred)  # Same as MAE in this context
    pcc = pearsonr(y_true, y_pred)[0]
    acc = anomaly_correlation_coefficient(y_true, y_pred, trend=fcst['trend'].values)
    ia = index_of_agreement(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    results.append({
        'params': params,
        'rmse': rmse,
        'mae': mae,
        'mad': mad,
        'pcc': pcc,
        'acc': acc,
        'ia': ia,
        'r2': r2
    })
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params

# Print best results
print(f"Best Parameters: {best_params}")
print(f"Best RMSE: {best_rmse:.2f}")

# Print all results
print("\nGrid Search Results:")
for res in results:
    print(f"Params: {res['params']}")
    print(f"RMSE: {res['rmse']:.2f}, MAE: {res['mae']:.2f}, MAD: {res['mad']:.2f}, PCC: {res['pcc']:.2f}, ACC: {res['acc']:.2f}, IA: {res['ia']:.2f}, R²: {res['r2']:.2f}")

# Train final model with best parameters
final_model = Prophet(**best_params, yearly_seasonality=True)
for reg in regressors:
    final_model.add_regressor(reg, prior_scale=0.5)
final_model.fit(train)

# Forecast and evaluate
fcst = final_model.predict(test)
y_true = test['y'].values
y_pred = fcst['yhat'].values

# Store actual, predicted values, and residual error in a CSV file
results_df = pd.DataFrame({
    'Date': test['ds'],
    'Actual': y_true,
    'Predicted': y_pred,
    'Residual': y_true - y_pred
})
results_df.to_csv('prophet_actual_vs_predicted.csv', index=False)

# Final metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mad = mean_absolute_deviation(y_true, y_pred)
pcc = pearsonr(y_true, y_pred)[0]
acc = anomaly_correlation_coefficient(y_true, y_pred, trend=fcst['trend'].values)
ia = index_of_agreement(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\nFinal Model Metrics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAD: {mad:.2f}")
print(f"PCC: {pcc:.2f}")
print(f"ACC: {acc:.2f}")
print(f"IA: {ia:.2f}")
print(f"R²: {r2:.2f}")

# Plot forecast
f, ax = plt.subplots(figsize=(15, 5))
ax.scatter(test['ds'], test['y'], color='r', label='Actual')
final_model.plot(fcst, ax=ax)
plt.legend()
plt.title('Forecast vs Actual')
plt.show()

# Plot components
final_model.plot_components(fcst)
plt.show()
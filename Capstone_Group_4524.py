# Libraries

#for downloading data from NSE website
import bhavcopy

#for several dataframe and other operations in python
import pandas as pd
from datetime import datetime, date
import os
import numpy as np
import math
import numpy as np
import io
import contextlib

#for calculating GARCH volatilities
from arch import arch_model

#for calculating Black Scholes prices and Implied Volatility
from scipy.stats import norm
from scipy.optimize import newton
from scipy.optimize import fsolve

#for applying ANN, LSTM, GRU
import tensorflow as tf
import sklearn.metrics as metrics
from keras.models import Sequential
from keras.layers import GRU, Dense, Concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense

#for plotting
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#function defined for calculating GARCH volatilties - same is accessed later in the code
#GARCH volatility has been further used as an input to Black Scholes model to find option prices

def garch_vol(returns, forecast_horizon, p, q):
    # Estimate GARCH(p,q) model for volatility
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        garch_model = arch_model(returns, vol='Garch', p=p, q=q)
        results = garch_model.fit()

    # Forecast volatility
    forecast = results.forecast(horizon=forecast_horizon)
    vol = forecast.mean.iloc[-1]
    return vol

#function defined for calculating Black Scholes option prices - same is accessed later in the code

def BS(S, K, T, r, sigma, type):
  d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  if (type=='CE'):
    BS = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
  elif (type=='PE'):
    BS = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
  return BS

#function defined for calculating implied volatilites - same is accessed later in the code
#3 different functions have been defined using bissection, fsolve and newton raphson methods of numerical estimation - currently the code later is using fsolver

def iv_bisec(opt_price, S, K, T, r, type):
    tol = 1e-5
    low_vol = 0.001
    high_vol = 5.0
    iterations = 100
    for i in range(iterations):
        mid_vol = (low_vol + high_vol) / 2.0
        price = BS(S, K, T, r, mid_vol, type)
        diff = price - opt_price
        if abs(diff) < tol:
            return mid_vol
        if diff < 0:
            low_vol = mid_vol
        else:
            high_vol = mid_vol
    return None  # Return None if no convergence

def iv_fsolve(opt_price, S, K, T, r, type):
  # Define the function to solve for implied volatility
    def function(sigma, *args):
        opt_price, S, K, T, r, type = args
        return BS(S, K, T, r, sigma, type) - opt_price

    # Initial guess for implied volatility
    initial_guess = 0.3  # You can start with any value here

    # Solve for implied volatility
    implied_vol = fsolve(function, initial_guess, args=(opt_price,S, K, T, r, type))

    return implied_vol[0]

def iv_newton(opt_price, S, K, T, r, type):
  # Define the function to solve for implied volatility
    def function(sigma):
        return BS(S, K, T, r, sigma, type) - opt_price

    # Initial guess for implied volatility
    initial_guess = 0.3  # You can start with any value here

    # Solve for implied volatility
    implied_vol = newton(function, initial_guess)

    return implied_vol

# Define start and end dates, and convert them into date format
start_date = date(2023, 1, 1)
end_date = date(2023, 12, 31)
# Define wait time in seconds to avoid getting blocked
wait_time = [1, 2]

# Define working directory, where files would be saved - this path has to be modified by the user
data_storage = r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524"

# Check if the folder already exists; if not, create it
if not os.path.exists(data_storage):
    os.makedirs(data_storage)
    print(f"Folder '{data_storage}' created successfully!")
else:
    print(f"Folder '{data_storage}' already exists.")

# change directory as per above path
os.chdir(data_storage)

# Instantiate bhavcopy class for equities, indices, and derivatives
if os.path.isfile(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\indices.csv"):
    data_nifty = pd.read_csv(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\indices.csv", parse_dates=['TIMESTAMP'])
else:
    nse = bhavcopy.bhavcopy("indices", start_date, end_date, data_storage, wait_time)
    nse.get_data()
    data_nifty = pd.read_csv(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\indices.csv", parse_dates=['TIMESTAMP'])

data_nifty = data_nifty.loc[data_nifty['Index Name'] == 'Nifty 50']
data_nifty.rename(columns={"Index Name": "Index", "Closing Index Value": "Close"}, inplace=True)

#creating input columns using the underlying data - returns, squared returns, historical volatilies with different tenors
data_nifty['rt'] = pd.to_numeric(data_nifty['Change(%)'])
data_nifty['rt2'] = pd.to_numeric(data_nifty['rt'])**2.
data_nifty['sigma2'] = data_nifty['rt'].rolling(2).std()*(252**0.5)
data_nifty['sigma3'] = data_nifty['rt'].rolling(3).std()*(252**0.5)
data_nifty['sigma5'] = data_nifty['rt'].rolling(5).std()*(252**0.5)
data_nifty['sigma20'] = data_nifty['rt'].rolling(20).std()*(252**0.5)
data_nifty['sigma60'] = data_nifty['rt'].rolling(60).std()*(252**0.5)
data_nifty['sigma110'] = data_nifty['rt'].rolling(110).std()*(252**0.5)

#saving the processed underlying data file in folder
data_nifty.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\data_nifty.csv')

#Getting option chain data on NIFTY 50 data fetching historical data (option chain for each day in 2023) from www.nseindia.com
dt = pd.date_range(start=start_date, end=end_date, freq='B')
datafno = pd.DataFrame()
if os.path.isfile(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\datafno.csv"):
    datafno = pd.read_csv(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\datafno.csv", parse_dates=['TIMESTAMP'])
else:
    for tday in dt:
        try:
            dd = datetime.strftime(tday, '%d')
            MM = datetime.strftime(tday, '%b').upper()
            YYYY = datetime.strftime(tday, '%Y')
            fnoBhavcopyUrl = 'http://archives.nseindia.com/content/historical/DERIVATIVES/' +YYYY+ '/' +MM+ '/fo' + dd+ MM+ YYYY+'bhav.csv.zip'
            print(fnoBhavcopyUrl)
            datafno1 = pd.read_csv(fnoBhavcopyUrl, parse_dates=['EXPIRY_DT', 'TIMESTAMP'])
            datafno = pd.concat([datafno, datafno1], join = 'outer', ignore_index=True)
        except:
            print("Error in" + dd + MM + YYYY)

    datafno = datafno.drop(datafno.columns[15:], axis=1)
    datafno.columns = [c.strip() for c in datafno.columns.values.tolist()]

    #only taking FnO data on underlying index and dropping other indices and stocks to make file of manageable size
    datafno = datafno.loc[datafno['SYMBOL'] == 'NIFTY']

    #saving the processed Nifty50 FnO data file in folder
    datafno.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\datafno.csv')

def check_date_format(date_string, date_format):
    try:
        datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

def convert_date_format(date_string):
    if check_date_format(date_string, "%d-%b-%Y") == True:
        return datetime.strptime(date_string, "%d-%b-%Y").strftime('%d-%m-%Y')
    else:
        return date_string

datafno['EXPIRY_DT'] = pd.to_datetime(datafno['EXPIRY_DT'].apply(convert_date_format), format='mixed', dayfirst=True)

#separating out the FnO data into 2 files - one with futures and other with options
datafno_fut = datafno.loc[datafno['INSTRUMENT'] == 'FUTIDX']
datafno_opt = datafno.loc[(datafno['INSTRUMENT'] == 'OPTIDX')&(datafno['CONTRACTS'] > 0)]

#Creating input file for applying Black Scholes, GARCH volatilites, neural networks (ANN, LSTM, GRU)
data_input = pd.merge(datafno_opt, data_nifty, on='TIMESTAMP')
data_input['S'] = data_input['Close']
data_input['K'] = data_input['STRIKE_PR']
data_input['Moneyness'] = data_input['Close']/data_input['STRIKE_PR']
data_input['T'] = pd.to_datetime(data_input['EXPIRY_DT'])-pd.to_datetime(data_input['TIMESTAMP'])
data_input['T'] = data_input['T'].dt.days
r = 6.9441 #risk free 30day t-bill rate as taken from Reserve Bank of India website

#implied vol calculation using iv_fsolve function
data_input['IV'] = list(map(lambda opt_price, S, K, T, type: iv_fsolve(opt_price, S, K, T, r/100,type), data_input['CLOSE'], data_input['S'], data_input['K'], data_input['T']/365, data_input['OPTION_TYP']))
data_input = data_input.dropna()

#saving the processed input data file in folder
data_input.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\data_input.csv')

#separating the input data into call options and put options
#we have used call options data only for the purpose of this study
data_inputCE = data_input.loc[data_input['OPTION_TYP'] == 'CE']
data_inputPE = data_input.loc[data_input['OPTION_TYP'] == 'PE']

#forecasting volatilities using GARCH(1,1)

#data taken till 30th Sep 2023 as input in order to forecast volatility upto 63 days ahead till end Dec 2023
filtered_df = data_nifty[(data_nifty['TIMESTAMP'] <= pd.to_datetime(date(2023, 9, 30)))]

#forecasting for upto 63 trading days (3 calender month) to cover the entire year till end of 2023
forecast_horizons = range(1, 63)
forecast_results = {}
filtered_df['Close'] = pd.to_numeric(filtered_df['Close'], errors='coerce')
filtered_df.dropna(subset=['Close'], inplace=True)
# filtered_df.loc[filtered_df['Close'].notnull(), 'Close'] =  filtered_df['Close'].dropna()
garchvol = pd.DataFrame()
garchvol['T'] = forecast_horizons

for horizon in forecast_horizons:
    # Create a new column for returns with the specified horizon
    returns = filtered_df['Close'].pct_change(periods=horizon)
    filtered_df[f'Return_{horizon}D'] = returns
    rescaled_returns = returns[~np.isnan(returns)] * 100
    forecast_results[horizon] = garch_vol(rescaled_returns, horizon, 1, 1)

#filtered_df.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\filtered_df.csv')

for horizon, volatility in forecast_results.items():
    print(f"Forecast horizon: {horizon}, Forecasted volatility: {volatility[0]*((252/horizon)**0.5):.4f}")
    garchvol.loc[(garchvol['T'] == horizon),'vol_garch'] = volatility[0]*((252/horizon)**0.5)

#saving the garchvol file in folder
garchvol.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\garchvol.csv')

#calculation of option prices using Black Scholes

output_BS = pd.DataFrame()
columns_to_replicate = data_inputCE.iloc[:, 2:6]
output_BS[columns_to_replicate.columns] = columns_to_replicate
output_BS['Close'] = pd.to_numeric(data_inputCE['CLOSE'])
output_BS['q'] = pd.to_numeric(data_inputCE['Div Yield'])
output_BS['S'] = pd.to_numeric(data_inputCE['S'])
output_BS['K'] = pd.to_numeric(data_inputCE['K'])
output_BS['T'] = pd.to_numeric(data_inputCE['T'])
output_BS['r-q'] = r - output_BS['q']
output_BS['Moneyness'] = pd.to_numeric(data_inputCE['Moneyness']).round(3)
output_BS = pd.merge(output_BS, garchvol, on='T')
output_BS['BS_price'] = output_BS.apply(lambda row: BS(row['S'], row['K'], row['T']/365, row['r-q']/100, row['vol_garch']/100, 'CE'), axis=1)

#saving the Black Scholes option prices output file in folder
output_BS.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\output_BS.csv')

#error metrics for BS output vis-a-vis actual market prices of corresponding options
mae = metrics.mean_absolute_error(output_BS['Close'], output_BS['BS_price'])
mse = metrics.mean_squared_error(output_BS['Close'], output_BS['BS_price'])
rmse = np.sqrt(mse)
mape = metrics.mean_absolute_percentage_error(output_BS['Close'], output_BS['BS_price'])
r2 = metrics.r2_score(output_BS['Close'], output_BS['BS_price'])

print("BS error metrics:")
print("MAE:", "%.2f" %mae)
print("MSE:", "%.2f" %mse)
print("RMSE:", "%.2f" %rmse)
print("MAPE:", "%.2f" %mape)
print("R-Squared:", "%.3f" %r2)

#Running ANN for call options

#number of input columns are last 14 columns of data_inputCE file
ncol = 14
X = data_inputCE.iloc[:,-ncol:]
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(data_inputCE['CLOSE'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create the neural network model
ANN = Sequential()
ANN.add(Dense(64, input_dim=ncol, activation='relu'))  # Input layer
ANN.add(Dense(32, activation='relu'))  # Hidden layer
ANN.add(Dense(32, activation='relu'))  # Hidden layer
ANN.add(Dense(32, activation='relu'))  # Hidden layer
ANN.add(Dense(1, activation='linear', kernel_constraint=tf.keras.constraints.NonNeg()))  # Output layer

def custom_loss(y_true, y_pred):
    # Compute the mean squared error loss
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    # Penalize negative predictions by adding their absolute values
    neg_penalty = tf.reduce_mean(tf.abs(tf.minimum(y_pred - y_true, 0)))
    # Total loss with an added penalty for negative predictions
    total_loss = mse_loss + neg_penalty
    return total_loss

# Compile the model
ANN.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Train the model
ANN.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model on the test set
loss, mae = ANN.evaluate(X_test, y_test)

# Predict option prices using the trained model
y_pred = ANN.predict(X_test)
output_ANN = pd.DataFrame()
output_ANN['S'] = X_test['S']
output_ANN['K'] = X_test['K']
output_ANN['T'] = X_test['T']
#output_ANN['actual_price'] = y_test
output_ANN['ANN_price'] = y_pred.round(2)

#saving the ANN output to folder
output_ANN.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\output_ANN.csv')

#preparing dataframe having comparison of actual prices, BS prices, ANN prices
comparemodels = pd.merge(output_BS, output_ANN, on=['S','K','T'])

#error metrics for ANN output vis-a-vis actual market prices of corresponding options
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("ANN error metrics:")
print("MAE:", "%.2f" %mae)
print("MSE:", "%.2f" %mse)
print("RMSE:", "%.2f" %rmse)
print("MAPE:", "%.2f" %mape)
print("R-Squared:", "%.3f" %r2)

#plotting ANN prices vs. actual prices
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel("Real Value")
plt.ylabel("ANN Value")
plt.annotate("r-squared = {:.3f}".format(r2_score(y_test,y_pred)), (20,1), size=15)
plt.savefig(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\plot_ANN.png", format="png")
plt.show()

#Running LSTM for call options
# Assuming data and ncol are defined similarly to the previous code
# Preprocessing
data = data_inputCE.iloc[:,-ncol:]
data = data.apply(pd.to_numeric, errors='coerce')
data['opt_price'] = pd.to_numeric(data_inputCE['CLOSE'])
# Normalizing the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Splitting data into features and target
X = scaled_data[:, :-1]  # Features (all columns except the last one)
y = scaled_data[:, -1]   # Target (last column - option_price)
X = X.reshape(X.shape[0], 1, X.shape[1])
# Reshaping the data for LSTM (samples, time steps, features)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=200, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=200))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, kernel_constraint=tf.keras.constraints.NonNeg()))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Predictions
predicted_values = model.predict(X_test)

# You can inverse_transform the predicted values to get the actual option prices if needed
predicted_values = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), predicted_values), axis=1))
actual_prices = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.reshape(len(y_test), 1)), axis=1))
actual_prices = pd.DataFrame(actual_prices)
predicted_values = pd.DataFrame(predicted_values)
actual_prices.columns = data.columns
predicted_values.columns = data.columns
y_test = actual_prices.iloc[:,-1:]
y_pred = predicted_values.iloc[:,-1:]
output_LSTM = pd.DataFrame()
output_LSTM['S'] = actual_prices['S']
output_LSTM['K'] = actual_prices['K']
output_LSTM['T'] = actual_prices['T']
output_LSTM['LSTM_price'] = y_pred.round(2)

#saving the LSTM output to folder
output_LSTM.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\output_LSTM.csv')

#comparison of actual prices, BS prices, ANN prices, LSTM prices
comparemodels = pd.merge(comparemodels, output_LSTM, on=['S','K','T'])

#error metrics for LSTM output vis-a-vis actual market prices of corresponding options
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("LSTM error metrics:")
print("MAE:", "%.2f" %mae)
print("MSE:", "%.2f" %mse)
print("RMSE:", "%.2f" %rmse)
print("MAPE:", "%.2f" %mape)
print("R-Squared:", "%.3f" %r2)

#plotting LSTM prices vs. actual prices
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred)
plt.xlabel("Real Value")
plt.ylabel("LSTM Value")
plt.annotate("r-squared = {:.3f}".format(r2_score(y_test,y_pred)), (20,1), size=15)
plt.savefig(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\plot_LSTM.png", format="png")
plt.show()

#Running GRU for call options
# Assuming data and ncol are defined similarly to the previous code
# Preprocessing
data = data_inputCE.iloc[:, -ncol:]
data = data.apply(pd.to_numeric, errors='coerce')
data['opt_price'] = pd.to_numeric(data_inputCE['CLOSE'])

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

X = scaled_data[:, :-1]
y = scaled_data[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build GRU model
model = Sequential()
model.add(GRU(units=200, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(units=200))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, kernel_constraint=tf.keras.constraints.NonNeg()))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Predictions
predicted_values = model.predict(X_test)

# Inverse transform for original scale
predicted_values = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), predicted_values), axis=1))
actual_prices = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.reshape(len(y_test), 1)), axis=1))
actual_prices = pd.DataFrame(actual_prices)
predicted_values = pd.DataFrame(predicted_values)
actual_prices.columns = data.columns
predicted_values.columns = data.columns
y_test = actual_prices.iloc[:, -1:]
y_pred = predicted_values.iloc[:, -1:]
output_GRU = pd.DataFrame()
output_GRU['S'] = actual_prices['S']
output_GRU['K'] = actual_prices['K']
output_GRU['T'] = actual_prices['T']
output_GRU['GRU_price'] = y_pred.round(2)

#saving the GRU output to folder
output_GRU.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\output_GRU.csv')
comparemodels = pd.merge(comparemodels, output_GRU, on=['S','K','T'])

#saving the final comparison of option prices from all models
comparemodels.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\comparemodels.csv')

#error metrics for GRU output vis-a-vis actual market prices of corresponding options
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("GRU error metrics:")
print("MAE:", "%.2f" %mae)
print("MSE:", "%.2f" %mse)
print("RMSE:", "%.2f" %rmse)
print("MAPE:", "%.2f" %mape)
print("R-Squared:", "%.3f" %r2)

#plotting GRU prices vs. actual prices
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Real Value")
plt.ylabel("GRU Value")
plt.annotate("r-squared = {:.3f}".format(r2_score(y_test, y_pred)), (20,1), size=15)
plt.savefig(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M7 submission\Capstone_Grp4524\plot_GRU.png", format="png")
plt.show()

# Define ranges for moneyness - OTM, ATM & ITM
#compare the error metrics of 4 models in each of the range
ranges = [(0.5, 0.9), (0.9, 1.1), (1.1, 1.5)]

# Iterate through the ranges
def calculate_errors(filter_df, model):
    mse = round(metrics.mean_squared_error(filter['Close'], filter[f'{model}_price']), 3)
    rmse = round(np.sqrt(mse), 3)
    mae = round(metrics.mean_absolute_error(filter['Close'], filter[f'{model}_price']), 3)
    mape = round(metrics.mean_absolute_percentage_error(filter['Close'], filter[f'{model}_price']), 3)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# Initialize dictionaries to store error metrics for each range and model
error_metrics = {r: {model: [] for model in ['BS', 'ANN', 'LSTM', 'GRU']} for r in ranges}

# Iterate through the ranges
for r in ranges:
    # Filter the DataFrame based on moneyness range
    filter = comparemodels[(comparemodels['Moneyness'] >= r[0]) & (comparemodels['Moneyness'] < r[1])]

    # Calculate errors for each model and store in the respective dictionary
    for model in ['BS', 'ANN', 'LSTM', 'GRU']:
        error_metrics[r][model] = calculate_errors(filter, model)

# Create DataFrames for each range and model
dfs = {r: {model: pd.DataFrame([error_metrics[r][model]]) for model in error_metrics[r]} for r in ranges}

# Combine OTM error metrics for each model into a single DataFrame
combined_dfs = {r: pd.concat([dfs[r][model] for model in ['BS', 'ANN', 'LSTM', 'GRU']],
                             keys=['BS', 'ANN', 'LSTM', 'GRU']).reset_index(level=0).rename(
    columns={'level_0': 'Model'}) for r in ranges}

# Display the combined DataFrames for each range
for r in ranges:
    print(f"Range {r} Error Metrics:")
    print(combined_dfs[r])
    print("\n")
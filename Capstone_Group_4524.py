import pandas as pd
from datetime import datetime, date
import os
import bhavcopy
import numpy as np
import math
import sklearn.metrics as metrics
from scipy.stats import norm
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.optimize import newton
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def BS(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def calculate_iv(opt_price, S, K, T, r):
    tol = 1e-5
    low_vol = 0.001
    high_vol = 5.0
    iterations = 100
    for i in range(iterations):
        mid_vol = (low_vol + high_vol) / 2.0
        price = BS(S, K, T, r, mid_vol)
        diff = price - opt_price
        if abs(diff) < tol:
            return mid_vol
        if diff < 0:
            low_vol = mid_vol
        else:
            high_vol = mid_vol
    return None  # Return None if no convergence

# Define start and end dates, and convert them into date format
start_date = date(2023, 5, 26)
end_date = date.today()
# Define wait time in seconds to avoid getting blocked
wait_time = [1, 2]

print(os.getcwd())
# Place data need to be stored.
data_storage = r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M6 submission\capstone code files"
# Define working directory, where files would be saved
os.chdir(data_storage)

# Instantiate bhavcopy class for equities, indices, and derivatives
if os.path.isfile(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M6 submission\capstone code files\indices.csv"):
    data_nifty = pd.read_csv(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M6 submission\capstone code files\indices.csv", parse_dates=['TIMESTAMP'])
else:
    nse = bhavcopy.bhavcopy("indices", start_date, end_date, data_storage, wait_time)
    nse.get_data()
    data_nifty = pd.read_csv(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M6 submission\capstone code files\indices.csv", parse_dates=['TIMESTAMP'])

data_nifty = data_nifty.loc[data_nifty['Index Name'] == 'Nifty 50']
data_nifty.rename(columns={"Index Name": "Index", "Closing Index Value": "Close"}, inplace=True)
data_nifty['rt'] = pd.to_numeric(data_nifty['Change(%)'])
data_nifty['rt2'] = pd.to_numeric(data_nifty['Change(%)'])**2.
data_nifty['sigma5'] = data_nifty['rt'].rolling(5).std()*(252**0.5)
data_nifty['sigma20'] = data_nifty['rt'].rolling(20).std()*(252**0.5)
data_nifty['sigma60'] = data_nifty['rt'].rolling(60).std()*(252**0.5)
data_nifty['sigma75'] = data_nifty['rt'].rolling(75).std()*(252**0.5)

dt = pd.date_range(start=start_date, end=end_date, freq='B')
datafno = pd.DataFrame()
if os.path.isfile(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M6 submission\capstone code files\datafno.csv"):
    datafno = pd.read_csv(r"C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M6 submission\capstone code files\datafno.csv", parse_dates=['TIMESTAMP'])
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
    datafno = datafno.loc[datafno['SYMBOL'] == 'NIFTY']
    datafno.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M6 submission\capstone code files\datafno.csv')

datafno_fut = datafno.loc[datafno['INSTRUMENT'] == 'FUTIDX']
datafno_opt = datafno.loc[(datafno['INSTRUMENT'] == 'OPTIDX')&(datafno['CONTRACTS'] > 0)]

data_input = pd.merge(datafno_opt, data_nifty, on='TIMESTAMP')
data_input['S'] = data_input['Close']
data_input['K'] = data_input['STRIKE_PR']
data_input['T'] = pd.to_datetime(data_input['EXPIRY_DT'])-pd.to_datetime(data_input['TIMESTAMP'])
data_input['T'] = data_input['T'].dt.days
data_input = data_input.dropna()

data_inputCE = data_input.loc[data_input['OPTION_TYP'] == 'CE']
data_inputPE = data_input.loc[data_input['OPTION_TYP'] == 'PE']
data_input.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M6 submission\capstone code files\data_input.csv')

np.random.seed(42)
ncol = 10
X = data_inputCE.iloc[:,-ncol:]
X = X.apply(pd.to_numeric, errors='coerce')

y = pd.to_numeric(data_inputCE['CLOSE'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Create the neural network model
ANN = Sequential()
ANN.add(Dense(64, input_dim=ncol, activation='relu'))  # Input layer
ANN.add(Dense(32, activation='relu'))  # Hidden layer
ANN.add(Dense(32, activation='relu'))  # Hidden layer
ANN.add(Dense(32, activation='relu'))  # Hidden layer
ANN.add(Dense(1, activation='linear'))  # Output layer

# Compile the model
ANN.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])

# Train the model
ANN.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model on the test set
loss, mae = ANN.evaluate(X_test, y_test)

# Predict option prices using the trained model
y_pred = ANN.predict(X_test)
output_ann = pd.DataFrame()
output_ann["y_test"] = y_test
output_ann["y_pred"] = y_pred

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

plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel("Real Value")
plt.ylabel("ANN Value")
plt.annotate("r-squared = {:.3f}".format(r2_score(y_test,y_pred)), (20,1), size=15)
plt.show()

# LSTM model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
ncol = 10
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
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Predictions
predicted_values = model.predict(X_test)

# You can inverse_transform the predicted values to get the actual option prices if needed
predicted_values = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), predicted_values), axis=1))
actual_prices = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.reshape(len(y_test), 1)), axis=1))
actual_prices = pd.DataFrame(actual_prices)
predicted_values = pd.DataFrame(predicted_values)
y_test = actual_prices.iloc[:,-1:]
y_pred = predicted_values.iloc[:,-1:]
output_lstm = pd.DataFrame()
output_lstm["y_test"] = y_test
output_lstm["y_pred"] = y_pred

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

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred)
plt.xlabel("Real Value")
plt.ylabel("LSTM Value")
plt.annotate("r-squared = {:.3f}".format(r2_score(y_test,y_pred)), (20,1), size=15)
plt.show()

output_ann.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M6 submission\capstone code files\output_ann.csv')
output_lstm.to_csv(r'C:\Users\kapil\Desktop\worldquant\Courses\10. Capstone\JB KA Capstone\M6 submission\capstone code files\output_lstm.csv')
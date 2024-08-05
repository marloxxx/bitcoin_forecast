import datetime
import requests
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBRegressor
from pyswarm import pso
from sklearn.metrics import mean_squared_error

# Function to get Bitcoin price data from the API


def get_bitcoin_price(time_start_unix, time_end_unix):
    api_url = f'https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical?id=1&convertId=2794&timeStart={
        time_start_unix}&timeEnd={time_end_unix}'
    r = requests.get(api_url)
    if r.status_code == 200:
        data = []
        for item in r.json()['data']['quotes']:
            data.append({
                'close': item['quote']['close'],
                'volume': item['quote']['volume'],
                'date': item['quote']['timestamp'],
                'high': item['quote']['high'],
                'low': item['quote']['low'],
                'open': item['quote']['open']
            })
        return data
    else:
        return None

# Function to predict using the LSTM model


def lstm_predict(data, time_step, pred_days):
    scaler = MinMaxScaler()
    data = np.array(data['close']).reshape(-1, 1)
    model = tf.keras.models.load_model('static/models/lstm_model.h5')
    scaled_data = scaler.fit_transform(data)
    prediction = []
    x_input = scaled_data[-time_step:].reshape(1, -1)
    temp_input = list(x_input[0])
    lst_output = []
    n_steps = time_step
    i = 0
    while i < pred_days:
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, n_steps, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1
    prediction = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
    return prediction

# Function to predict using the PSO-XGBoost model


def pso_xgboost_predict(data, time_step, pred_days):
    scaler = MinMaxScaler()
    data = np.array(data['close']).reshape(-1, 1)
    model = joblib.load('static/models/pso_xgboost_model.pkl')
    scaled_data = scaler.fit_transform(data)
    prediction = []
    x_input = scaled_data[-time_step:].reshape(1, -1)
    temp_input = list(x_input[0])
    i = 0
    while i < pred_days:
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            yhat = model.predict(x_input)
            temp_input.extend(yhat.tolist())
            temp_input = temp_input[1:]
            prediction.extend(yhat.tolist())
            i += 1
        else:
            yhat = model.predict(x_input)
            temp_input.extend(yhat.tolist())
            prediction.extend(yhat.tolist())
            i += 1
    prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
    return prediction


def update_lstm_model(X_train, y_train):
    model_path = 'static/models/lstm_model.h5'
    model = tf.keras.models.load_model(model_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=6, min_lr=0.0001, verbose=1)
    model.fit(X_train, y_train, epochs=400, batch_size=64,
              verbose=1, callbacks=[early_stopping, reduce_lr])
    model.save(model_path)
    print("LSTM Model updated successfully!")

# Function to update XGBoost model with PSO optimization


def update_xgb_model_with_pso(X_train, y_train, X_test, y_test):
    # Define objective function for PSO
    def objective_function(params):
        learning_rate, max_depth, n_estimators = params
        model = XGBRegressor(learning_rate=learning_rate,
                             n_estimators=int(n_estimators),
                             max_depth=int(max_depth),
                             reg_lambda=0.1,
                             reg_alpha=0.1,
                             early_stopping_rounds=100,
                             n_jobs=4,
                             booster='gbtree',
                             objective='reg:squarederror',
                             random_state=42,
                             gamma=0,
                             subsample=1,
                             colsample_bytree=1,
                             verbosity=0)
        model.fit(X_train, y_train, eval_set=[
                  (X_train, y_train), (X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse

    # Define bounds for PSO
    lb = [0.01, 1, 10]
    ub = [0.11, 20, 100]

    # Use PSO to find optimal hyperparameters
    best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=10)

    # Train XGBoost model with the best parameters
    learning_rate, max_depth, n_estimators = best_params
    xgb_model = XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        reg_lambda=0.1,
        reg_alpha=0.1,
        n_jobs=4,
        booster='gbtree',
        objective='reg:squarederror',
        random_state=42,
        gamma=0,
        subsample=1,
        colsample_bytree=1,
        verbosity=0
    )

    # Fit the model
    xgb_model.fit(X_train, y_train, verbose=False)

    # Save the trained model
    joblib.dump(xgb_model, 'static/models/xgb_model_pso.pkl')
    print("XGBoost Model with PSO updated successfully!")


def combine_data(df, data):
    if data:
        new_data = pd.DataFrame(data)
        new_data['date'] = pd.to_datetime(new_data['date'])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv('static/data/crypto_data.csv', index=False)
        print("Data updated successfully!")


def update_model():
    df = pd.read_csv('static/data/crypto_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    last_date = df['date'].iloc[-1]

    time_end = datetime.datetime.now()
    time_start = last_date
    time_end_unix = int(time_end.timestamp())
    time_start_unix = int(time_start.timestamp())
    try:
        data = get_bitcoin_price(time_start_unix, time_end_unix)
        combine_data(df, data)
    except:
        print("Error fetching data. Please try again later.")
    if data is None:
        df = pd.read_csv('static/data/crypto_data.csv')

    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    # Define the split dates
    train_start_date = '2016-01-01'
    train_end_date = '2022-12-31'
    test_start_date = '2023-01-01'

    # Split the data into train and test based on year
    train_df = df[(df['date'] >= train_start_date)
                  & (df['date'] <= train_end_date)]
    test_df = df[(df['date'] >= test_start_date)]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_close = scaler.fit_transform(train_df['close'].values.reshape(-1, 1))
    test_close = scaler.transform(test_df['close'].values.reshape(-1, 1))

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    time_step = 15
    X_train, y_train = create_dataset(train_close, time_step)
    X_test, y_test = create_dataset(test_close, time_step)

    update_lstm_model(X_train, y_train)
    update_xgb_model_with_pso(X_train, y_train, X_test, y_test)

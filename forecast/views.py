from django.shortcuts import render
import datetime
import pandas as pd
from .utils import *


def lstm_prediction(request):
    # URL API yang digunakan untuk mendapatkan data harga Bitcoin
    time_end = datetime.datetime.now()
    time_start = time_end - datetime.timedelta(days=30)

    # Konversi waktu ke format timestamp Unix
    time_end_unix = int(time_end.timestamp())
    time_start_unix = int(time_start.timestamp())

    data = get_bitcoin_price(time_start_unix, time_end_unix)
    cols = ["close", "date"]
    df = pd.DataFrame(data, columns=cols)
    df.sort_values(by='date', inplace=True, ascending=True)
    data = df.copy()

    # get prediction date from request
    prediction_days = request.GET.get('prediction_days', 7)
    prediction_days = int(prediction_days)
    if prediction_days > 30:
        # return back with error message
        context = {'error': 'Prediction days must be less than 30 days'}
        return render(request, 'lstm_prediction.html', context)

    predictions = lstm_predict(data, 15, prediction_days)

    # Generate date for the next 7 days
    # add 1 day to get 7 days
    start_date = datetime.datetime.now() + datetime.timedelta(days=1)
    date = pd.date_range(
        start=start_date, periods=prediction_days).tolist()

    # join the prediction with the date
    result = pd.DataFrame({'date': date, 'close': predictions.flatten()})
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')

    predictions = result.to_dict('records')

    context = {'predictions': predictions, 'prediction_days': prediction_days}
    return render(request, 'lstm_prediction.html', context)


def pso_xgboost_prediction(request):
    # URL API yang digunakan untuk mendapatkan data harga Bitcoin
    time_end = datetime.datetime.now()
    time_start = time_end - datetime.timedelta(days=30)

    # Konversi waktu ke format timestamp Unix
    time_end_unix = int(time_end.timestamp())
    time_start_unix = int(time_start.timestamp())

    data = get_bitcoin_price(time_start_unix, time_end_unix)
    cols = ["close", "date"]
    df = pd.DataFrame(data, columns=cols)
    df.sort_values(by='date', inplace=True, ascending=True)
    data = df.copy()

    # get prediction date from request
    prediction_days = request.GET.get('prediction_days', 7)
    prediction_days = int(prediction_days)
    if prediction_days > 30:
        # return back with error message
        context = {'error': 'Prediction days must be less than 30 days'}
        return render(request, 'lstm_prediction.html', context)

    predictions = pso_xgboost_predict(data, 15, prediction_days)

    # Generate date for the next 7 days from now
    # add 1 day to get 7 days
    start_date = datetime.datetime.now() + datetime.timedelta(days=1)
    date = pd.date_range(
        start=start_date, periods=prediction_days).tolist()

    # join the prediction with the date
    result = pd.DataFrame({'date': date, 'close': predictions.flatten()})
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')

    predictions = result.to_dict('records')

    context = {'predictions': predictions, 'prediction_days': prediction_days}
    return render(request, 'pso_xgboost_prediction.html', context)


def dashboard(request):
    df = pd.read_csv('static/data/crypto_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df_timezone = df['date'].dt.tz

    start_date = pd.to_datetime('2024-06-01').tz_localize(df_timezone)
    end_date = pd.to_datetime('2024-07-01').tz_localize(df_timezone)

    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    data = filtered_df.copy()
    filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')

    # ambil 30 hari terakhir sebelum tanggal start_date
    pass_data = df[(df['date'] >= start_date - pd.Timedelta(days=30)) & (
        df['date'] <= start_date
    )].copy()

    lstm_predictions, pso_xgboost_predictions = [], []
    for i in range(0, len(data), 3):  # Iterate until the end of the data
        lstm_predictions.extend(lstm_predict(pass_data, 15, 3).flatten())
        pso_xgboost_predictions.extend(
            pso_xgboost_predict(pass_data, 15, 3).flatten())
        # tambahkan 7 hari ke pass_data
        pass_data = pd.concat([pass_data, data.iloc[i:i+3]], ignore_index=True)

    table_data = pd.DataFrame({
        'date': filtered_df['date'].tolist(),
        'actual_close': filtered_df['close'].values.tolist(),
        'lstm_prediction': lstm_predictions,
        'pso_xgboost_prediction': pso_xgboost_predictions,
    })

    # sub 1 days from end date
    end_date = end_date - pd.Timedelta(days=1)
    # Create a list for dates to be used in the context
    date_range = pd.date_range(
        start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()

    context = {
        'table_data': table_data.to_dict('records'),  # Data for the table
        'actual_close': filtered_df['close'].values.tolist(),
        'date': date_range,
        'lstm_predictions': lstm_predictions,
        'pso_xgboost_predictions': pso_xgboost_predictions
    }
    return render(request, 'index.html', context)

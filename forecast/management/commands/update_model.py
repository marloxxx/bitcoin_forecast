# forecast/management/commands/update_model.py
import datetime
import requests
import tensorflow as tf
import numpy as np
import joblib
import os
from django.core.management.base import BaseCommand
from sklearn.preprocessing import MinMaxScaler
from forecast.utils import get_bitcoin_price, lstm_predict, pso_xgboost_predict, update_model


class Command(BaseCommand):
    help = 'Update LSTM and PSO-XGBoost models with the latest data'

    def handle(self, *args, **kwargs):
        update_model()
        self.stdout.write(self.style.SUCCESS(
            'Successfully updated the models'))

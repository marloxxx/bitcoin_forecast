from django.urls import path
from .views import *

urlpatterns = [
    path('', dashboard, name='dashboard'),
    path('lstm_prediction/', lstm_prediction, name='lstm_prediction'),
    path('pso_xgboost_prediction/', pso_xgboost_prediction,
         name='pso_xgboost_prediction'),
]

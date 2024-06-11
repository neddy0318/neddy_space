!apt-get install google-cloud-sdk

import yfinance as yf
import pandas as pd
import pandas_gbq
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay #주식 예측 시평일만 카운트 해주기 위함
from google.oauth2 import service_account
from google.cloud import bigquery

plt.style.use('fivethirtyeight')



def sendtogbq(corp_name):
    current_datetime = datetime.now().strftime("%Y-%m-%d")
    start_year = datetime.now().year - 5
    start_datetime = datetime.now().replace(year=start_year)
    start_date_str = start_datetime.strftime("%Y-%m-%d")
    corp = yf.download(corp_name, start=start_date_str, end=current_datetime)
    corp['Name'] = str(corp_name)
    corp.columns = ['Open','High','Low','Close','Adj_Close','Volume','Name']

    corp['ds'] = pd.to_datetime(corp.index, format='%Y-%m-%d')
    corp['y'] = corp['Adj_Close']
    corp_train = corp[['ds', 'y']][:-251]

    model_prophet = Prophet(changepoint_prior_scale=0.15, daily_seasonality=True, seasonality_mode='multiplicative', n_changepoints=100, seasonality_prior_scale=0.05)
    model_prophet.fit(corp)

    fcast_time_with_weekends = 365
    corp_forecast = model_prophet.make_future_dataframe(periods=fcast_time_with_weekends, freq='D')

    corp_forecast['exclude_weekend'] = corp_forecast['ds'] + BDay(0)
    corp_forecast = corp_forecast[corp_forecast['ds'].dt.weekday < 5]
    corp_forecast = model_prophet.predict(corp_forecast)
    model_prophet.plot(corp_forecast, xlabel='Date', ylabel='adj price($)')
    plt.show()

    corp_test = corp[-251:]
    future = corp_test[['ds']]
    forecast = model_prophet.predict(future)

    y_pred = forecast['yhat'].values
    y_true = corp_test['y'].values
    mae = mean_absolute_error(y_true, y_pred)
    print('MAE: %.3f' % mae)
    mse = mean_squared_error(y_true, y_pred)
    print('MSE: %.3f' % mse)

    rmse = np.sqrt(mse)
    print('RMSE: %.3f' % rmse)

    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('Predictability Test')
    plt.legend()
    plt.show()

    client = bigquery.Client()

    corp_forecast = corp_forecast[['ds','yhat_lower', 'yhat_upper', 'yhat']]
    corp_pred = pd.merge(corp, corp_forecast, on=['ds'], how='inner')
    corp_pred = corp_pred.sort_values(by='ds')

    table_id = 'instant-bonfire-426101-f8.practical_project.' + corp_name

    try:
        query = f"SELECT * FROM `{table_id}`"
        existing_data = pandas_gbq.read_gbq(query, project_id='instant-bonfire-426101-f8')
    except Exception as e:
        print(f"Error reading table: {e}")
        existing_data = pd.DataFrame()

    if not existing_data.empty:
        corp_pred = corp_pred[~corp_pred.apply(tuple, 1).isin(existing_data.apply(tuple, 1))]

    if not corp_pred.empty:
        pandas_gbq.to_gbq(corp_pred, table_id, project_id='instant-bonfire-426101-f8', if_exists='append')

    y_pred_df = pd.DataFrame(y_pred).rename(columns={0: "pred"})
    y_true_df = pd.DataFrame(y_true).rename(columns={0: "true"})
    combined_df = pd.concat([y_pred_df, y_true_df], axis=1)
    table_id_2 = 'instant-bonfire-426101-f8.practical_project.' + corp_name + '_predictability'

    try:
        query_2 = f"SELECT * FROM `{table_id_2}`"
        existing_data_2 = pandas_gbq.read_gbq(query_2, project_id='instant-bonfire-426101-f8')
    except Exception as e:
        print(f"Error reading table: {e}")
        existing_data_2 = pd.DataFrame()

    if not existing_data_2.empty:
        combined_df = combined_df[~combined_df.apply(tuple, 1).isin(existing_data_2.apply(tuple, 1))]

    if not combined_df.empty:
        pandas_gbq.to_gbq(combined_df, table_id_2, project_id='instant-bonfire-426101-f8', if_exists='append')

    return corp_pred




sendtogbq('MSFT')
sendtogbp('AAPL')

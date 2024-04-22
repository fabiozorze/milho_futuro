import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from ML import Feature_eng_modules as eg
from ML import Modules as fm
from ML import extra_modules as exm
import time
from datetime import datetime
import pickle
from sklearn.preprocessing import StandardScaler
import order_module as om
import warnings

# Suppress specific RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")

# Open the .pkl file in binary mode
with open('features_selected.pkl', 'rb') as f:
    # Load the object from the file
    features_selected = pickle.load(f)


# Open the .pkl file in binary mode
with open('rf_clf.pkl', 'rb') as f:
    # Load the object from the file
    model = pickle.load(f)


global connection
connection = False
if not mt5.initialize():
    connection = False
    print("initialize() fail")
    mt5.shutdown()
connection = True
print(connection)



if __name__ == '__main__':

    ### Dados para armazenar
    data_hour = pd.DataFrame()
    data_dollar = pd.DataFrame()
    data_pred = pd.DataFrame()

    dt = datetime.now()
    on = True
    get_diff = dt.hour-5
    dt_hour = datetime.now().hour

    data_minute = pd.DataFrame()

    first_dollarbar = exm.dollar_bar_startday()

    while on == True:

        dt = datetime.now()
        if (dt.hour >= 9) & (dt.minute >= 0):


            dt_hour_new = datetime.now().hour

            if (dt.minute == 0) & (get_diff != dt_hour_new):
                dt = datetime.now()
                dt_minute = datetime.now().hour
                get_diff = dt.hour
                print(dt)

                data_minute_new = exm.get_ohlc('CCM$', mt5.TIMEFRAME_M1, n=70000)

                data = data_minute_new.resample('H').agg({
                                                            'open': 'first',
                                                            'high': 'max',
                                                            'low': 'min',
                                                            'close': 'last',
                                                            'real_volume': 'sum'
                                                        }).dropna().iloc[:-1]

                data['volume'] = data['close'] * data['real_volume']
                del data['real_volume']

                data['v_mean'] = data['volume'].rolling(240).mean()

                dollar_bar, trades = exm.generate_dollar_bars_frequency(data, mean=240, frequency='yes')


                windows = [1,2,3,5,9,15]
                main, return_outcomes, binary_outcomes_main = exm.features_target(dollar_bar,windows=windows)
                del main['ibs']
                main_1 = main[features_selected]

                ml_data = pd.concat([main_1, binary_outcomes_main['return_3']], axis=1).fillna(method='ffill').dropna()
                features = ml_data.iloc[::, :-1]
                binary_outcomes = ml_data.iloc[::, -1:]

                ### Scale
                scaler = StandardScaler()
                X_train_data_scaled = scaler.fit_transform(features)
                ### Model Run
                pred = pd.DataFrame(model.predict_proba(X_train_data_scaled))


                pred['short'] = np.where(pred[0] > 0.90, -1, 0)
                pred['long'] = np.where(pred[1] > 0.90, 1, 0)
                pred['pred'] = pred['long'] + pred['short']
                pred.index = features.index

                print(str(datetime.now()) + '  RODOU TUDO')

                print(first_dollarbar.index[-1])
                print(pred.index[-1])

                if pred.index[-1] != first_dollarbar.index[-1]:

                    try:
                        positions = mt5.positions_get(symbol='CCMK24')[0][9]
                    except:
                        positions = 0

                    first_dollarbar = dollar_bar.copy()

                    print(pred['pred'].iloc[-2])
                    print(pred['pred'].iloc[-1])

                    if pred['pred'].iloc[-1] != pred['pred'].iloc[-2]:

                        if positions == 0:
                            if (pred['pred'].iloc[-1] == 1):
                                mt5.order_send(om.Buy(symbol='CCMK24',position_size=1))
                            else:
                                mt5.order_send(om.Sell(symbol='CCMK24',position_size=1))

                        else:
                            if (pred['pred'].iloc[-1] == 1) & (pred['pred'].iloc[-2] == -1):
                                mt5.order_send(om.close(symbol='CCMK24'))
                                mt5.order_send(om.Buy(symbol='CCMK24',position_size=1))

                            elif (pred['pred'].iloc[-1] == -1) & (pred['pred'].iloc[-2] == 1):
                                mt5.order_send(om.close(symbol='CCMK24'))
                                mt5.order_send(om.Sell(symbol='CCMK24',position_size=1))

                            elif (pred['pred'].iloc[-1] == 0):
                                mt5.order_send(om.close(symbol='CCMK24'))


                ### Dados resample hour
                novos_dados = data[~data.index.isin(data_hour.index)]
                data_hour = pd.concat([data_hour, novos_dados])
                ### Dados dollar_bar
                novos_dados_dollar = dollar_bar[~dollar_bar.index.isin(data_dollar.index)]
                data_dollar = pd.concat([data_dollar, novos_dados_dollar])
                ### Dados pred
                novos_dados_pred = pred[~pred.index.isin(data_pred.index)]
                data_pred = pd.concat([data_pred, novos_dados_pred])

            if (dt.hour >= 19) & (dt.minute >= 0):
                on = False

                ### Dados resample hour
                novos_dados = data[~data.index.isin(data_hour.index)]
                data_hour = pd.concat([data_hour, novos_dados])
                ### Dados dollar_bar
                novos_dados_dollar = dollar_bar[~dollar_bar.index.isin(data_dollar.index)]
                data_dollar = pd.concat([data_dollar, novos_dados_dollar])
                ### Dados pred
                novos_dados_pred = pred[~pred.index.isin(data_pred.index)]
                novos_dados_pred = pd.concat([data_pred, novos_dados_pred])

                data_hour.to_csv(r'C:\Users\Fabio\PycharmProjects\milho_futuro\dados_save\data_hour.csv')
                data_dollar.to_csv(r'C:\Users\Fabio\PycharmProjects\milho_futuro\dados_save\data_dollar.csv')
                novos_dados_pred.to_csv(r'C:\Users\Fabio\PycharmProjects\milho_futuro\dados_save\novos_dados_pred.csv')

                print('break')
                break

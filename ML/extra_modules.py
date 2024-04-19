import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from ML import Feature_eng_modules as eg
from ML import Modules as fm
import shap

def get_ohlc(ativo, timeframe, n=None):
    ativo = mt5.copy_rates_from_pos(ativo, timeframe, 0, n)
    ativo = pd.DataFrame(ativo)
    ativo = ativo.rename(columns={'time': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'})
    ativo['date'] = pd.to_datetime(ativo['date'], unit='s')
    ativo.set_index('date', inplace=True)

    return ativo


def ma_strategy (data):

    data['ret'] = data['close'] - data['close'].shift(1)
    data['ma'] = data.close.rolling(9).mean()
    data['ma_sinal'] = np.sign(data['ma'].pct_change())

    data['ma_st'] = data['ma_sinal'] * data['ret'].shift(-1)

    return data



def generate_dollar_bars_frequency(data, mean=None, frequency=None):

    trades = data.copy()

    trades['Dollar_Volume_mean'] = (trades['volume'].rolling(mean).mean())

    if frequency != None:
        # the average amount we want to sample from
        trades['frequency'] = trades.close / trades.close.shift(mean)
        sample_frequency = trades['frequency']
        # the thresholds (which is series) of amounts that need to be crossed before a new bars is formed based on the times we want to sample per day
        trades['Dollar_Volume_mean'] = trades['Dollar_Volume_mean'] * sample_frequency

    times = trades.index
    prices = trades['close'].values
    volumes = trades['volume'].values
    ans = np.zeros(shape=(len(prices), 6))
    dates = []
    candle_counter = 0
    dollars = 0
    lasti = 0
    for i in range(len(prices)):
        dollars += volumes[i]
        if dollars >= trades['Dollar_Volume_mean'].iloc[i]:
            dates.append(times[i])                   # time
            ans[candle_counter][1] = prices[lasti]                     # open
            ans[candle_counter][2] = np.max(prices[lasti:i + 1])       # high
            ans[candle_counter][3] = np.min(prices[lasti:i + 1])       # low
            ans[candle_counter][4] = prices[i]                         # close
            ans[candle_counter][5] = np.sum(volumes[lasti:i + 1])      # volume
            candle_counter += 1
            lasti = i + 1
            dollars = 0

    dollar_bar = pd.DataFrame(ans)
    dollar_bar = dollar_bar.iloc[::, 1:]
    dollar_bar = dollar_bar.loc[(dollar_bar != 0).any(axis=1)]

    dollar_bar.index = dates
    dollar_bar.columns = ['open', 'high', 'low', 'close', 'volume']

    return dollar_bar,trades


def features_target(data,windows=None):

    #windows = [1,3,5,9,15,30,40,80]
    data = ma_strategy(data)
    features = eg.add_and_save_features_ALL(data, windows)
    features = fm.return_lag(features)
    features = fm.MA(features)
    features.drop(['open', 'high', 'low', 'close', 'volume','ret','ma','ma_sinal','ma_st'], axis=1, inplace=True)

    return_outcomes, binary_outcomes = fm.target_outcomes(data)

    return features, return_outcomes, binary_outcomes

def features_target_modelo(data):

    windows = [1,3,5,9,15,30,40,80]
    data = ma_strategy(data)
    features = eg.add_and_save_features_ALL(data, windows)
    features = fm.return_lag(features)
    features = fm.MA(features)
    features.drop(['open', 'high', 'low', 'close', 'volume','ret','ma','ma_sinal','ma_st'], axis=1, inplace=True)

    return_outcomes, binary_outcomes = fm.target_outcomes(data)

    return features, return_outcomes, binary_outcomes


def shap_selection(shap_values):
    # Calculate absolute SHAP values
    abs_shap_values = np.abs(shap_values)

    # Aggregate SHAP values across all samples
    aggregated_shap_values = np.mean(abs_shap_values, axis=0)

    # Select top-k features with the highest aggregated SHAP values
    #top_k_features = np.argsort(aggregated_shap_values)[::-1][:k]

    # Alternatively, select features above a threshold
    threshold = 0.2  # Adjust threshold as needed
    selected_features = np.where(aggregated_shap_values > threshold)[0]

    return selected_features, top_k_features


def dollar_bar_startday():

    data_minute_new = get_ohlc('CCM$', mt5.TIMEFRAME_M1, n=70000)

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

    dollar_bar, trades = generate_dollar_bars_frequency(data, mean=240, frequency='yes')

    return dollar_bar

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle
from sklearn import preprocessing
pd.options.mode.chained_assignment = None

RANDOM_STATE = 835

# ML MODELS
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    VotingClassifier,
    # StackingClassifier,
)
import shap
from sklearn.metrics import precision_score


def target_outcomes(data):
    # CALCULATE TARGET OUTCOMES
    return_outcomes = pd.DataFrame(index=data.index)
    binary_outcomes = pd.DataFrame(index=data.index)

    periods = [1, 2, 3, 4, 5, 10, 20, 30, 45, 60,80, 120, 240]

    for p in (periods):
        print(p)
        return_outcomes[f"return_{p}"] = data.close.pct_change(p)
        binary_outcomes[f"return_{p}"] = return_outcomes[f"return_{p}"].apply(np.sign)

    return return_outcomes, binary_outcomes



# R#RADNDOM FOREST
def run_rf_model_2(X_train_, y_train_, params = None):
    if not params:

        rf_clf_ = RandomForestClassifier(
            criterion="entropy",
            max_depth=7,
            class_weight="balanced_subsample",
            n_estimators=2000,
            random_state=RANDOM_STATE,
            oob_score=True,
            n_jobs=-1,
        )

    else:
        rf_clf_ = RandomForestClassifier(
            criterion="entropy",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            oob_score=True,
            n_jobs=-1,
            **params
        )

    rf_clf_.fit(X_train_, y_train_)

    return rf_clf_


def return_lag(df):

    data = df
    lags = 10
    cols = []
    for lag in range(1, lags + 1):
        col = f'lag_{lag}'
        data[col] = data['close'].pct_change().shift(lag)
        cols.append(col)

    #data.dropna(inplace=True)

    return data


def MA (df):

    data = df
    lags = [3,6,9]
    cols = []
    for lag in lags:
        col = f'ma_{lag}'
        data[col+'mean'] = data.close.rolling(lag).mean().pct_change().shift(1)
        cols.append(col)

    return data


def ma_strate_return (data, retorno=None):

    if retorno == 'points':
        data['return'] = data['Close'] - data['Close'].shift(1)
    else:
        data['return'] = data['Close'].pct_change()

    data['ma'] = data['Close'].rolling(9).mean()
    data['sinal'] = np.sign(data['ma'] - data['ma'].shift(1))
    data['st_ret'] = data['sinal'] * data['return'].shift(-1)

    return data['st_ret']

def report(data):

    wealth_index = (1+data).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks)/previous_peaks

    return print(drawdown.min()), print(wealth_index[-2])


def report_2(data):

    wealth_index = (1+data).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks)/previous_peaks

    return drawdown.min(), wealth_index[-2]



def calculate_adx(data, window=14):
    """
    Calculate the Average Directional Index (ADX) for a given dataset.

    Parameters:
        - data: DataFrame containing 'high', 'low', and 'close' columns.
        - window: Window size for calculating the ADX (default: 14).

    Returns:
        - A new DataFrame with 'adx' column containing ADX values.
    """

    # Calculate the True Range (TR)
    data['tr1'] = data['High'] - data['Low']
    data['tr2'] = np.abs(data['High'] - data['Close'].shift())
    data['tr3'] = np.abs(data['Low'] - data['Close'].shift())
    data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)

    # Calculate the Directional Movement (DM) and Directional Index (DI)
    data['up_move'] = data['High'] - data['High'].shift()
    data['down_move'] = data['Low'].shift() - data['Low']
    data['plus_dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
    data['minus_dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)
    data['plus_di'] = 100 * (data['plus_dm'].rolling(window=window).sum() / data['tr'].rolling(window=window).sum())
    data['minus_di'] = 100 * (data['minus_dm'].rolling(window=window).sum() / data['tr'].rolling(window=window).sum())

    # Calculate the Average Directional Index (ADX)
    data['dx'] = 100 * (np.abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di']))
    data['adx'] = data['dx'].rolling(window=window).mean()

    return data


def calculate_atr(data, period=None):
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values

    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    atr = np.zeros_like(tr)
    atr[period - 1] = np.mean(tr[:period])

    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def shap_imp(clf, X):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    fi0 = np.abs(shap_values[0]).mean(axis=0)
    fi1 = np.abs(shap_values[1]).mean(axis=0)
    fi = fi0 + fi1
    imp = pd.DataFrame({"feature": X.columns.tolist(), "mean": fi})
    imp = imp.set_index("feature")

    return imp

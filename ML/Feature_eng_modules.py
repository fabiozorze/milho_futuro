import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm, tqdm_notebook  # progress bar
from pprint import pprint
import statsmodels.api as sm

import numpy_ext as npx


def time_resample(data,period):

    aggregate = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    return data.resample(f"{period}Min").agg(aggregate).dropna()

# FEATURES CALCULATION
def add_log_return (df: pd.DataFrame, column: str)  -> pd.DataFrame:
    df[f"log_{column}_return"] = np.log(df[column]).diff()
    return df

#######################

def internal_bar_strengh(df: pd.DataFrame) -> float:
    return (df.close - df.low) / (df.high - df.low)

def add_internal_bar_strength(df: pd.DataFrame) -> pd.DataFrame:
    df['ibs'] = internal_bar_strengh(df)
    return df

########################

#array = np.array(teste['close'])

def aqr_momentum(array: np.array)  -> float:
    returns = np.diff(np.log(array))  # .diff()
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = stats.linregress(x, returns)
    ((1 + slope) ** 252) * (rvalue ** 2)

def add_aqr_momentum(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    df[f"aqr_momo_{column}_{window}"] = npx.rolling_apply(
        aqr_momentum, window, df[column].values, n_jobs=10
    )
    return df


def aqr_momo_numba(array: np.ndarray) -> float:
    y = np.diff(np.log(array))
    x = np.arange(y.shape[0])
    A = np.column_stack((x, np.ones(x.shape[0])))
    model, resid = np.linalg.lstsq(A, y, rcond=None)[:2]
    r2 = 1 - resid / (y.size * y.var())
    return (((1 + model[0]) ** 252) * r2)[0]


def add_aqr_momentum_numba(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    df[f"aqr_momo_{column}_{window}"] = npx.rolling_apply(
        aqr_momo_numba, window, df[column].values, n_jobs=10
    )
    return df


#################

def np_racorr(array: np.ndarray, window: int, lag: int) -> np.ndarray:
    """
    rolling autocorrelation
    """
    return npx.rolling_apply(
        lambda array, lag: sm.tsa.acf(array, nlags=lag, fft=True)[lag],
        window,
        array,
        lag=lag,
        n_jobs=10,
    )

def add_rolling_autocorr(
    df: pd.DataFrame, column: str, window: int, lag: int
) -> pd.DataFrame:
    log_changes_array = np.log(df[column]).diff().values
    df[f"racorr_{column}_{window}"] = np_racorr(log_changes_array, window, lag)
    return df

def relative_strength_index(df: pd.DataFrame, n: int) -> pd.Series:
    """
    Calculate Relative Strength Index(RSI) for given data.
    https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, "high"] - df.loc[i, "high"]
        DoMove = df.loc[i, "low"] - df.loc[i + 1, "low"]
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(round(PosDI * 100.0 / (PosDI + NegDI)), name="RSI_" + str(n))
    return RSI

def add_rsi(df: pd.DataFrame, column: str = "close", window: int = 14) -> pd.DataFrame:
    out = df.reset_index()
    rsi = relative_strength_index(out, window)
    df[f"rsi_{column}_{window}"] = pd.Series(data=rsi.values, index=df.index)
    return df

def add_volatility(
    df: pd.DataFrame, column: str = "close", window: int = 10
) -> pd.DataFrame:
    returns = df[column].pct_change()
    df[f"rvol_{column}_{window}"] = returns.rolling(window).std()
    return df


#################
def add_acceleration(
    df: pd.DataFrame, column: str = "close", window: int = 10
) -> pd.DataFrame:
    return_diff = df[column].pct_change().diff()
    df[f"racc_{column}_{window}"] = return_diff.rolling(
        window
    ).std()  # standard deviation of second deriv aka acceleration
    return df


#################
def add_rolling_bands(
    df: pd.DataFrame, column: str, dist: int, window: int
) -> pd.DataFrame:
    upper = df[column] + dist * df[column].rolling(window).std()
    lower = df[column] - dist * df[column].rolling(window).std()

    df[f"upper_band_{column}"] = upper
    df[f"lower_band_{column}"] = lower
    return df


#################

# for some reason njit is generating zerodivision errors whereas numpy is not
def numba_vwap(
    avg: np.ndarray, v: np.ndarray, idx: np.ndarray, len_df: int, window: int
) -> np.ndarray:
    n = np.shape(np.arange(len_df - window))[0]
    A = np.empty((n, 2))
    for i in np.arange(len_df - window):
        tmp_avg = avg[i : i + window]
        tmp_v = v[i : i + window]
        aa = np.sum(tmp_v * tmp_avg) / np.sum(tmp_v)
        jj = idx[i + window]
        A[i, 0] = jj
        A[i, 1] = aa
    return A


def numpy_vwap(
    avg: np.ndarray, v: np.ndarray, idx: np.ndarray, len_df: int, window: int
) -> np.ndarray:
    n = np.shape(np.arange(len_df - window))[0]
    A = np.empty((n, 2))
    for i in tqdm(np.arange(len_df - window)):
        tmp_avg = avg[i : i + window]
        tmp_v = v[i : i + window]
        aa = np.sum(tmp_v * tmp_avg) / np.sum(tmp_v)
        jj = idx[i + window]
        A[i, 0] = jj
        A[i, 1] = aa
    return A


def add_rolling_vwap(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    v = df.volume.values
    avg = df[column].values
    idx = df.index.asi8
    # A = numba_vwap(avg, v, idx, len(df), window)
    A = numpy_vwap(avg, v, idx, len(df), window)
    outdf = (
        pd.DataFrame(A, columns=["index", f"rvwap_{window}"])
        .assign(datetime=lambda df: pd.to_datetime(df["index"], unit="ns"))
        .drop("index", axis=1)
        .set_index("datetime")
    )
    try:
        df = df.join(outdf, how="left")
    except:
        outdf.index = outdf.index.tz_localize('UTC')
        df = df.join(outdf, how="left")

    return df

column = 'low'
#################
def add_rolling_min(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    array = npx.rolling_apply(np.min, window, df[column].values, n_jobs=10)
    df[f"rmin_{column}_{window}"] = array
    return df


def add_rolling_max(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    array = npx.rolling_apply(np.max, window, df[column].values, n_jobs=10)
    df[f"rmax_{column}_{window}"] = array
    return df

#################
def add_average_price(df: pd.DataFrame) -> pd.DataFrame:
    df["average_price"] = (df.high + df.low + df.close + df.open) / 4
    return df


#################


def get_slope(array: np.ndarray) -> float:
    returns = np.diff(np.log(array))
    x = np.arange(len(returns))
    slope, _, rvalue, _, _ = stats.linregress(x, returns)
    return slope

def get_slope_numba(array: np.ndarray) -> float:
    y = np.diff(np.log(array))
    # y = y[~np.isnan(y)]
    x = np.arange(y.shape[0])
    A = np.column_stack((x, np.ones(x.shape[0])))
    model, resid = np.linalg.lstsq(A, y, rcond=None)[:2]
    return model[0]

def add_slope_column(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    df[f"slope_{column}_{window}"] = npx.rolling_apply(
        get_slope, window, df[column].values, n_jobs=10
    )
    return df

def add_slope_column_numba(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    df[f"slope_{column}_{window}"] = npx.rolling_apply(
        get_slope_numba, window, df[column].values, n_jobs=1
    )
    return df

#################
def custom_percentile(array: np.ndarray) -> float:
    if (array.shape[0] - 1) == 0:
        return np.nan
    return (array[:-1] > array[-1]).sum() / (array.shape[0] - 1)


def add_custom_percentile(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    df[f"rank_{column}_{window}"] = npx.rolling_apply(
        custom_percentile, window, df[column].values, n_jobs=5
    )
    return df


def add_and_save_features(data_dic, data_frequency, periods):
    """
    function to create features according to frequency and periods and save to
    hdf store. store must already be created.

    # Args
        data: dict, keys are frequency, values are dataframes
        data_frequency: str, one of the data frequencies from the data dict
        periods: dict, keys are period labels, values are the integers
        hdf_filepath: pathlib or str object
        rank_window: int to divide window by for ranking features

            """
    #log_errors = []
    #pprint(periods)

    df = data_dic[data_frequency].copy()

    for key, window in tqdm(periods.items()):
        tqdm._instances.clear()
        try:
            df = (
                df.pipe(add_average_price)
                .pipe(add_rolling_vwap, column="average_price", window=window)
                .pipe(
                    add_rolling_bands, column=f"rvwap_{window}", dist=2, window=window
                )
                .pipe(add_internal_bar_strength)
                .pipe(add_rolling_min, column="low", window=window)
                .pipe(add_rolling_max, column="high", window=window)
                .pipe(
                    add_slope_column_numba,
                    column=f"lower_band_rvwap_{window}",
                    window=window,
                )
                .pipe(
                    add_slope_column_numba,
                    column=f"upper_band_rvwap_{window}",
                    window=window,
                )
                .pipe(
                    add_slope_column_numba, column=f"rmin_low_{window}", window=window
                )
                .pipe(
                    add_slope_column_numba, column=f"rmax_high_{window}", window=window
                )
                .pipe(add_acceleration, column="close", window=window)
                .pipe(add_aqr_momentum_numba, column="close", window=window)
                .pipe(add_acceleration, column="average_price", window=window)
                .pipe(add_aqr_momentum_numba, column="average_price", window=window)
                .pipe(add_acceleration, column=f"rvwap_{window}", window=window)
                .pipe(add_acceleration, column="close", window=window)
                .pipe(add_acceleration, column="average_price", window=window)
                .pipe(add_aqr_momentum_numba, column=f"rvwap_{window}", window=window)
                .pipe(add_volatility, column="close", window=window)
                .pipe(add_volatility, column="average_price", window=window)
                .pipe(add_rsi, column="close", window=window)
                .pipe(add_rsi, column="average_price", window=window)
                .pipe(add_rolling_autocorr, column="close", window=window, lag=1)
                .pipe(add_rolling_autocorr, column="average_price", window=window, lag=1
                )
            )
        except:
            continue

    return df


def add_and_save_features(data, windows):
    """
    function to create features according to frequency and periods and save to
    hdf store. store must already be created.

    # Args
        data: dict, keys are frequency, values are dataframes
        data_frequency: str, one of the data frequencies from the data dict
        periods: dict, keys are period labels, values are the integers
        hdf_filepath: pathlib or str object
        rank_window: int to divide window by for ranking features

            """
    df = data
    for window in windows:

        try:
            df = (
                df.pipe(add_average_price)
                .pipe(add_rolling_vwap, column="average_price", window=window)
                #.pipe(
                    #add_rolling_bands, column=f"rvwap_{window}", dist=2, window=window
                #)
                .pipe(add_internal_bar_strength)
                .pipe(add_rolling_min, column="low", window=window)
                .pipe(add_rolling_max, column="high", window=window)
                #.pipe(
                    #add_slope_column_numba,
                    #column=f"lower_band_rvwap_{window}",
                    #window=window,
                #)
                #.pipe(
                    #add_slope_column_numba,
                    #column=f"upper_band_rvwap_{window}",
                    #window=window,
                #)
                .pipe(add_acceleration, column="close", window=window)
                #.pipe(add_aqr_momentum_numba, column="close", window=window)
                .pipe(add_acceleration, column="average_price", window=window)
                #.pipe(add_aqr_momentum_numba, column="average_price", window=window)
                #.pipe(add_acceleration, column=f"rvwap_{window}", window=window)
                .pipe(add_acceleration, column="close", window=window)
                .pipe(add_acceleration, column="average_price", window=window)
                #.pipe(add_aqr_momentum_numba, column=f"rvwap_{window}", window=window)
                .pipe(add_volatility, column="close", window=window)
                .pipe(add_volatility, column="average_price", window=window)
                .pipe(add_rsi, column="close", window=window)
                .pipe(add_rsi, column="average_price", window=window)
                .pipe(add_rolling_autocorr, column="close", window=window, lag=1)
                .pipe(add_rolling_autocorr, column="average_price", window=window, lag=1
                )
            )
        except:
            continue

    return df


def add_and_save_features_ALL(data, windows):
    """
    function to create features according to frequency and periods and save to
    hdf store. store must already be created.

    # Args
        data: dict, keys are frequency, values are dataframes
        data_frequency: str, one of the data frequencies from the data dict
        periods: dict, keys are period labels, values are the integers
        hdf_filepath: pathlib or str object
        rank_window: int to divide window by for ranking features

            """
    df = data
    for window in windows:


        try:
            df = (
                df.pipe(add_average_price)
                .pipe(add_rolling_vwap, column="average_price", window=window)
                .pipe(
                    add_rolling_bands, column=f"rvwap_{window}", dist=2, window=window
                )
                .pipe(add_internal_bar_strength)
                .pipe(add_rolling_min, column="low", window=window)
                .pipe(add_rolling_max, column="high", window=window)
                .pipe(
                    add_slope_column_numba,
                    column=f"lower_band_rvwap_{window}",
                    window=window,
                )
                .pipe(
                    add_slope_column_numba,
                    column=f"upper_band_rvwap_{window}",
                    window=window,
                )
                .pipe(add_acceleration, column="close", window=window)
                .pipe(add_aqr_momentum_numba, column="close", window=window)
                .pipe(add_acceleration, column="average_price", window=window)
                .pipe(add_aqr_momentum_numba, column="average_price", window=window)
                .pipe(add_acceleration, column=f"rvwap_{window}", window=window)
                .pipe(add_acceleration, column="close", window=window)
                .pipe(add_acceleration, column="average_price", window=window)
                .pipe(add_aqr_momentum_numba, column=f"rvwap_{window}", window=window)
                .pipe(add_volatility, column="close", window=window)
                .pipe(add_volatility, column="average_price", window=window)
                .pipe(add_rsi, column="close", window=window)
                .pipe(add_rsi, column="average_price", window=window)
                .pipe(add_rolling_autocorr, column="close", window=window, lag=1)
                .pipe(add_rolling_autocorr, column="average_price", window=window, lag=1
                )
            )
        except:
            continue

    return df


def add_and_save_features_ALL_crypto(data, windows):
    """
    function to create features according to frequency and periods and save to
    hdf store. store must already be created.

    # Args
        data: dict, keys are frequency, values are dataframes
        data_frequency: str, one of the data frequencies from the data dict
        periods: dict, keys are period labels, values are the integers
        hdf_filepath: pathlib or str object
        rank_window: int to divide window by for ranking features

            """
    df = data
    for window in windows:


        try:
            df = (
                df.pipe(add_average_price)
                .pipe(add_rolling_vwap, column="average_price", window=window)
                .pipe(
                    add_rolling_bands, column=f"rvwap_{window}", dist=2, window=window
                )
                .pipe(add_internal_bar_strength)
                .pipe(add_rolling_min, column="low", window=window)
                .pipe(add_rolling_max, column="high", window=window)
                .pipe(
                    add_slope_column_numba,
                    column=f"upper_band_rvwap_{window}",
                    window=window,
                )
                .pipe(add_acceleration, column="close", window=window)
                .pipe(add_acceleration, column="average_price", window=window)
                .pipe(add_acceleration, column=f"rvwap_{window}", window=window)
                .pipe(add_acceleration, column="close", window=window)
                .pipe(add_acceleration, column="average_price", window=window)
                .pipe(add_volatility, column="close", window=window)
                .pipe(add_volatility, column="average_price", window=window)
                .pipe(add_rsi, column="close", window=window)
                .pipe(add_rsi, column="average_price", window=window)
                .pipe(add_rolling_autocorr, column="close", window=window, lag=1)
                .pipe(add_rolling_autocorr, column="average_price", window=window, lag=1
                )
            )
        except:
            continue

    return df

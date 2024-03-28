import pandas as pd
import numpy as np

import shap

from sklearn.preprocessing import scale

data_base = pd.read_csv(r'C:\Users\Fabio\PycharmProjects\fundo_zorze\Data\data_brapi\data_brapi.csv')

data = data_base[data_base['ticker'] == 'VALE3']

# Calculate the adjustment factor based on the difference between Close and Adj Close
adjustment_factor = data['adjustedClose'] / data['close']

# Adjust OHLC prices
data['open'] *= adjustment_factor
data['high'] *= adjustment_factor
data['low'] *= adjustment_factor

data = data[['date','open','high','low','adjustedClose','volume']]
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.resample('M').last()

data.rename(columns={'adjustedClose': 'close'}, inplace=True)

ret = data.close.pct_change()

return_outcomes, binary_outcomes = target_outcomes(data)

windows = [3,5,10]

data = add_and_save_features(data, windows)
data = return_lag(data)
data = MA (data)

data = data.loc['2015':]



data.drop(['open','high','low','close','volume'], axis=1, inplace=True)

data = data.fillna(method='ffill')

### TRAINA AND TEST DATA
X_train = data.loc['2015-01-01':'2018-11'].dropna()
Y_train = binary_outcomes['return_1'].loc['2015-01-01':'2018-11'].dropna()

X_train = preprocessing.scale(X_train)

rf_clf = run_rf_model_2(X_train, Y_train, params=None)


### TRAINA AND TEST DATA
X_test = data.loc['2018-11':'2019-12'].dropna()
Y_test = binary_outcomes['return_1'].loc['2018-11':'2019-12'].dropna()

X_test = preprocessing.scale(X_test)

pred = pd.DataFrame(rf_clf.predict_proba(X_test))

pred.index = binary_outcomes['return_1'].loc['2018-11':'2019-12'].index

pred['ret'] = ret.loc['2018-11':'2019-12']
pred = pred.iloc[1:]

pred['st'] = np.where(pred[1] > 0.60, pred['ret'].shift(-1)*1, pred['ret'].shift(-1)*0)


pred['st'].cumsum().plot()
pred['ret'].cumsum().plot()

pred['ret'].cumsum()
pred['st'].cumsum()


nan_count = data.isna().sum()

all_pred = pd.DataFrame()
asset_list = []
lixo = []

for asset in list(selec.ticker):

    try:
        data = data_base[data_base['ticker'] == asset]

        # Calculate the adjustment factor based on the difference between Close and Adj Close
        adjustment_factor = data['adjustedClose'] / data['close']

        # Adjust OHLC prices
        data['open'] *= adjustment_factor
        data['high'] *= adjustment_factor
        data['low'] *= adjustment_factor

        data = data[['date','open','high','low','adjustedClose','volume']]
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        data = data.resample('M').last()

        data.rename(columns={'adjustedClose': 'close'}, inplace=True)

        ret = data.close.pct_change()

        return_outcomes, binary_outcomes = target_outcomes(data)

        windows = [3,5,10]

        data = add_and_save_features(data, windows)
        data = return_lag(data)
        data = MA (data)

        del data['ibs']

        data = data.loc['2015':]

        data.drop(['open','high','low','close','volume'], axis=1, inplace=True)

        data = data.fillna(method='ffill')

        ### TRAINA AND TEST DATA
        X_train = data.loc['2018-01-01':'2020-10'].dropna()
        Y_train = binary_outcomes['return_1'].loc['2018-01-01':'2020-10'].dropna()

        X_train = preprocessing.scale(X_train)

        rf_clf = run_rf_model_2(X_train, Y_train, params=None)


        ### TRAINA AND TEST DATA
        X_test = data.loc['2020-11':'2021-12'].dropna()
        Y_test = binary_outcomes['return_1'].loc['2020-11':'2021-12'].dropna()

        X_test = preprocessing.scale(X_test)

        pred = pd.DataFrame(rf_clf.predict_proba(X_test))
        pred.index = ret.loc['2020-11':'2021-12'].index

        all_pred = pd.concat([all_pred, pred[1]],axis =1)
        asset_list.append(asset)
    except:
        lixo.append(asset)
        continue

all_pred.columns = asset_list

top_n_columns = 10


# Find the top N columns with the highest values for each row
top_columns_by_row = all_pred.apply(lambda row: row.nlargest(top_n_columns).index.tolist(), axis=1)
top_columns_by_row = top_columns_by_row.iloc[1:]

ret_list = []


for i in range(len(top_columns_by_row)):

    ret = assets_return.loc[top_columns_by_row.index[i+1]][top_columns_by_row.iloc[i]].mean()
    ret_list.append(ret)

a = pd.DataFrame(top_columns_by_row)

unique_strings_count = len(set([item for sublist in a[0] for item in sublist]))

r = pd.DataFrame(ret_list).cumsum().plot()

(pd.DataFrame(ret_list)+1).cumprod().plot()





assets_return = pd.read_csv(r'C:\Users\Fabio\PycharmProjects\fundo_zorze\Data\data_brapi\data_close.csv')
assets_return['date'] = pd.to_datetime(assets_return['date'])
assets_return = assets_return.set_index('date')

assets_return = (assets_return.resample('M').last()).pct_change()











################## SHAP


for year in years:

    all_pred = pd.DataFrame()
    asset_list = []
    lixo = []

    for asset in list(selec.ticker):

        try:
            data = data_base[data_base['ticker'] == asset]

            # Calculate the adjustment factor based on the difference between Close and Adj Close
            adjustment_factor = data['adjustedClose'] / data['close']

            # Adjust OHLC prices
            data['open'] *= adjustment_factor
            data['high'] *= adjustment_factor
            data['low'] *= adjustment_factor

            data = data[['date','open','high','low','adjustedClose','volume']]
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
            data = data.resample('M').last()

            data.rename(columns={'adjustedClose': 'close'}, inplace=True)

            ret = data.close.pct_change()

            return_outcomes, binary_outcomes = target_outcomes(data)

            windows = [3,5,10]

            data = add_and_save_features(data, windows)
            data = return_lag(data)
            data = MA (data)

            del data['ibs']0

            data = data.loc['2015':]

            data.drop(['open','high','low','close','volume'], axis=1, inplace=True)

            data = data.fillna(method='ffill')

            ### TRAINA AND TEST DATA
            X_train = data.loc['2015-01-01':'2017-10'].dropna()
            Y_train = binary_outcomes['return_4'].loc['2015-01-01':'2017-10'].dropna()

            X_train = pd.DataFrame(scale(X_train), columns=X_train.columns)

            rf_clf = run_rf_model_2(X_train, Y_train, params=None)

            explainer = shap.TreeExplainer(rf_clf)
            shap_values = explainer.shap_values(X_train)

            # Handle the case when shap_values is a list with multiple arrays
            if isinstance(shap_values, list):
                shap_values = np.mean(np.abs(shap_values), axis=0)

            # Calculate mean absolute SHAP values
            mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

            threshold = 0.02

            # Select features based on threshold
            selected_features = X_train.columns[mean_abs_shap_values > threshold]


            X_train = X_train[selected_features]

            rf_clf = run_rf_model_2(X_train, Y_train, params=None)


            ### TRAINA AND TEST DATA
            X_test = data.loc['2017-11':'2018-12'].dropna()
            Y_test = binary_outcomes['return_4'].loc['2017-11':'2018-12'].dropna()

            X_test = pd.DataFrame(scale(X_test), columns=X_test.columns)

            X_test = X_test[selected_features]

            pred = pd.DataFrame(rf_clf.predict_proba(X_test))
            pred.index = binary_outcomes['return_4'].loc['2017-11':'2018-12'].index

            all_pred = pd.concat([all_pred, pred[1]],axis =1)
            asset_list.append(asset)
        except:
            lixo.append(asset)
            continue

    all_pred.columns = asset_list

    top_n_columns = 10


    # Find the top N columns with the highest values for each row
    top_columns_by_row = all_pred.apply(lambda row: row.nlargest(top_n_columns).index.tolist(), axis=1)
    top_columns_by_row = top_columns_by_row.iloc[1:]

    ret_list = []


    for i in range(len(top_columns_by_row)):

        ret = assets_return.loc[top_columns_by_row.index[i+1]][top_columns_by_row.iloc[i]].mean()
        ret_list.append(ret)

    a = pd.DataFrame(top_columns_by_row)




unique_strings_count = len(set([item for sublist in a[0] for item in sublist]))

r = pd.DataFrame(ret_list).cumsum().plot()

(pd.DataFrame(ret_list)+1).cumprod().plot()



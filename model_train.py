from ML import Feature_eng_modules as eg
from ML import Modules as fm
import pandas as pd
import numpy as np
import shap
import MetaTrader5 as mt5
from sklearn.preprocessing import scale, StandardScaler
from sklearn import preprocessing
from ML import extra_modules as exm
import pickle

import matplotlib
matplotlib.use('TkAgg')


if not mt5.initialize():
    print("initialize() fail")
    mt5.shutdown()


data_all = exm.get_ohlc('CCM$', mt5.TIMEFRAME_H1, n=70000)
data_all = data_all[['open','high','low','close','real_volume']]
data_all['real_volume'] = data_all['real_volume'] * data_all['close']
data_all.columns = ['open','high','low','close','volume']


dollar_bar, trades = exm.generate_dollar_bars_frequency(data_all, mean=240, frequency='yes')
main, return_outcomes, binary_outcomes_main = exm.features_target_modelo(dollar_bar)
del main['ibs']


ml_data = pd.concat([main,binary_outcomes_main['return_3']],axis=1).fillna(method='ffill').dropna().iloc[-440:]
features = ml_data.iloc[::,:-1]
binary_outcomes = ml_data.iloc[::,-1:]


X_train_data = features
Y_train_data = binary_outcomes['return_3']

Y_train_data = np.where(Y_train_data == 0, -1, Y_train_data)

#X_train_data_outro = pd.read_csv(r'C:\Users\Fabio\PycharmProjects\milho_futuro\veai.csv',parse_dates=True,index_col=[0])

### Scale
scaler = StandardScaler()
X_train_data_scaled = scaler.fit_transform(X_train_data)

### Model Train
rf_clf = fm.run_rf_model_2(X_train_data_scaled, Y_train_data, params=None)

### Calculate SHAP values
X_train_data_scaled_df = pd.DataFrame(X_train_data_scaled, columns=X_train_data.columns)
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_train_data_scaled_df)
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

# Select features with mean absolute SHAP values higher than 0.25
selected_features = np.where(mean_abs_shap_values > 0.03)

features_selected = X_train_data_scaled_df.columns[np.unique(selected_features[1])]

position = np.unique(selected_features[1])

scaler = StandardScaler()
X_train_data_scaled = scaler.fit_transform(X_train_data.iloc[:, position])

rf_clf = fm.run_rf_model_2(X_train_data_scaled, Y_train_data, params=None)

with open('rf_clf.pkl', 'wb') as file:
    pickle.dump(rf_clf, file)

### lista indicadores selecionados
with open('features_selected.pkl', 'wb') as file:
    pickle.dump(features_selected, file)



data_all = exm.get_ohlc('CCM$', mt5.TIMEFRAME_H1, n=70000)
data_all = data_all[['Open','High','Low','Close','real_volume']]
data_all['real_volume'] = data_all['real_volume'] * data_all['Close']
data_all.columns = ['open','high','low','close','volume']



def features_target(data):

    windows = [1,3,5,9,15,30,40,80]
    data = ma_strategy(data)
    features = eg.add_and_save_features_ALL(data, windows)
    features = fm.return_lag(features)
    features = fm.MA(features)
    features.drop(['open', 'high', 'low', 'close', 'volume','ret','ma','ma_sinal','ma_st'], axis=1, inplace=True)

    return_outcomes, binary_outcomes = fm.target_outcomes(data)

    return features, return_outcomes, binary_outcomes

dollar_bar, trades = exm.generate_dollar_bars_frequency(data_all, mean=240, frequency='yes')

main, return_outcomes, binary_outcomes_main = exm.features_target(dollar_bar)

del main['ibs']



#### WITH SHAP SELECTION

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



ml_data = pd.concat([main,binary_outcomes_main['return_3']],axis=1).fillna(method='ffill').dropna()
#ml_data = ml_data.loc['2015':].fillna(method='ffill').dropna()
features = ml_data.iloc[::,:-1]
binary_outcomes = ml_data.iloc[::,-1:]


LKBK = 440

prediction_data = pd.DataFrame()
index_list = []

fea_se = {}

#i = 440

for i in range(LKBK,len(features),220):

    print(i)

    X_train_data = features[i - LKBK:i]
    Y_train_data = binary_outcomes[i - LKBK:i]['return_3']

    #X_train_data = features
    #Y_train_data = binary_outcomes['return_3']


    Y_train_data = np.where(Y_train_data == 0,-1,Y_train_data)

    ### Scale
    scaler = StandardScaler()
    X_train_data_scaled = scaler.fit_transform(X_train_data)


    ### Model Train
    rf_clf = fm.run_rf_model_2(X_train_data_scaled, Y_train_data, params=None)

    ### Calculate SHAP values
    X_train_data_scaled_df = pd.DataFrame(X_train_data_scaled, columns=X_train_data.columns)
    explainer = shap.TreeExplainer(rf_clf)
    shap_values = explainer.shap_values(X_train_data_scaled_df)
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Select features with mean absolute SHAP values higher than 0.25
    selected_features = np.where(mean_abs_shap_values > 0.03)

    features_selected = X_train_data_scaled_df.columns[np.unique(selected_features[1])]
    fea_se[i] = features_selected

    position = np.unique(selected_features[1])

    scaler = StandardScaler()
    X_train_data_scaled = scaler.fit_transform(X_train_data.iloc[:, position])

    rf_clf = fm.run_rf_model_2(X_train_data_scaled, Y_train_data, params=None)


    ### Model Test

    index_data = features.index[i:i+220]
    X_test_data = pd.DataFrame(features[i:i+220])#220

    X_test_data = scaler.transform(X_test_data.iloc[:, position])
    pred = pd.DataFrame(rf_clf.predict_proba(X_test_data))

    index_list.append(index_data)
    prediction_data = pd.concat([prediction_data, pred])

prediction_data.index = features.loc['2020-03-04 16:00:00':].index


pred = prediction_data.copy()

pred['short'] = np.where(pred[0] > 0.50, -1, 0)
pred['long'] = np.where(pred[1] > 0.50, 1, 0)
pred['pred'] = pred['long'] + pred['short']

pred['close'] = data_all['close'].loc['2020-03-04 16:00:00':]
pred['ret'] = pred['close'] - pred['close'].shift(1)

pred = exm.ma_strategy(pred)

pred['st_return'] = pred['pred'] * pred['ret'].shift(-1)

pred['st_return'].cumsum().plot()
pred['ma_st'].cumsum().plot()


import matplotlib.pyplot as plt


plt.plot(pred['st_return'].cumsum())
plt.show()

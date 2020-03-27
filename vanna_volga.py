import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from Black_Scholes import BlackScholes
import SABR_params as SABR


def calculate_weight(underlying, mtr, ATM_vol, K, K1, K2, K3):
    b_s = BlackScholes(underlying, 0, 0, mtr)
    vega_K = b_s.BS_vega(K, ATM_vol)
    vega_K_1 = b_s.BS_vega(K1, ATM_vol)
    vega_K_2 = b_s.BS_vega(K2, ATM_vol)
    vega_K_3 = b_s.BS_vega(K3, ATM_vol)

    weight_1 = (vega_K * np.log(K2 / K) * np.log(K3 / K)) / (vega_K_1 * np.log(K2 / K1) * np.log(K3 / K1))
    weight_2 = (vega_K * np.log(K / K1) * np.log(K3 / K)) / (vega_K_2 * np.log(K2 / K1) * np.log(K3 / K2))
    weight_3 = (vega_K * np.log(K / K1) * np.log(K / K2)) / (vega_K_3 * np.log(K3 / K1) * np.log(K3 / K2))
    return [weight_1, weight_2, weight_3]

def calculate_price(underlying, mtr, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3):
    weight_list = calculate_weight(underlying, mtr, ATM_vol, K, K1, K2, K3)
    b_s = BlackScholes(underlying, 0, 0, mtr)
    BS_price_K = b_s.BS_call(K, ATM_vol)
    BS_price_K1 = b_s.BS_call(K1, ATM_vol)
    BS_price_K3 = b_s.BS_call(K3, ATM_vol)
    price = BS_price_K + weight_list[0] * (mkt_price_K1 - BS_price_K1) + weight_list[2] * (mkt_price_K3 - BS_price_K3)
    return price

def get_ATM_strike(strike_list, underlying_price):
    ATM_strike = strike_list[0]
    strike_diff = abs(ATM_strike - underlying_price)
    for x in strike_list:
        if abs(x - underlying_price) < strike_diff:
            ATM_strike = x
            strike_diff = abs(x - underlying_price)
    return ATM_strike

def get_delta25_call(asset_df):

    call_vol = asset_df[asset_df['CallPut'] == 'C']
    call_vol['delta25_c'] = abs(call_vol['m_delta'] - 0.25)
    c_strike = call_vol.loc[ call_vol[call_vol['delta25_c'] == min(call_vol['delta25_c'])].index,'Strike']

    underlying_price = np.mean(call_vol['underlying_price'])
    strike_list = list(call_vol['Strike'])
    ATM_strike = get_ATM_strike(strike_list, underlying_price)


    put_vol = asset_df[asset_df['CallPut'] == 'P']
    if not (float(c_strike) == ATM_strike):
        strike_list.remove(float(c_strike))
        strike_list.remove(ATM_strike)
        temp_strike_list = set(put_vol['Strike']).intersection(strike_list)
        put_vol = put_vol[put_vol['Strike'].isin(temp_strike_list)]
        put_vol['delta25_p'] = abs(put_vol['m_delta'] - (-0.25))
        try:
            p_strike = put_vol.loc[put_vol[put_vol['delta25_p'] == min(put_vol['delta25_p'])].index, 'Strike']
        except ValueError:
            print('No delta25 strike')
            p_strike = 1e7

        if int(p_strike) >= ATM_strike:
            print('wrong delta25p data, for Date:' + call_vol.iloc[0,2] +' Maturity:' + call_vol.iloc[0,3])
            result_df = pd.DataFrame()
        else:
            result_df = call_vol[call_vol['Strike'].isin([int(p_strike), int(c_strike), ATM_strike])]
            result_df = result_df.reset_index()
    else:
        print('lack call market price')
        result_df = pd.DataFrame()
    return result_df

def vanna_volga_delta(underlying, mtr, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3):
    price_1 = calculate_price(underlying + 1, mtr, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3)
    price_2 = calculate_price(underlying -1, mtr, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3)
    delta = (price_1 - price_2) / 2
    return delta

def vanna_volga_gamma(underlying, mtr, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3):
    delta_1 = vanna_volga_delta(underlying + 1, mtr, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3)
    delta_2 = vanna_volga_delta(underlying - 1, mtr, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3)
    gamma = (delta_1 - delta_2) / 2
    return gamma

def vanna_volga_theta(underlying, mtr, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3):
    price_1 = calculate_price(underlying , mtr * 1.01 , ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3)
    price_2 = calculate_price(underlying, mtr * 0.99, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3)
    theta =  - (price_1 - price_2) / (0.02 * mtr)
    return theta

def vanna_volga_vega(underlying, mtr, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3):
    price_1 = calculate_price(underlying , mtr, ATM_vol * 1.01, K, K1, K2, K3, mkt_price_K1, mkt_price_K3)
    price_2 = calculate_price(underlying , mtr, ATM_vol * 0.99, K, K1, K2, K3, mkt_price_K1, mkt_price_K3)
    vega = (price_1 - price_2) / (0.02 * ATM_vol)
    return vega

def cal_ttm(row):
    tradetime = datetime.datetime.strptime(row['Date'], '%Y-%m-%d') + datetime.timedelta(hours= 16)
    mtr = datetime.datetime.strptime(row['Maturity'], '%Y-%m-%d') + datetime.timedelta(hours= 8)
    ttm = (mtr - tradetime).total_seconds() / (365 * 24 * 60 * 60)
    return ttm

historical_vol = pd.read_csv('D:\\SABR model/vol_data/SABR_model_param_4.csv')
historical_vol['mtr'] = historical_vol.apply(cal_ttm, axis= 1)
historical_group = historical_vol.groupby(['Date','Maturity'])
result_df = pd.DataFrame()
for name, asset_df in historical_group:
    asset_df = asset_df.replace(-1, np.nan)
    asset_df = asset_df.dropna()
    asset_call_df = asset_df[asset_df['CallPut'] == 'C']
    if asset_df.empty:
        continue
    delta25_call_df = get_delta25_call(asset_df)
    if delta25_call_df.empty:
        continue

    k = np.array(asset_call_df['Strike'])
    strike_list = list(asset_call_df['Strike'])

    vv_vol_list = []
    for K in strike_list:
        t = np.mean(asset_call_df['mtr'])
        f = np.mean(asset_call_df['underlying_price'])
        ATM_vol = delta25_call_df.loc[1, 'm_mid_vol']
        K1, K2, K3 = delta25_call_df.loc[0, 'Strike'], delta25_call_df.loc[1, 'Strike'], delta25_call_df.loc[2, 'Strike']
        mkt_price_K1 = delta25_call_df.loc[0, 'm_mid'] * delta25_call_df.loc[0, 'underlying_price']
        mkt_price_K3 = delta25_call_df.loc[2, 'm_mid'] * delta25_call_df.loc[0, 'underlying_price']
        weight_list = calculate_weight(f, t, ATM_vol, K, K1, K2, K3)
        price = calculate_price(f,t, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3)
        delta = vanna_volga_delta(f, t, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3)

        b_s = BlackScholes(f,0,0,t)
        vv_vol = b_s.BS_impliedVol(K, price, 'C')
        vv_vol_list.append(vv_vol)
    asset_call_df['VV_vol'] = vv_vol_list
    result_df = result_df.append(asset_call_df)
result_df.to_csv('D:\\SABR model/vol_data/SABR_VV_model_param.csv')





vol_fit = [SABR.lognormal_vol(k_, f, t, param[0], 1, param[1], param[2]) * 100 for k_ in strike_list]
#vol_surface_df.loc[t, strike_list] = vol_fit
plt.figure(figsize=(6, 6))
fig = plt.figure(1)
ax1 = plt.subplot(111)
ax1.plot(strike_list, vol_fit, label = 'SABR')
ax1.plot(strike_list, vv_vol_list, label = 'Vanna-Volga')
ax1.scatter(k, mkt_vol, label = 'mkt_vol')
ax1.legend(loc='best')
ax1.set_xlabel('Strike')
ax1.set_title(str(datetime.date(2019, 10, 1)) + ' Vol Curve, Maturity :' + str('2019-10-11'))
plt.show()






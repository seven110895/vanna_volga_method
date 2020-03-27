import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from Black_Scholes import BlackScholes
import SABR_params as SABR
import vanna_volga as VV

historical_vol = pd.read_csv("D:/optionmaster/Straddle/Straddle_vol/ALL_MTR_STRIKE_V2.csv")
one_day_vol = historical_vol[historical_vol['Date'] == str(datetime.date(2019, 10, 1))]
asset_df = one_day_vol[one_day_vol['Maturity'] == '2019-10-11']
asset_df = asset_df.replace(-1, np.nan)
asset_df = asset_df.dropna()
asset_call_df = asset_df[asset_df['CallPut'] == 'C']
delta25_call_df = VV.get_delta25_call(asset_df)



k = np.array(asset_call_df['Strike'])
mkt_vol = np.array(asset_call_df['m_mid_vol'])
t = np.mean(asset_call_df['mtr'])
f = np.mean(asset_call_df['underlying_price'])
param = SABR.fit(f, t, 1, k, mkt_vol)
strike_list = np.arange(min(k), max(k) + 100, 50)

vv_vol_list = []
for K in strike_list:
    t = np.mean(asset_call_df['mtr'])
    f = np.mean(asset_call_df['underlying_price'])
    ATM_vol = delta25_call_df.loc[1, 'm_mid_vol']
    K1, K2, K3 = delta25_call_df.loc[0, 'Strike'], delta25_call_df.loc[1, 'Strike'], delta25_call_df.loc[2, 'Strike']
    mkt_price_K1 = delta25_call_df.loc[0, 'm_mid'] * delta25_call_df.loc[0, 'underlying_price']
    mkt_price_K3 = delta25_call_df.loc[2, 'm_mid'] * delta25_call_df.loc[0, 'underlying_price']
    weight_list = VV.calculate_weight(f, t, ATM_vol, K, K1, K2, K3)
    price = VV.calculate_price(f,t, ATM_vol, K, K1, K2, K3, mkt_price_K1, mkt_price_K3)
    b_s = BlackScholes(f,0,0,t)
    vv_vol = b_s.BS_impliedVol(K, price, 'C')
    vv_vol_list.append(vv_vol)

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

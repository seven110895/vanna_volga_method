import numpy as np
from scipy.optimize import minimize
import pandas as pd
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from Black_Scholes import BlackScholes

def lognormal_vol(k, f, t, alpha, beta, rho, volvol):
    #Hegan's paper log normal implied volatility, equation 2.17a
    if k <= 0 or f <= 0:
        return 0
    eps = 1e-07
    logfk = np.log(f / k)
    fkbeta = (f*k)**(1 - beta)
    a = (1 - beta) ** 2 * alpha ** 2 / (24 * fkbeta)
    b = 0.25 * rho * beta * volvol * alpha / fkbeta ** 0.5
    c = (2 - 3 * rho ** 2) * volvol ** 2 / 24
    d = fkbeta ** 0.5
    v = (1 - beta) ** 2 * logfk ** 2 / 24
    w = (1 - beta) ** 4 * logfk ** 4 / 1920
    z = volvol * fkbeta ** 0.5 * logfk / alpha
    if abs(z) > eps:
        vz = alpha * z * (1 + (a + b + c) * t) / (d * (1 + v + w) * _x(rho, z))
        return vz
        # if |z| <= eps
    else:
        v0 = alpha * (1 + (a + b + c) * t) / (d * (1 + v + w))
        return v0

def _x(rho, z):
    a = (1 - 2 * rho * z + z ** 2) ** .5 + z - rho
    b = 1 - rho
    return np.log(a / b)

def alpha(v_atm_ln, f, t, beta, rho, volvol):
    # use ATM lognormal volatility to calibrate the alpha parameter, use equation
    f_ = f ** (beta - 1)
    p = [
        t * f_ ** 3 * (1 - beta) ** 2 / 24,
        t * f_ ** 2 * rho * beta * volvol / 4,
        (1 + t * volvol ** 2 * (2 - 3 * rho ** 2) / 24) * f_,
        -v_atm_ln
    ]
    roots = np.roots(p)
    roots_real = np.extract(np.isreal(roots), np.real(roots))
    alpha_first_guess = v_atm_ln * f ** (1 - beta)
    i_min = np.argmin(np.abs(roots_real - alpha_first_guess))
    return roots_real[i_min]

def ATM_vol(f,t, alpha, beta, rho, volvol):
    f_ = f ** (beta - 1)
    a =  t * f_ ** 3 * (1 - beta) ** 2 / 24 * (alpha **3)
    b = t * f_ ** 2 * rho * beta * volvol / 4 * (alpha **2)
    c = (1 + t * volvol ** 2 * (2 - 3 * rho ** 2) / 24) * f_ * alpha
    v_atm = a + b + c
    return v_atm * 100

def fit(f, t, beta, k, v_sln):
    # calibrate SABR parameters alpha, rho and volvol by the market obeserved k and implied log normal volatility
    def vol_square_error(x):
        vols = [lognormal_vol(k_, f, t, x[0], beta, x[1], x[2]) * 100 for k_ in k]
        return sum((vols - v_sln) **2)
    x0 = np.array([0.01, 0.00, 0.10])
    bounds = [(0.0001, None), (-0.9999, 0.9999), (0.0001, None)]
    res = minimize(vol_square_error, x0, method='SLSQP', bounds=bounds)
    alpha, rho, volvol = res.x
    return [alpha, rho, volvol]

def fit_beta(f, f_new, t, t2, k , v_sln_t1, C_t1, C_t2, C_P_list, v_sln_atm_t1, v_sln_atm_t2):
    beta_list = np.arange(0, 11, 1)/10
    f_diff = f_new - f
    C_diff = C_t2 - C_t1
    min_hedge_error = 1e10
    best_beta = beta_list[0]
    hedge_error_list = []
    for beta in beta_list:
        hedge_error = 0
        param = fit(f, t, beta, k, v_sln_t1)
        alpha_t1 = alpha(v_sln_atm_t1/100, f, t, beta, param[1], param[2])
        alpha_t2 = alpha(v_sln_atm_t2/100, f_new, t2, beta, param[1], param[2])

        f2 = f * 1.01
        b_s = BlackScholes(f, 0, 0, t)
        b_s_2 = BlackScholes(f2, 0, 0, t)
        delta_list = []
        vega_list = []
        for num in range(len(k)):
            s_k = k[num]
            c_p = C_P_list[num]
            if c_p == 'C':
                delta_i = (b_s.BS_call(s_k, 100*lognormal_vol(s_k, f, t, param[0], beta, param[1], param[2])) -
                           b_s_2.BS_call(s_k, 100*lognormal_vol(s_k , f2, t, param[0], beta, param[1], param[2]))) / (f - f2)
                vega_i = (b_s.BS_call(s_k, 100*lognormal_vol(s_k, f, t, alpha_t1, beta, param[1], param[2])) -
                          b_s.BS_call(s_k, 100*lognormal_vol(s_k, f, t, alpha_t1 *1.01, beta, param[1], param[2]))) /\
                         (alpha_t1 - 1.01 * alpha_t1)
            else:
                delta_i = (b_s.BS_put(s_k, 100*lognormal_vol(s_k, f, t, param[0], beta, param[1], param[2])) -
                           b_s_2.BS_put(s_k, 100*lognormal_vol(s_k , f2, t, param[0], beta, param[1], param[2]))) / (f - f2)
                vega_i = (b_s.BS_put(s_k,
                                      100 * lognormal_vol(s_k, f, t, alpha_t1, beta, param[1], param[2])) - b_s.BS_put(
                    s_k, 100 * lognormal_vol(s_k, f, t, alpha_t1 * 1.01, beta, param[1], param[2]))) / (
                                     alpha_t1 - 1.01 * alpha_t1)
            delta_list.append(delta_i)
            vega_list.append(vega_i)
        for i in range(len(C_diff)):
            #sig = (C_diff[i] - delta_list[i] * f_diff ) / C_t2[i]
            sig = (C_diff[i] - delta_list[i] * f_diff - vega_list[i] * (alpha_t2 - alpha_t1)) / C_t2[i]
            hedge_error += sig**2
        hedge_error_list.append(hedge_error)
        if hedge_error < min_hedge_error:
            min_hedge_error = hedge_error
            best_beta = beta
    return best_beta

def get_ATM_strike(strike_list, underlying_price):
    ATM_strike = strike_list[0]
    strike_diff = abs(ATM_strike - underlying_price)
    for x in strike_list:
        if abs(x - underlying_price) < strike_diff:
            ATM_strike = x
            strike_diff = abs(x - underlying_price)
    return ATM_strike

def SABR_delta(f, t, k, CallPut , alpha, beta, rho, volvol):
    f2 = f * 1.01
    b_s = BlackScholes(f * 1.01, 0, 0, t)
    b_s_2 = BlackScholes(f * 0.99, 0, 0, t)
    if CallPut == 'C':
        delta_i = (b_s.BS_call(k, 100 * lognormal_vol(k, f*1.01, t, alpha, beta, rho, volvol)) -
                   b_s_2.BS_call(k, 100 * lognormal_vol(k, f *0.99, t, alpha, beta, rho, volvol))) / (0.02 *f)
    else:
        delta_i = (b_s.BS_put(k, 100 * lognormal_vol(k, f*1.01, t, alpha, beta, rho, volvol)) -
                   b_s_2.BS_put(k, 100 * lognormal_vol(k, f * 0.99, t, alpha, beta, rho, volvol))) / (0.02*f)
    return delta_i

def SABR_delta_2(f,t,k,CallPut, alpha, beta,rho,volvol):
    f2 = f* 1.01
    b_s = BlackScholes(f,0,0,t)
    SABR_vol = lognormal_vol(k,f,t,alpha,beta,rho,volvol)*100
    SABR_vol_shift = lognormal_vol(k,f2,t,alpha,beta,rho,volvol)*100
    bs_delta = b_s.BS_delta(k, SABR_vol,CallPut)
    bs_vega = b_s.BS_vega(k,SABR_vol)
    delta = bs_delta + bs_vega * (SABR_vol_shift -SABR_vol ) / (f2 - f)
    return delta

def SABR_delta_3(f,t,k,CallPut, alpha, beta,rho,volvol):
    f2 = f* 1.01
    b_s = BlackScholes(f,0,0,t)
    SABR_vol = lognormal_vol(k,f,t,alpha,beta,rho,volvol)*100
    SABR_vol_shift = lognormal_vol(k,f2,t,alpha,beta,rho,volvol)*100
    bs_delta = b_s.BS_delta(k, SABR_vol,CallPut)
    bs_vega = b_s.BS_vega(k,SABR_vol)


    SABR_vol_alpha_shift = lognormal_vol(k, f, t, alpha * 1.01, beta, rho, volvol)*100
    alpha_deri = (SABR_vol - SABR_vol_alpha_shift) / (alpha - 1.01 * alpha)
    delta = bs_delta + bs_vega * ((SABR_vol_shift -SABR_vol ) / (f2 - f) + alpha_deri * (rho * volvol / (f ** beta)))

    return delta

def SABR_bs_delta(f,t,k,CallPut, alpha, beta,rho,volvol):
    b_s = BlackScholes(f,0,0,t)
    SABR_vol = lognormal_vol(k,f,t,alpha,beta,rho,volvol)*100
    bs_delta = b_s.BS_delta(k, SABR_vol,CallPut)
    return bs_delta

def SABR_theta(f, t, k, CallPut , alpha, beta, rho, volvol):
    t2 = t * 1.01
    b_s = BlackScholes(f, 0, 0, t)
    b_s_2 = BlackScholes(f, 0, 0, t2)
    if CallPut == 'C':
        theta = -(b_s.BS_call(k, 100 * lognormal_vol(k, f, t, alpha, beta, rho, volvol)) -
                   b_s_2.BS_call(k, 100 * lognormal_vol(k, f, t2, alpha, beta, rho, volvol))) / (t - t2)
    else:
        theta = -(b_s.BS_put(k, 100 * lognormal_vol(k, f, t, alpha, beta, rho, volvol)) -
                   b_s_2.BS_put(k, 100 * lognormal_vol(k, f, t2, alpha, beta, rho, volvol))) / (t - t2)
    return theta

def SABR_vega(f, t, k, alpha, beta, rho, volvol):
    b_s = BlackScholes(f, 0, 0, t)
    atm_vol = ATM_vol(f, t, alpha, beta, rho, volvol)
    strike_vol = 100 * lognormal_vol(k, f, t, alpha, beta, rho, volvol)
    bs_vega = b_s.BS_vega(k, strike_vol)
    vega = bs_vega * strike_vol / atm_vol
    return vega


def SABR_gamma(f, t, k, CallPut , alpha, beta, rho, volvol):
    delta_1 = SABR_delta(f, t, k, CallPut , alpha, beta, rho, volvol)
    delta_2 = SABR_delta(f*1.01, t, k, CallPut , alpha, beta, rho, volvol)
    gamma = (delta_1 - delta_2) / (f - f*1.01)
    return gamma

def SABR_gamma_der(f,t,k,CallPut,alpha,beta,rho,volvol):
    gamma_1 = SABR_gamma(f, t, k, CallPut, alpha, beta, rho, volvol)
    gamma_2 = SABR_gamma(f*1.01, t, k, CallPut, alpha, beta, rho, volvol)
    gamma_der = (gamma_2 - gamma_1) / (0.01 * f)
    return gamma_der

def SABR_vanna(f, t, k, CallPut, alpha, beta, rho, volvol):
    vol_1 = lognormal_vol(k, f, t, alpha, beta, rho, volvol) * 100
    vol_2 = lognormal_vol(k, f, t, alpha, beta, rho*1.01, volvol)*100
    b_s = BlackScholes(f, 0, 0, t)
    if CallPut == 'C':
        vanna = (b_s.BS_call(k, vol_2) - b_s.BS_call(k, vol_1)) / (0.01* rho)
    else:
        vanna = (b_s.BS_put(k, vol_2) - b_s.BS_put(k, vol_1)) / (0.01* rho)
    return vanna

def SABR_volga(f, t, k, CallPut, alpha, beta, rho, volvol):
    vol_1 = lognormal_vol(k, f, t, alpha, beta, rho, volvol) * 100
    vol_2 = lognormal_vol(k, f, t, alpha, beta, rho, volvol * 1.01) * 100
    b_s = BlackScholes(f, 0, 0, t)
    if CallPut == 'C':
        volga = (b_s.BS_call(k, vol_2) - b_s.BS_call(k, vol_1)) / (0.01 * volvol)
    else:
        volga = (b_s.BS_put(k, vol_2) - b_s.BS_put(k, vol_1)) / (0.01 * volvol)
    return volga

def SABR_beta_d(f, t, k, CallPut, alpha, beta, rho, volvol):
    vol_1 = lognormal_vol(k, f, t, alpha, beta, rho, volvol) * 100
    vol_2 = lognormal_vol(k, f, t, alpha, beta + 0.01, rho, volvol)*100
    b_s = BlackScholes(f, 0, 0, t)
    if CallPut == 'C':
        vanna = (b_s.BS_call(k, vol_2) - b_s.BS_call(k, vol_1)) / (0.01* rho)
    else:
        vanna = (b_s.BS_put(k, vol_2) - b_s.BS_put(k, vol_1)) / (0.01* rho)
    return vanna

def SABR_alpha_deri(f, t, k, CallPut, alpha, beta, rho, volvol):
    vol_1 = lognormal_vol(k, f, t, alpha*0.99, beta, rho, volvol) * 100
    vol_2 = lognormal_vol(k, f, t, alpha * 1.01, beta, rho, volvol) * 100
    b_s = BlackScholes(f, 0, 0, t)
    if CallPut == 'C':
        volga = (b_s.BS_call(k, vol_2) - b_s.BS_call(k, vol_1)) / (0.02 * alpha)
    else:
        volga = (b_s.BS_put(k, vol_2) - b_s.BS_put(k, vol_1)) / (0.02 * alpha)
    return volga

def max_SABR_gamma_der(f,f_change,t,k,CallPut,alpha,beta,rho,volvol):
    if f_change < 0:
        f_list = np.arange(f + f_change, f+1, 1)
    else:
        f_list = np.arange(f, f + f_change + 1, 1)
    max_result = 0
    for st in f_list:
        gamma_der = SABR_gamma_der(st, t, k, CallPut, alpha, beta, rho, volvol)
        if abs(gamma_der) > max_result:
            max_result = gamma_der
    return max_result

def alpha_d(f, t, k, CallPut, alpha, beta, rho, volvol):
    alpha_d_1 = SABR_alpha_deri(f,t,k,CallPut,alpha,beta,rho,volvol)
    alpha_d_2 = SABR_alpha_deri(f,t,k,CallPut,alpha * 1.01,beta,rho,volvol)
    alpha_2j = (alpha_d_2 - alpha_d_1) / (0.01 * alpha)
    return alpha_2j

def vanna_d(f, t, k, CallPut, alpha, beta, rho, volvol):
    vanna_1 = SABR_vanna(f,t,k,CallPut,alpha,beta,rho,volvol)
    vanna_2 = SABR_vanna(f,t,k,CallPut,alpha,beta,rho * 1.01,volvol)
    vanna_d = (vanna_2 - vanna_1) / (0.01 * rho)
    return vanna_d

def volga_d(f, t, k, CallPut, alpha, beta, rho, volvol):
    volga_1 = SABR_volga(f,t,k,CallPut,alpha,beta,rho,volvol)
    volga_2 = SABR_volga(f,t,k,CallPut,alpha,beta,rho,volvol * 1.01)
    volga_d = (volga_2 - volga_1) / ( 0.01 * volvol)
    return volga_d




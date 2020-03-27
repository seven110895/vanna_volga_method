import numpy as np
import scipy.stats as stats


# The following is the basics of Black-Scholes model, including pricing for calls and puts, implied volatility by
# Newton-Raphson method, and calculation for delta.
class BlackScholes:
    def __init__(self, underlying, interest_rate, dividend, time_to_maturity):
        self.underlying = underlying
        self.interest_rate = interest_rate
        self.dividend = dividend
        self.time_to_maturity = time_to_maturity

    def BS_call(self, strike, volatility):
        d1 = (np.log(self.underlying / strike) + (
                    self.interest_rate + 0.5 * (volatility ** 2)) * self.time_to_maturity) / (
                         volatility * np.sqrt(self.time_to_maturity))
        d2 = d1 - volatility * np.sqrt(self.time_to_maturity)
        bs_call = self.underlying * np.exp(-self.dividend * self.time_to_maturity) * stats.norm.cdf(
            d1) - strike * np.exp(-self.interest_rate * self.time_to_maturity) * stats.norm.cdf(d2)
        return bs_call

    def BS_put(self, strike, volatility):
        d1 = (np.log(self.underlying / strike) + (
                    self.interest_rate + 0.5 * (volatility ** 2)) * self.time_to_maturity) / (
                         volatility * np.sqrt(self.time_to_maturity))
        d2 = d1 - volatility * np.sqrt(self.time_to_maturity)
        bs_put = strike * np.exp(-self.interest_rate * self.time_to_maturity) * stats.norm.cdf(
            -d2) - self.underlying * np.exp(-self.dividend * self.time_to_maturity) * stats.norm.cdf(-d1)
        return bs_put

    # Call='C', put='P'
    def BS_impliedVol(self, strike, price, call_put):
        min_price = self.BS_minimum_price(strike, call_put)
        vol_error = 10 ** (-6)
        vol0 = 0
        vol1 = 1
        volDiff = abs(vol1 - vol0)
        loop_count = 0
        while volDiff > vol_error:
            if loop_count > 100:
                print('Inefficient method. 2 approximate values: vol_0 = ', vol0, '; vol_1 = ', vol1, '.')
                break
            vol0 = vol1
            d1 = (np.log(self.underlying / strike) + (
                        self.interest_rate + 0.5 * (vol0 ** 2)) * self.time_to_maturity) / (
                             vol0 * np.sqrt(self.time_to_maturity))
            if (d1 ** 2) > (64 * np.log(2)):
                vol0 = -1
                break
            else:
                if call_put == 'C':
                    BS_price = self.BS_call(strike, vol0)
                else:
                    BS_price = self.BS_put(strike, vol0)
                vol1 = vol0 - (BS_price - price) * np.exp(self.dividend * self.time_to_maturity) * np.exp(
                    0.5 * (d1 ** 2)) * np.sqrt(2 * np.pi) / (self.underlying * np.sqrt(self.time_to_maturity))
                volDiff = abs(vol0 - vol1)
                if not (0 < vol0 < 10 and 0 < vol1 < 10):
                    vol0 = -1
                    break
            loop_count += 1
        if price > min_price and vol0 == -1:
            vol0 = self.BS_impliedVol_dichotomy(strike, price, call_put)
        return vol0

    def BS_impliedVol_dichotomy(self,strike, price, call_put):
        c_est = 0
        top = 5
        floor = 0
        sigma = (top + floor)/2
        loop_count = 0
        while abs(price - c_est) > 1e-8:
            if loop_count > 100:
                print('Inefficient method. 2 approximate values: top = ', top, '; floor = ', floor, '.')
                break
            if call_put == 'C':
                c_est = self.BS_call(strike, sigma)
            else:
                c_est = self.BS_put(strike, sigma)
            if price - c_est > 0:
                floor = sigma
                sigma = (sigma + top)/2
            else:
                top = sigma
                sigma = (sigma + floor)/2
            loop_count += 1
        return sigma

    def BS_minimum_price(self,strike,call_put):
        if call_put == "C":
            min_price = self.BS_call(strike, 0.001)
        else:
            min_price = self.BS_put(strike, 0.001)
        return min_price

    def BS_delta(self, strike, vol, call_put):
        d1 = (np.log(self.underlying / strike) + (self.interest_rate + 0.5 * (vol ** 2)) * self.time_to_maturity) / (
                    vol * np.sqrt(self.time_to_maturity))
        if call_put == 'C':
            delta = np.exp(-self.dividend * self.time_to_maturity) * stats.norm.cdf(d1)
        else:
            delta = -np.exp(self.dividend * self.time_to_maturity) * stats.norm.cdf(-d1)
        if vol == -1 :
            delta = -1
        return delta

    def BS_gamma(self, strike, vol):
        d1 = (np.log(self.underlying / strike) + (self.interest_rate + 0.5 * (vol ** 2)) * self.time_to_maturity) / (
                    vol * np.sqrt(self.time_to_maturity))
        gamma = np.exp(-0.5*(d1 ** 2)) / (self.underlying * vol * np.sqrt(2 * np.pi * self.time_to_maturity))
        return gamma

    def BS_vega(self, strike, vol):
        d1 = (np.log(self.underlying / strike) + (self.interest_rate + 0.5 * (vol ** 2)) * self.time_to_maturity) / (
                vol * np.sqrt(self.time_to_maturity))
        vega = np.exp(-0.5*(d1 ** 2)) * self.underlying * np.sqrt( self.time_to_maturity/ (2 *np.pi))
        return vega

    def BS_theta(self,strike, vol, call_put):
        d1 = (np.log(self.underlying / strike) + (self.interest_rate + 0.5 * (vol ** 2)) * self.time_to_maturity) / (
                vol * np.sqrt(self.time_to_maturity))
        d2 = d1 - vol * np.sqrt(self.time_to_maturity)
        d1_derivative = np.exp(-0.5*(d1 ** 2)) / np.sqrt( 2 * np.pi)
        if call_put == 'C':
            theta = - self.underlying * d1_derivative * vol / (2 * np.sqrt(self.time_to_maturity)) + \
                    self.interest_rate * strike * np.exp( - self.interest_rate * self.time_to_maturity) * \
                    stats.norm.cdf(-d2)
        else:
            theta = - self.underlying * d1_derivative * vol / (2 * np.sqrt(self.time_to_maturity)) - \
                    self.interest_rate * strike * np.exp(- self.interest_rate * self.time_to_maturity) * \
                    stats.norm.cdf(-d2)
        return theta

    def BS_vanna(self, strike, vol):
        d1 = (np.log(self.underlying / strike) + (self.interest_rate + 0.5 * (vol ** 2)) * self.time_to_maturity) / (
                vol * np.sqrt(self.time_to_maturity))
        d2 = d1 - vol * np.sqrt(self.time_to_maturity)
        vega = self.BS_vega(strike, vol)
        vanna = - vega * d2 / (self.underlying * np.sqrt(self.time_to_maturity) * vol)
        return vanna

    def BS_volga(self, strike, vol):
        d1 = (np.log(self.underlying / strike) + (self.interest_rate + 0.5 * (vol ** 2)) * self.time_to_maturity) / (
                vol * np.sqrt(self.time_to_maturity))
        d2 = d1 - vol * np.sqrt(self.time_to_maturity)
        vega = self.BS_vega(strike, vol)
        volga_1 = vega * d1 * d2 / vol
        return volga_1

    def BS_gamma_deri(self, strike, vol):
        gamma1 = self.BS_gamma(strike, vol)
        new_underlying = self.underlying * 1.01
        d1 = (np.log(new_underlying / strike) + (self.interest_rate + 0.5 * (vol ** 2)) * self.time_to_maturity) / (
                vol * np.sqrt(self.time_to_maturity))
        gamma2 = np.exp(-0.5 * (d1 ** 2)) / (new_underlying * vol * np.sqrt(2 * np.pi * self.time_to_maturity))
        gamma_deri = (gamma2 - gamma1) / (new_underlying *0.01)
        return gamma_deri







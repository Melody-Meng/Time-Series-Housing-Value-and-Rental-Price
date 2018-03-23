#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import datetime
import matplotlib.pylab as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA



# Generate house purchasing cash flow = down payment + interest + monthly payment + property tax + insurance
def Cashfl_Sale(df, Years, i, downper, n):
    lis = []
    princl = []
    interest = []
    down = df * downper
    monthly = (df-df * downper) * (i/12) * ((1+i/12)**(Years * 12))/((1+i/12)**(Years * 12)-1)
    month = df * (0.001604) + 120 + monthly
    for x in range (n+1):
        if x == 0:
            lis.append(int(down))
            interest.append(0)
            princl.append((df-down))

        else:
            lis.append(int(month))
            interest.append(princl[x-1]* (i/12))
            princl.append(princl[x-1]-monthly+interest[x])
    return lis, princl,interest

def convert(data):
    data = data.convert_objects(convert_numeric=True)
    data.index = pd.to_datetime(data.index)
    return data

def timeseries(data): 
    data = data.values   
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=46)[0]
    forecast, stderr, conf = model_fit.forecast(steps=46,alpha=0.05)
    return forecast

Rent_Data = pd.read_csv('Neighborhood_MedianRentalPrice_1Bedroom.csv')
Sale_Data = pd.read_csv('Neighborhood_Zhvi_1bedroom.csv')

Rent_Data_nyc = Rent_Data[Rent_Data['CountyName'] == 'New York'].dropna(axis = 1)
Sale_Data_nyc = Sale_Data[Sale_Data['CountyName'] == 'New York'].dropna(axis = 1)


# if buy at 2011-10 and sell after 10 years
regionname1 = Sale_Data_nyc['RegionName'].values
regionname2 = Rent_Data_nyc['RegionName'].values
regionname = list(set(regionname1).intersection(regionname2))

npvdata = pd.DataFrame({"buy_npv" : 0,"rent_npv" : 0}, index = [0])
for region in regionname:
    Rent = Rent_Data_nyc[Rent_Data_nyc['RegionName'] == region].dropna().T.tail(74)
    Sale = Sale_Data_nyc[Sale_Data_nyc['RegionName'] == region].dropna().T.tail(74)
    rent = Rent.T.values.tolist()
    b = np.npv(0.01,rent)
    sale_lis,princl,interest = Cashfl_Sale(Sale.values[0], 30.00, 0.04, 0.2, 119)
    sale_price = timeseries(convert (Sale))[-1]
    sale_lis[-1] = float(interest[-1]+princl[-1]-sale_price)

    rent_new = timeseries(convert(Rent))
    rent = np.concatenate([b,rent_new])
    e = np.npv(0.01,sale_lis)
    f = np.npv(0.01,rent)
    
    rent = pd.DataFrame(rent)
    sale = pd.DataFrame(sale_lis)
    data = pd.concat([rent,sale],axis = 1)
    
    df = pd.DataFrame({"buy_npv" : e,"rent_npv" : f}, index = [region])
    npvdata = npvdata.append(df)

npvdata[1:].to_csv("npvdata1b.csv", encoding='utf-8', index = True)



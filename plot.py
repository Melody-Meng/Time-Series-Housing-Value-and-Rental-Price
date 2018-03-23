#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import datetime
import matplotlib.pylab as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA


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
Rent = Rent_Data_nyc[Rent_Data_nyc['RegionName'] == 'Murray Hill'].dropna().T.tail(74)
Sale = Sale_Data_nyc[Sale_Data_nyc['RegionName'] == 'Murray Hill'].dropna().T.tail(74)

# ARIMA simulate data
rent_sim = timeseries(convert(Rent))
idx = pd.date_range('2017-12-01', '2021-09-01', freq = 'MS')
rent_sim = pd.DataFrame(rent_sim, index = idx, columns = [3])
Rentnew = pd.concat([Rent,rent_sim])


#plot rent trend
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

fig0 = plt.figure('Rent Trend')
ax = plt.subplot(111)
ax.plot(Rent.index, Rent.values,linestyle='-', marker='o',label="Rent")
plt.title('Median Rent - Murray Hill - 1B')
plt.show()
fig0.savefig('Median Rent - Murray Hill - 1B.png')

fig1 = plt.figure('Buy Trend')
ax = plt.subplot(111)
ax.plot(Sale.index, Sale.values,linestyle='-', marker='o',label="Buy")
plt.title('Median house price - Murray Hill - 1B')
plt.show()
fig1.savefig('Median house price - Murray Hill - 1B.png')



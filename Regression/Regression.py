import pandas as pd
import quandl
import numpy as np
import math
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HiLowPCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCTChange'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HiLowPCT', 'PCTChange', 'Adj. Volume']]

forecastCol = 'Adj. Close'
df.fillna(-99999, inplace=True)

#change the coefficient to change the distance out of the forecast.
forecastOut = int(math.ceil(0.01*len(df)))

df['label'] = df[forecastCol].shift(-forecastOut)
df.dropna(inplace=True)

x = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

x = preprocessing.scale(x)
#test split means 20% of the data is used in testing... I think.
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size= 0.2)

classifier = LinearRegression()
classifier.fit(x_train, y_train)

accuracy = classifier.score(x_test, y_test)

print('Can predict', str(forecastOut), 'days out with', str(accuracy*100) + '% accuracy.')

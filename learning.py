import numpy as np
import pandas as pd



#Importing the dataset
from main import season, btselem_to_df, plot_data, add_features_from_json, clean_prices, months_to_weeks

years_weeks_df = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
years_weeks_df = pd.concat([y_train[['year_weeks']], years_weeks_df])

years_weeks_df = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
years_weeks_df = pd.concat([y_train[['year_weeks']], years_weeks_df])

data = y_train.assign(date=lambda x: x['year_weeks'].str[:10])
data = data.assign(year=lambda x: x['date'].str[:4])
data = data.assign(month=lambda x: x['date'].str[5:7])
data = data.assign(date=lambda x: x['year_weeks'])
# Importing additional datasets
btselem = btselem_to_df()
btselem = btselem[btselem['date'] >= '2009-12']
btselem = btselem[btselem['date'] <= '2020-12']
btselem = months_to_weeks(btselem, years_weeks_df, ['days_of_arrest'], ['sum'])
# btselem.rename(columns={'date': 'year_weeks'}, inplace=True)

y_train_cols=y_train.columns.tolist()[1:]

prices = pd.read_excel('prices_after_dates.xlsx', index_col=0)
prices = clean_prices(prices)
data = pd.merge(data, prices, on='date', how='left')
data = pd.merge(data, btselem, on='date', how='left')

#features list
featureslist=prices.columns.tolist()[:-1] + ['days_of_arrest']
featureslist.remove('צנונית')
for feat in featureslist:
    data[feat] = data[feat].fillna(data[feat].mean())
features = data.drop(y_train_cols, axis=1)
features = pd.DataFrame(features).reset_index()
features = features.set_index('year_weeks', drop=True)
features = features.drop(['date', 'index', 'month', 'year','צנונית'], axis=1)
y_labels = data[y_train_cols]
# make dictionary of weeks and numbers of weeks
#split to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, y_labels, test_size=0.2, random_state=42)
#linar regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
# print(mean_squared_error(y_test, y_pred)**0.5)


# test the model on the whole dataset
url = 'https://drive.google.com/file/d/1-7VK3dNry2-AYnfRsxMWsOKhHHMTN_ZA/view?usp=sharing'
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
test_indices = pd.read_csv(url, index_col=0)
new_prices=pd.read_excel('Future_veggies.xlsx', index_col=0)
new_prices = clean_prices(new_prices)
new_prices.rename(columns={'date': 'year_weeks'}, inplace=True)
new_btselem = btselem_to_df()
new_btselem = new_btselem[new_btselem['date'] >= '2020-01']
new_btselem = new_btselem[new_btselem['date'] <= '2022-07']
new_btselem = months_to_weeks(new_btselem, years_weeks_df, ['days_of_arrest'], ['sum'])
new_btselem.rename(columns={'date': 'year_weeks'}, inplace=True)

new_data = pd.merge(test_indices, new_prices, on='year_weeks', how='left')
new_data = pd.merge(new_data, new_btselem, on='year_weeks', how='left')
new_data = new_data.set_index('year_weeks')


for feat in featureslist:
    new_data[feat] = new_data[feat].fillna(new_data[feat].mean())
y_pred = lin_reg.predict(new_data)
y_pred = pd.DataFrame(y_pred, columns=y_train_cols, index=new_data.index)

y_pred.to_csv('y_pred.csv')

import numpy as np
import pandas as pd

# Importing the dataset
from main import *

years_weeks_df = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
years_weeks_df = pd.concat([y_train[['year_weeks']], years_weeks_df])

data = y_train.assign(date=lambda x: x['year_weeks'].str[:10])
data = data.assign(month=lambda x: x['date'].str[5:7])
data = data.assign(date=lambda x: x['year_weeks'])
data = data.assign(year=lambda x: x['date'].str[:4])
p=y_train.axes[0][0:524].tolist()
for i in range(0,len(y_train.axes[0])):
    p[i]=np.sin(i%52 * np.pi / 52)
data['sin']=p
y_train_cols = y_train.columns.tolist()[1:]

# Importing additional datasets

btselem = btselem_to_df()
btselem = btselem[btselem['date'] >= '2009-12']
btselem = btselem[btselem['date'] <= '2020-01']
btselem = months_to_weeks(btselem, years_weeks_df, ['days_of_arrest'], ['sum'])

export = pd.read_csv('export.csv')
export = export[export['date'] >= '2010-01']
export = export[export['date'] <= '2020-01']
export = clean_export(export)
feats = export.columns.tolist()
export = months_to_weeks(export, years_weeks_df, feats[:-1], ['sum' for i in range(len(feats) - 1)])

prices = pd.read_excel('prices_after_dates.xlsx', index_col=0)
prices = clean_prices(prices)
# merges the datasets
data = pd.merge(data, prices, on='date', how='left')
data = pd.merge(data, btselem, on='date', how='left')
data = pd.merge(data, export, on='date', how='left')

# features list

featureslist = data.columns.tolist()[191:]
featureslist.remove('צנונית')
for feat in featureslist:
    data[feat] = data[feat].fillna(data[feat].mean())
features = data[['year_weeks'] + featureslist]
features = pd.DataFrame(features).reset_index()
features = features.set_index('year_weeks', drop=True)
features = features.drop(['index'], axis=1)
features = features.drop(['בצלים_export','קלמנטינות_export','תפוחים_export'], axis=1)
y_labels = data[y_train_cols]
# make dictionary of weeks and numbers of weeks
# split to train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, y_labels, test_size=0.1, random_state=42)
# linar regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# lin_reg = LinearRegression()
# lin_reg.fit(X_train, y_train)
# y_pred = lin_reg.predict(X_test)

# print(mean_squared_error(y_test, y_pred)**0.5)


# test the model on the whole dataset
url = 'https://drive.google.com/file/d/1-7VK3dNry2-AYnfRsxMWsOKhHHMTN_ZA/view?usp=sharing'
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
test_indices = pd.read_csv(url, index_col=0)

year_weeks_df_new = pd.read_csv(url)
new_prices = pd.read_excel('Future_veggies.xlsx', index_col=0)
new_prices = clean_prices(new_prices)
new_prices.rename(columns={'date': 'year_weeks'}, inplace=True)
new_dates_features = year_weeks_df_new.assign(year=lambda x: x['year_weeks'].str[:4])
p=new_dates_features.axes[0].tolist()
for i in range(0,len(new_dates_features.axes[0])):
    p[i]=np.sin(i%52 * np.pi / 52)
new_dates_features['sin']=p

new_btselem = btselem_to_df()
new_btselem = new_btselem[new_btselem['date'] >= '2020-01']
new_btselem = new_btselem[new_btselem['date'] <= '2022-07']
new_btselem = months_to_weeks(new_btselem, year_weeks_df_new, ['days_of_arrest'], ['sum'])
new_btselem.rename(columns={'date': 'year_weeks'}, inplace=True)

new_export = pd.read_csv('export.csv')
new_export = new_export[new_export['date'] >= '2020-01']
new_export = new_export[new_export['date'] <= '2022-07']
new_export = clean_export(new_export)
feats = new_export.columns.tolist()
new_export = months_to_weeks(new_export, year_weeks_df_new, feats[:-1], ['sum' for i in range(len(feats) - 1)])
new_export.rename(columns={'date': 'year_weeks'}, inplace=True)


new_data = pd.merge(test_indices, new_prices, on='year_weeks', how='left')
new_data = pd.merge(new_data, new_btselem, on='year_weeks', how='left')
new_data = pd.merge(new_data, new_export, on='year_weeks', how='left')
new_data = pd.merge(new_data, new_dates_features, on='year_weeks', how='left')
new_data = new_data.set_index('year_weeks')
featureslist = new_data.columns.tolist()

for feat in featureslist:
    new_data[feat] = new_data[feat].fillna(new_data[feat].mean())

# y_pred = lin_reg.predict(new_data)
# y_pred = pd.DataFrame(y_pred, columns=y_train_cols, index=new_data.index)
# y_pred.to_csv('y_pred.csv')

#***************************************************************************************************************************
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200,criterion='squared_error', random_state=42, max_depth=70)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# y_pred=regressor.predict(new_data)
# y_pred = pd.DataFrame(y_pred, columns=y_train_cols, index=new_data.index)
# y_pred.to_csv('y_pred.csv')
#***************************************************************************************************************************
# run cross validation to find the best parameters
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['mse'],
    'bootstrap': [True, False]
}

# define the cross validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)
# create the grid search object
grid_search = GridSearchCV(regressor, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
# fit the grid search object to the data
grid_search.fit(X_train, y_train)
# print the best parameters and the best score
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))



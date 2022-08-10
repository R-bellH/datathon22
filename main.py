import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns


def season(month):
    if month in ['03', '04', '05']:
        return "spring"
    elif month in ['06', '07', '08']:
        return "summer"
    elif month in ['09', '10', '11']:
        return "fall"
    elif month in ['12', '01', '02']:
        return "winter"


def remove_items(test_list, item):
    # using list comprehension to perform the task
    res = [i for i in test_list if i != item]
    return res


def json_to_df(json_file_path):
    with open(json_file_path) as json_file:
        name = json_file_path.split('.')[0]
        data = json.load(json_file)
        df = pd.DataFrame(data)
        df = df.assign(date=lambda x: x['time_obs'].str[:7])
        df = df.assign(day=lambda x: x['time_obs'].str[8:10])
        df = df.assign(month=lambda x: x['time_obs'].str[5:7])
        for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            df.replace(month, season(month), inplace=True)
        df = df.assign(year=lambda x: x['time_obs'].str[:4])
        df = df.drop(['time_obs', 'stn_name'], axis=1)
        df.rename(
            {'hmd_rlt': f'{name}_hmd_rlt', 'tmp_air_dry': f'{name}_tmp_air_dry', 'cld_low_cvr': f'{name}_cld_low_cvr',
             'weather_crr': f'{name}_weather_crr', 'weather_past_1': f'{name}_weather_past_1'}, axis=1, inplace=True)
        df = df[['date', 'day', 'month', 'year', f'{name}_hmd_rlt', f'{name}_tmp_air_dry', f'{name}_cld_low_cvr',
                 f'{name}_weather_crr',
                 f'{name}_weather_past_1']]
        df.replace(-9999, np.nan, inplace=True)
        df = df.groupby(['date', 'day', 'month', 'year']).aggregate(np.mean).reset_index()
        df.drop(['day', 'month', 'year'], axis=1, inplace=True)
        df = df.groupby(['date']).aggregate(np.mean).reset_index()
        return df


# make function to add btselem to df

def plot_json_df_features(df, features):
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.where(df['year'] <= '2015')
    plt.figure(figsize=(20, 6))
    for feature in features:
        sns.lineplot(data=df, x='date', y=feature)
    plt.title(features)
    plt.legend()
    plt.show()


def btselem_to_df():
    # read text from txt file
    with open('btselem.txt', encoding='utf-8') as txt:
        text = txt.read()
        lines = text.split('\n')
        lines = [line.split('\t') for line in lines]
        lines = [line for line in lines if line != [''] and line != ['', '']]
        # make dictionary with hebrew month names as keys and numbers as values
        months = {'ינואר': '01', 'פברואר': '02', 'מרץ': '03', 'אפריל': '04', 'מאי': '05', 'יוני': '06', 'יולי': '07',
                  'אוגוסט': '08', 'ספטמבר': '09', 'אוקטובר': '10', 'נובמבר': '11', 'דצמבר': '12'}
        data = {}
        for line in lines:
            if 'שנת' in line[0]:
                year = line[0].split(' ')[1]
            if line[0].isdigit():
                value = int(line[0])
            elif line[0] in ['ינואר', 'פברואר', 'מרץ', 'אפריל', 'מאי', 'יוני', 'יולי', 'אוגוסט', 'ספטמבר', 'אוקטובר',
                             'נובמבר', 'דצמבר']:
                month = months[line[0]]
                data[year + '-' + month] = value
        df = pd.DataFrame.from_dict(data, orient='index', columns=['days_of_arrest'])
        df.index.name = 'date'
        df.reset_index(inplace=True)
        df = df[['days_of_arrest', 'date']]
        return pd.DataFrame(df)


def months_to_weeks(df, year_weeks_df, features, aggregation_methods):
    """
    features: list of features to be aggregated from df
    aggregation_methods: list of aggregation methods to be used for each feature
    """
    cols = features + ['date']
    new_df = pd.DataFrame(columns=cols)
    for entry in year_weeks_df.itertuples():
        week = entry[1]
        first_year_month = week[0:7]
        temp_df = df[df['date'] == first_year_month]
        temp_df = pd.DataFrame(temp_df).values.flatten().tolist()
        i = 0
        if not temp_df:
            temp_df = [np.nan for i in features] + [week]
        else:
            for agg in aggregation_methods:
                if agg == 'sum':
                    temp_df[i] = temp_df[i] / 4
                i += 1
            temp_df[i] = week
        new_df.loc[len(new_df)] = temp_df
    return new_df


def days_to_weeks(df, features=['tmp_air_dry'], aggregation_methods=[np.mean]):
    """
    takes a dataframe with date in some format of (year, month, day) and returns formatted date
    :param df:
    :param features: list of features to be aggregated from df
    :param aggregation_methods: list of aggregation methods to be used for each feature
    :return:
    """
    if features is None:
        features = ['tmp_air_dry']
    df['date'] = pd.to_datetime(df['date'])
    df = df.assign(year=lambda x: x['date'].dt.strftime("%Y-%U"))
    df['date'] = (pd.to_datetime(df['year'] + '0', format='%Y-%W%w').dt.to_period('W-SUN')).astype(str)
    for feature, agg in zip(features, aggregation_methods):
        df[feature] = df.groupby('date')[feature].agg(agg)
    return df


# make function to add tmp and hmd to the dataframe from the json file such that the avg feature for the week is there
def add_features_from_json(df, json_file_path):
    json_df = json_to_df(json_file_path)
    name = json_file_path.split('.')[0]
    json_df = days_to_weeks(json_df, features=[f'{name}_tmp_air_dry', f'{name}_hmd_rlt'],
                            aggregation_methods=[np.mean, np.mean])
    df = pd.merge(df, json_df, on='date', how='left')
    return df


def clean_prices(prices):
    weeks = prices['תאריך'].unique().tolist()
    vegs = prices['שם ירק'].unique().tolist()
    new_df = pd.DataFrame(columns=vegs + ['date'])
    for week in weeks:
        prices_dict_today = {}
        for veg in vegs:
            try:
                prices_dict_today[veg] = prices[(prices['תאריך'] == week) & (prices['שם ירק'] == veg)]['מחיר'].values[0]
            except:
                prices_dict_today[veg] = np.nan
        prices_dict_today['date'] = week
        new_df.loc[len(new_df)] = prices_dict_today
    for feat in new_df.columns.tolist()[:-1]:
        new_df[feat] = new_df[feat].fillna(new_df[feat].mean())
    return new_df

def clean_export(export):
    dates=export['date'].unique().tolist()
    vegs=export['product_code'].unique().tolist()
    vegs= [f'{veg}_export' for veg in vegs]
    new_df=pd.DataFrame(columns=vegs+['date'])
    for date in dates:
        export_dict_today = {}
        for veg in vegs:
            try:
                export_dict_today[veg] = export[(export['date'] == date) & (export['product_code'] == veg.split('_')[0])]['sum'].values[0]
            except:
                export_dict_today[veg] = np.nan
        export_dict_today['date'] = date
        new_df.loc[len(new_df)] = export_dict_today
    for feat in new_df.columns.tolist()[:-1]:
        new_df[feat] = new_df[feat].fillna(new_df[feat].mean())
    return new_df

def plot_data(df, param_food='all', param_district='all', year_range=['2010', '2020']):
    # filter dataframe by year range
    df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    df = df.assign(date=lambda x: x['date'].str[:7])
    plt.figure(figsize=(20, 6))
    for fd in df.columns.tolist():
        if '_' not in fd: continue
        food, district = fd.split('_')
        if param_food != 'all' and param_food != food: continue
        if param_district != 'all' and param_district != district: continue
        curr_df = df[[f'{food}_{district}', 'year', 'date', 'season', 'month']].groupby(['date']).aggregate(
            np.mean).reset_index()
        # represent column in a line on the plot
        sns.lineplot(data=curr_df, x='date', y=f'{food}_{district}')
    plt.title(f'{param_food} in {param_district}')
    plt.show()
    return df


if __name__ == '__main__':
    # #Importing the dataset
    # url='https://drive.google.com/file/d/1-4YpXkd2kIOM5viSRw8g7oOQm8sicciB/view?usp=sharing'
    # url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    # y_train = pd.read_csv(url, index_col=0)
    #
    # url = 'https://drive.google.com/file/d/1-7VK3dNry2-AYnfRsxMWsOKhHHMTN_ZA/view?usp=sharing'
    # url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    # test_indices = pd.read_csv(url, index_col=0)

    years_weeks_df = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv')
    years_weeks_df = pd.concat([y_train[['year_weeks']], years_weeks_df])

    data = y_train.assign(date=lambda x: x['year_weeks'].str[:10])
    data = data.assign(season=lambda x: x['date'].str[5:7])
    for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        data.replace(month, season(month), inplace=True)
    data = data.assign(year=lambda x: x['date'].str[:4])
    data = data.assign(month=lambda x: x['date'].str[5:7])

    # plot_data(data, param_food='חסה', param_district='מחוז הדרום', year_range=['2010', '2012'])

    # Importing additional datasets
    btselem = btselem_to_df()
    btselem = btselem[btselem['date'] >= '2009-12']
    btselem = btselem[btselem['date'] <= '2020-12']
    btselem = months_to_weeks(btselem, years_weeks_df, ['days_of_arrest'], ['sum'])
    btselem.rename(columns={'date': 'year_weeks'}, inplace=True)
    # btselemplot = btselem_to_df()
    # btselemplot = btselemplot[btselemplot['date'] >= '2009-12']
    # btselemplot = btselemplot[btselemplot['date'] <= '2012-12']
    # plt.figure(figsize=(20, 6))
    # plt.plot(btselemplot['date'], btselemplot['days_of_arrest'])
    # plt.title('Days of arrest')
    # plt.show()

    # add btselem to dataframe
    data = pd.merge(data, btselem, on='year_weeks', how='left')
    df = data

    # # add meteorology data to the dataframe
    # for json_path in ['HaZafon.json', 'telaviv.json', 'jerusalem-west-bank.json']:
    #     df = add_features_from_json(df, json_path)

    # add prices to the dataframe
    prices = pd.read_excel('prices_after_dates.xlsx', index_col=0)
    prices = clean_prices(prices)
    df = pd.merge(df, prices, on='date', how='left')


    # add export data to the dataframe
    export = pd.read_csv('export.csv')
    export = clean_export(export)
    feats= export.columns.tolist()
    export = months_to_weeks(export, years_weeks_df, feats[:-1], ['sum' for i in range(len(feats) - 1)])
    export.rename(columns={'date': 'year_weeks'}, inplace=True)
    df = pd.merge(df, export, on='year_weeks', how='left')
    df = df.drop(['year_weeks'], axis=1)
    # fillna
    for feat in df.columns.tolist()[:-1]:
        df[feat] = df[feat].fillna(df[feat].mean())
    print(df.head())



    # url = 'https://drive.google.com/file/d/1-07zZ5oLAJZDck0WLMGVONV62jgTtxzZ/view?usp=sharing'
    # url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    # unprocessed_data_train = pd.read_csv(url, index_col=0)
    #
    # # plot the data by col over date
    # df = unprocessed_data_train.assign(date=lambda x: x['date'].str[:-3])
    # df = df.assign(season=lambda x: x['date'].str[-2:])
    # df = df.assign(year=lambda x: x['date'].str[:4])
    # for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
    #     df.replace(month, season(month), inplace=True)
    #
    # food = 'other'
    # df = df[df['very_simple_food'] == food]
    # df = df[df['district'] == 'מחוז הצפון']
    # df = df.where(df['year'] <= '2015')
    # plt.figure(figsize=(20, 6))
    # sns.lineplot(data=df, x='date', y='quantity(kg)', hue='season')
    # plt.show()
    # plt.title(food)

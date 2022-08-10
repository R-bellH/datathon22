import pandas as pd
import numpy as np


# switch low values with 0
def zeroing_52_averages(df):
    # add column with 0.75 quantile of each row
    df['median'] = df.median(axis=1)
    # for each row, replace values with 0 if they are less than median
    for i in range(len(df)):
        for j in range(len(df.columns)):
            if df.iloc[i, j] < df.iloc[i]['median']:
                df.iloc[i, j] = 0
            else:
                df.iloc[i, j] = df.iloc[i, j]*1.15
    # drop median column
    df = df.drop(['median'], axis=1)

    return df


if __name__ == '__main__':
    df = pd.read_csv('y_train_52_averages.csv', index_col=0)
    df = zeroing_52_averages(df)
    df.to_csv('y_52_averages_zeroed.csv')

    df = pd.read_csv('y_pred.csv', index_col=0)
    df = zeroing_52_averages(df)
    df.to_csv('y_pred_zeroed.csv')

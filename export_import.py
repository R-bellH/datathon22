import pandas as pd
import os
import matplotlib.pyplot as plt


# create dataframe from list of zip files, containing txt files
def create_dataframe(list_of_zip_files):
    df = pd.DataFrame()
    # append each file to df
    for file in list_of_zip_files:
        # create dataframe from txt files in zip file
        df_zip = pd.read_csv(file, sep='    ', header=None)
        # add dataframe to df
        df = df.append(df_zip)
    # drop columns 2 and 3
    df = df.drop(columns=[2, 3])
    # rename columns
    df.columns = ['date', 'product_code', 'value']
    # convert column product_code to string
    df['product_code'] = df['product_code'].astype(str)
    # clean prefix and suffix of time_code and product_code
    df['date'] = df['date'].str[4:8] + '-' + df['date'].str[8:10]
    df['product_code'] = df['product_code'].str[:5]

    return df


# create list of zip files from folder
def create_list_of_zip_files(folder):
    list_of_zip_files = []
    for file in os.listdir(folder):
        if file.endswith(".zip"):
            list_of_zip_files.append(folder + file)
    return list_of_zip_files


# product by month data
def product_by_month(df, product_prefixes_list):
    # filter df by multiple product prefixes
    df = df[df['product_code'].str.startswith(tuple(product_prefixes_list))]
    # aggregate based on substring of time_code
    df = df.groupby(['date', 'product_code'])['value'].agg(['sum']).reset_index()
    # translate product_code to product_name
    df = df.replace(
        {'product_code':{'80711':'אבטיחים/מלונים', '80540':'אשכוליות/פומלית', '71420':'בטטות', '70310':'בצלים',
                         '70610':'גזרים/לפת', '70500':'חסה', '70930':'חצילים', '70490':'כרוב/קולורבי',
                         '80550':'לימונים', '70700':'מלפפונים', '71290':'סלק', '70200':'עגבניות', '80610':'ענבים',
                         '71080':'פלפלים', '70690':'צנון/צנונית', '70933':'קישואים', '80520':'קלמנטינות',
                         '70100':'תפו"א', '80810':'תפוחים', '80510':'תפוזים', '':''}})

    return df


# plot sum of export deals by month
def plot_export_deals_by_month(df):
    # create list of months
    months = sorted(list(df['date'].unique()))
    # create list of sums of export deals by month
    sums = []
    for month in months:
        sums.append(df[df['date'] == month]['sum'])
    # create plot
    plt.figure(figsize=(20, 6))
    plt.plot(months, sums)
    plt.xlabel('month')
    plt.ylabel('sum of export deals')
    plt.title('sum of export deals by month')
    plt.show()


# get all products names
def get_all_products_names(df):
    very_simple_food = df.very_simple_food.unique().tolist()
    # sort list of strings alphabetically
    del very_simple_food[0]
    very_simple_food.sort()
    print(very_simple_food)


if __name__ == '__main__':
    # create list of zip files from folder
    folder = 'C:/Users/mshil/שולחן העבודה/datathon2022/appendix/export_import/'
    list_of_zip_files = create_list_of_zip_files(folder)
    # create dataframe from list of zip files, containing txt files
    df = create_dataframe(list_of_zip_files)
    # organize data based on product and month
    prefixes_list = ['80711', '80540', '71420', '70310', '70610', '70500', '70930', '70490', '80550', '70700', '71290',
                     '70200', '80610', '71080', '70690', '70933', '80520', '70100', '80810', '80510']
    df = product_by_month(df, prefixes_list)
    # save dataframe to csv file in folder
    df.to_csv(folder + 'export.csv', index=False)

import pandas as pd
from geopy.geocoders import Nominatim

#get location from address
def get_location(address):
    geolocator = Nominatim(user_agent="my-application")
    location = geolocator.geocode(address)
    return location.address

# 3-5 , 6-8, 9-11, 12-2
def season(month):
    if month in ['03', '04', '05']:
        return "spring"
    elif month in ['06', '07', '08']:
        return "summer"
    elif month in ['09', '10', '11']:
        return "fall"
    elif month in ['12', '01', '02']:
        return "winter"

# #Importing the dataset
# url='https://drive.google.com/file/d/1-4YpXkd2kIOM5viSRw8g7oOQm8sicciB/view?usp=sharing'
# url='https://drive.google.com/uc?id=' + url.split('/')[-2]
# y_train = pd.read_csv(url, index_col=0)

# url='https://drive.google.com/file/d/1-7VK3dNry2-AYnfRsxMWsOKhHHMTN_ZA/view?usp=sharing'
# url='https://drive.google.com/uc?id=' + url.split('/')[-2]
# test_indices = pd.read_csv(url, index_col=0)

#Importing additional datasets
url='https://drive.google.com/file/d/1-07zZ5oLAJZDck0WLMGVONV62jgTtxzZ/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
unprocessed_data_train = pd.read_csv(url, index_col=0)

#plot the data by col over date
import matplotlib.pyplot as plt
import seaborn as sns
df=unprocessed_data_train.assign(date = lambda x: x['date'].str[:-3])
df=df.assign(season = lambda x: x['date'].str[-2:])
df=df.assign(year = lambda x: x['date'].str[:4])
for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
    df.replace(month, season(month), inplace=True)

food='ברוקולי'
df=df[df['food']==food]
df=df[df['district']=='מחוז הצפון']
df=df.where(df['year']<='2015')
plt.figure(figsize=(20,6))
sns.lineplot(data=df, x='date', y='quantity(kg)', hue='season')
plt.show()
plt.title(food)


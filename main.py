import pandas as pd

#Importing the dataset
url='https://drive.google.com/file/d/1-4YpXkd2kIOM5viSRw8g7oOQm8sicciB/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
y_train = pd.read_csv(url, index_col=0)

url='https://drive.google.com/file/d/1-7VK3dNry2-AYnfRsxMWsOKhHHMTN_ZA/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
test_indices = pd.read_csv(url, index_col=0)

#Importing additional datasets


print(y_train)
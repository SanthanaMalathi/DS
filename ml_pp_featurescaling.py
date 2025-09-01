import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("diamonds.csv")

standard_scaler_object= StandardScaler()
df["converted_price"]=standard_scaler_object.fit_transform(df[['price']])
print(df[["price","converted_price"]])

minmax_scaler= MinMaxScaler()
df["converted_price"]=minmax_scaler.fit_transform(df[['price']])
print(df[["price","converted_price"]])



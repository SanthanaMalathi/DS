import pandas as pd
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("diamonds.csv")

standard_scaler_object= StandardScaler()
df["converted_price"]=standard_scaler_object.fit_transform(df[['price']])
print(df[["price","converted_price"]])



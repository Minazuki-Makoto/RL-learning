import pandas as pd
import numpy as np

price_data = pd.read_csv(
    r'D:/pycharmcode/project/HEMS/database/price_data.csv',
    encoding='gbk'
)

Temperature_data=pd.read_csv(r'D:\pycharmcode\project\HEMS\database\temp_data.csv',
                             encoding='gbk')


prices=[]
temps=[]
for i in range(len(price_data)):
    if isinstance(price_data['实时电价'].iloc[i],(int, float, np.number)):
        prices.append(float(price_data['实时电价'].iloc[i]))
    if isinstance(Temperature_data['temp'].iloc[i*398],(int, float, np.number)):
        temps.append(float(Temperature_data['temp'].iloc[i]))

def price_t(t):
    idx=t % 24
    return prices[idx]

def T_t(t):
    idx= t % 24
    return temps[idx]
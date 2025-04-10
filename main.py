import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression



plt.rcParams['figure.figsize'] = [10, 5]

df = pd.read_csv("day.csv")
df['date'] = pd.to_datetime(df['dteday'])

# Графік кількості орендованих велосипедів по днях
#df.plot.scatter(x='dteday', y='cnt')
plt.scatter(df['dteday'], df['cnt'], c = df['weathersit'])
plt.show()



print("Середнє значення для погодних умов 1: ",df[df['weathersit']==1]['cnt'].mean())
print("Середнє значення для погодних умов 2: ",df[df['weathersit']==2]['cnt'].mean())

# Графік оренди відносно нормалізованої температури
plt.scatter(df['atemp'], df['cnt'])
plt.show()


# Лінійна регресія
lr = LinearRegression()
lr.fit(df['atemp'].values.reshape(-1,1), df['cnt'].values.reshape(-1,1))
plt.scatter(df['atemp'], df['cnt'])
plt.plot(df['atemp'], lr.predict(df['atemp'].values.reshape(-1,1)), c='red')
plt.show()

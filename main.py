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
plt.plot(df['atemp'], lr.predict(df['atemp'].values.reshape(-1,1)), c='red', label='Лінія регресії')
plt.xlabel('Відчутна температура (atemp)')
plt.ylabel('Кількість оренд велосипедів (cnt)')
plt.title('Залежність між температурою та кількістю оренд велосипедів')
plt.show()



# Побудова та оційнка лінійної регресії
training_set = df[df['date'] < '2012-06-01']
validation_set = df[df['date'] >= '2012-06-01']

training_inputs = training_set[['atemp', 'workingday', 'hum', 'weathersit']].values
training_outputs = training_set[['cnt']].values

# Створюється та навчається лінійна регресія на основі тренувального набору.
lr = LinearRegression()
lr.fit(training_inputs, training_outputs)

validation_inputs = validation_set[['atemp', 'workingday', 'hum', 'weathersit']].values
validation_outputs = validation_set[['cnt']].values

# Виводить розсіювання справжніх та передбачених значень для валідаційного набору по датах.
# plt.scatter(lr.predict(validation_inputs), validation_outputs)
plt.scatter(validation_set['date'], validation_set['cnt']) # Справжні значення
plt.scatter(validation_set['date'], lr.predict(validation_inputs)) # Передбачені значення
plt.show()

# Виводить розсіювання справжніх та передбачених значень для тренуванльного набору по датах.
plt.scatter(training_set['date'], training_set['cnt'])
plt.scatter(training_set['date'], lr.predict(training_inputs))
plt.show()

rmse = np.sqrt(((lr.predict(validation_inputs) - validation_outputs)**2).mean())
print("Середньоквадратичне відхилення: ", rmse)

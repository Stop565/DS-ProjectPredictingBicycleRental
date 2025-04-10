import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.figsize'] = [10, 5]

df = pd.read_csv("day.csv")
df['date'] = pd.to_datetime(df['dteday'])
print(df)
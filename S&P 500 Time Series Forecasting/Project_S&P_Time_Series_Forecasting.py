import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense

df = pd.read_csv(r'D:/Documents/Course Documents/ECON 511B/Project/all_stocks_5yr.csv')

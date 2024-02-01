import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import mahalanobis
from numpy import cov
#import sns.heatmap as sns
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

#%%
# Read csv

df = pd.read_csv(r'D:/Documents/Course Documents/ECON 511B/Project/all_stocks_5yr.csv')

#%% 
# print columns
print(df.columns)

# print data types
print(df.dtypes)

# set index
df.reset_index()

#print dataframe
print(df)
#%%

# Data Pre-processing
# Check missing values & remove them
df.dropna(inplace = True)

# droping duplicate rows
df.drop_duplicates(inplace=True)
#%%

# Check for outliers using Mahalanobis method
threshold = 3
#mean = df.mean()
#print(mean)
covariance = df.cov()
#%%
numeric_df = df.select_dtypes(include=[np.number])
# numeric_df.reset_index()
numeric_df_T = numeric_df.T
#%%
# covariance = numeric_df.cov()
# covariance = covariance.rename_axis(None).rename_axis(None, axis = 1)
# covariance_reshaped = covariance.stack().reset_index()
# print(covariance_reshaped)
# covariance_reshaped.columns = ['a', 'b', 'c']
# covariance_reshaped.columns = ["high","low","open","close","volume"]
mean = mean.reshape(1, -1)
distance = mahalanobis(numeric_df.to_numpy().flatten(), mean, covariance)
#%%

#covariance = numeric_df.cov(numeric_df.all())

#distance = distance.reshape(numeric_df.shape[0], numeric_df.shape[1])
#outliers = df[distance > threshold]
# Heatmap for corelation
corr_1 = df.corr()
sns.heatmap(corr_1,fmt=".1%",annot = True)
Covariance_gen = df.cov()
#%%
#Create a new feature 'price_change'
df['price_change'] = df['close'] - df['open']

#Create a new feature 'returns'
df['returns'] = df['close'].pct_change()

#Create a new feature 'average_price'
df['average_price'] = (df['close'] + df['open']) / 2

df.dropna(inplace = True)
#corr_2 = df.corr()
#sns.heatmap(corr_2, annot = True,fmt=".1%",linewidths=1.5,)
#%%
# Create a dataframe without Name column
df_prime = df.copy(["high","low","open","close","volume","date"])
df_prime = df_prime.drop("Name",axis=1)

df.dropna(inplace = True)

# change date column to date-time format
df_prime["date"] = pd.to_datetime(df["date"])
df_prime = df_prime.set_index("date")
df_prime = df_prime.sort_index()

# Data cleaning
df_prime = df_prime.dropna()
#%%
# Normalize the data for better performance
scaler = MinMaxScaler()
df_prime[["high","low","open","close","volume","price_change","average_price",
          "returns"]] = scaler.fit_transform(df[["high","low","open","close",
                                                 "volume","price_change","average_price","returns"]])

# Generate stats
stats_df = df_prime[["high","low","open","close","volume"]].describe()
#stats_df = df[["high","low","open","close","volume","Name"]].groupby("Name").describe()
print(stats_df)
Covariance_gen = df.cov()


#%%
# change date column to date-time format
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
df = df.sort_index()

# Normalize the data for better performance
scaler = MinMaxScaler()
df[["high","low","open","close","volume","price_change","average_price",
          "returns"]] = scaler.fit_transform(df[["high","low","open","close",
                                                 "volume","price_change","average_price","returns"]])

# Generate stats
stats_df = df[["high","low","open","close","volume"]].describe()
#stats_df = df[["high","low","open","close","volume","Name"]].groupby("Name").describe()
print(stats_df)
Covariance_gen = df.cov()
                                                 
                                              
# Check correlation after normalization
corr_3 = df_prime.corr()
sns.heatmap(corr_3, annot = True,fmt=".1%",linewidths=1.5,)

#%%

#Split the dateset into training & testing dataset
X = df_prime.drop("close",axis=1)
Y = df_prime["close"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,shuffle=False)

#%%

# Build a LSTM model

#Initialize the model
model = Sequential()

#Add the first LSTM layer
model.add(LSTM(units=50, return_sequences=True,input_shape=(X_train.shape[1],1)))

#Add additional LSTM layers
model.add(LSTM(units=50))

#Add a fully connected layer
model.add(Dense(3))
#%%
from keras.layers import  LSTM,Dense
from keras import backend as k
import keras.optimizers as optimizers
from sklearn.metrics import f1_score
from keras.callbacks import EarlyStopping

# MSE & MAE

def mean_squared_error(Y,Y_test):
    return k.mean(k.square(Y - Y_test),axis = -1)

print(k.mean(k.square(Y - Y_test),axis = -1))

def mean_absolute_error(Y,Y_test):
    return k.mean(k.abs(Y - Y_test),axis = -1)

# Compile the model
optimizer = optimizers.Adam(lr = 0.02)
model.compile(optimizer=optimizer, 
              loss="mean_squared_error", 
              metrics=[mean_squared_error,
                       mean_absolute_error])

early_stopping = EarlyStopping(monitor="val_loss",patience=10,mode="min")

history = model.fit(X_train,Y_train,epochs=30,batch_size=32,validation_data=(X_test,Y_test),
                    callbacks=[early_stopping])

# Evaluate the model
scores = model.evaluate(X_test,Y_test,verbose=0)

# Get the range of the values
min_value = df["close"].min()
max_value = df["close"].max()
value_range = max_value - min_value

print("Max Value: ",max_value)
print("Min Value: ",min_value)
print("Range: ",value_range)

scores = model.evaluate(X_test,Y_test,verbose = 0)
# np.savetxt("trainloss.txt",scores[1])
# np.savetxt("trainloss.txt",scores[2])
print("MSE: ",scores[1])
print("MAE: ",scores[2])


# Prediction on test data
test_predictions = model.predict(X_test)

# Plot the training and validation losses
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
#%%
# Plot the training and validation losses from epoch 5 onwards
plt.plot(history.history['loss'][0:], label='Training Loss')
plt.plot(history.history['val_loss'][0:], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
#%%
# F1 Score

def f1score_mi(y_true,y_pred):
    F1_Score_micro = f1_score(y_true, y_pred,average="micro")
    print("F1_Score_micro",F1_Score_micro)
    

f1score_mi(test_predictions,Y_test)
#%%

def f1score_ma(test_predictions,Y_test):
    F1_Score_macro = f1_score(test_predictions,Y_test,average="macro")
    print("F1_Score_macro",F1_Score_macro)
    

f1score_ma(test_predictions,Y_test)
#%%
Y_test = np.float32(Y_test)
#%%

def f1score_we(test_predictions,Y_test):
    F1_Score_weighted = f1_score(test_predictions,Y_test,average=None)
    print("F1_Score_weighted",F1_Score_weighted)    
    

f1score_we(test_predictions,Y_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix

def confusion_mat(Y,Y_test):
    conf_matrix = confusion_matrix(Y,Y_test)
    print("Confusion Matrix:",conf_matrix)

confusion_mat(Y,Y_test)
#%%
from scipy import stats
import statsmodels.api as sm
# LASSO model
# Split the dateset into training & testing dataset
X = df_prime.drop("close",axis=1)
X = df_prime.loc[:, ["high","price_change", "returns", "volume", "average_price"]].copy()

Y = df_prime["close"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1 / 3, random_state = 0)

#lasso = LassoCV(cv=3, max_iter=100, random_state=0)

# Build Lasso Model
lasso = Lasso(alpha=0.0006,max_iter=5000)

# Fit the model
lasso.fit(X_train,Y_train)

# Prediction
Y_pred = lasso.predict(X_test)

# Generate Regression model coefficients
lasso_coeff = (lasso.coef_)

# set print options
np.set_printoptions(precision=5)
print("Coefficients",lasso_coeff)

# Confidence intervals for coefficients

n = len(X_train)
p = len(X_train.columns)
dof = max(0, n - p - 1)
t_value = stats.t.ppf(1 - 0.025, dof)
std_err = np.sqrt(np.sum((Y_train - lasso.predict(X_train))**2) / dof / (n - p))
conf_ints = np.array([lasso_coeff - t_value * std_err, lasso_coeff + t_value * std_err]).T
print("95% confidence intervals for coefficients:")
for i in range(len(X_train.columns)):
    print(X_train.columns[i], ": [", conf_ints[i,0], ",", conf_ints[i,1], "]")

# Residual plots
residuals = Y_test - Y_pred
fig, ax = plt.subplots(1,2)
ax[0].scatter(Y_pred, residuals)
ax[0].set_xlabel('Predicted values')
ax[0].set_ylabel('Residuals')
ax[0].set_title('Residual plot')

ax[1].hist(residuals, bins=30)
ax[1].set_xlabel('Residuals')
ax[1].set_ylabel('Frequency')
ax[1].set_title('Histogram of residuals')

plt.show()
#%%
# Create the Lasso score
print('R squared test set', round(lasso.score(X_test, Y_test)*100, 2))

mse_test = mean_squared_error(Y_test, Y_pred)
# print('MSE test set', round(mse_test, 4))

# calculate root mean squared error
rmse_test = np.sqrt(mse_test)

# calculate range of dependent variable
y_range = np.max(Y_test) - np.min(Y_test)

# convert to percentage
mse_percent = rmse_test / y_range * 100  

print("MSE as a percentage: {:.2f}%".format(mse_percent))

# calculate MSE
mse_test = mean_squared_error(Y_test, Y_pred)

# calculate R-squared
r_squared = lasso.score(X_test, Y_test)

# plot MSE and R-squared
plt.plot([mse_test]*len(Y_test), label='MSE')
plt.plot([r_squared]*len(Y_test), label='R-squared')
plt.legend()
plt.ylabel('MSE and R-squared')
plt.xlabel('Observation')

plt.show()


#%%
import statsmodels.api as sm
# Ridge Model
# Split the dateset into training & testing dataset
X = df_prime.loc[:, ["high", "price_change", "returns", "volume", "average_price"]].copy()

Y = df_prime["close"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1 / 3, random_state = 0)

# define model
model = Ridge(alpha=2e-5)

# Fit the model
model.fit(X_train,Y_train)


# Prediction
Y_pred = model.predict(X_test)

# Generate Regression model coefficients
ridge_coeff = (model.coef_)

# set print options
np.set_printoptions(precision=5)
print("Coefficients",ridge_coeff)

# Confidence intervals for coefficients
n = len(X_train)
p = len(X_train.columns)
dof = max(0, n - p - 1)
t_value = stats.t.ppf(1 - 0.025, dof)
std_err = np.sqrt(np.sum((Y_train - model.predict(X_train))**2) / dof / (n - p))
conf_ints = np.array([ridge_coeff - t_value * std_err, ridge_coeff + t_value * std_err]).T
print("95% confidence intervals for coefficients:")
for i in range(len(X_train.columns)):
    print(X_train.columns[i], ": [", conf_ints[i,0], ",", conf_ints[i,1], "]")

# Create the Ridge score
print('R squared test set', round(model.score(X_test, Y_test)*100, 2))

mse_test = mean_squared_error(Y_test, Y_pred)
# print('MSE test set', round(mse_test, 4))

# calculate root mean squared error
rmse_test = np.sqrt(mse_test)

# calculate range of dependent variable
y_range = np.max(Y_test) - np.min(Y_test)

# convert to percentage
mse_percent = rmse_test / y_range * 100  

print("MSE as a percentage: {:.2f}%".format(mse_percent))

# calculate MSE
mse_test = mean_squared_error(Y_test, Y_pred)

# calculate R-squared
r_squared = model.score(X_test, Y_test)

# plot MSE and R-squared
plt.plot([mse_test]*len(Y_test), label='MSE')
plt.plot([r_squared]*len(Y_test), label='R-squared')
plt.legend()
plt.ylabel('MSE and R-squared')
plt.xlabel('Observation')

plt.show()

# Residual plots
residuals = Y_test - Y_pred
fig, ax = plt.subplots(1,2)
ax[0].scatter(Y_pred, residuals)
ax[0].set_xlabel('Predicted values')
ax[0].set_ylabel('Residuals')
ax[0].set_title('Residual plot')

ax[1].hist(residuals, bins=30)
ax[1].set_xlabel('Residuals')
ax[1].set_ylabel('Frequency')
ax[1].set_title('Histogram of residuals')

plt.show()

#%%
# Calculate evaluation metrics
mse_test = mean_squared_error(Y_test, Y_pred)
r_squared = model.score(X_test, Y_test)
n = len(Y_test)
p = X_test.shape[1]
adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))

# Print results
print('MSE test set:', round(mse_test, 4))
print('R-squared test set:', round(r_squared, 4))
print('Adjusted R-squared test set:', round(adj_r_squared, 4))

# Calculate MSE as a percentage of the range of the target variable
y_range = np.max(Y_test) - np.min(Y_test)
mse_percent = rmse_test / y_range * 100  
print("MSE as a percentage: {:.2f}%".format(mse_percent))

# Plot MSE and R-squared
mse_vec = np.full_like(Y_test, mse_test)
r_squared_vec = np.full_like(Y_test, r_squared)
adj_r_squared_vec = np.full_like(Y_test, adj_r_squared)

plt.plot(mse_vec, label='MSE')
plt.plot(r_squared_vec, label='R-squared')
plt.plot(adj_r_squared_vec, label='Adjusted R-squared')
plt.legend()
plt.ylabel('MSE and R-squared')
plt.xlabel('Observation')
plt.title('MSE and R-squared on test set')
plt.show()
#%%
from matplotlib import pyplot
from arch import arch_model

# GARCH Model
df_prime = df_prime.sort_index()
data = df_prime.loc[:, ["returns"]].copy()
# data = data.set_index("Date")
data = data.sort_index()
data.index = pd.to_datetime(data.index)
data_train,data_test = train_test_split(data, test_size = 1 / 3, random_state = 0)

# define model
model = arch_model(data_train, mean='Zero', vol='GARCH', p=1,q=1,o=0,dist="ged",rescale=False)

# fit model
model_fit = model.fit(update_freq=5)

# forecast the test set
Y_pred = model_fit.forecast(horizon=5,reindex=False)

# print predicted values
print(Y_pred.mean.iloc[-1,:])

# plot actual test set values
pyplot.plot(data_test, label='Actual')

# plot forecasted values
pyplot.plot(Y_pred.mean['h.5'], label='Forecast')

# add plot labels and legend
pyplot.xlabel('Date')
pyplot.ylabel('Returns')
pyplot.title('GARCH Model Forecast')
pyplot.legend()

# show plot
pyplot.show()

#%%
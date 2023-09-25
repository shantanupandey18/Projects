# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 22:59:00 2023

@author: presh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.ticker as mtick
import snowflake.connector as connector
import os

df1 = pd.read_csv("D:/Documents/Course Documents/MIS 587/Project/cleaned_iowa_dataset.csv")

# Convert date column to datetime dtype
df1['Date'] = pd.to_datetime(df1['Date'])

# print(df1[["Invoice_Num"]].count().unique())
#%%
connection_string = connector.connect(user = "SHANTANUPANDEY",
                                      password = "Shantanu@18",
                                      account = "wdhfhcu-qlb85986",
                                      warehouse = "COMPUTE_WH",
                                      database = "iowa_liquor_stores")

cursor = connection_string.cursor()
#%%
### CREATES TABLE TRANSACTIONS
cursor.execute("""CREATE OR REPLACE TABLE SCHEMA_IOWA.HY_VEE (
                        INVOICE_NUM NUMBER PRIMARY KEY,
                        DATE_TIME DATETIME,
                        STORE_NUM INT,
                        STORE_NAME VARCHAR(60),
                        ADDRESS VARCHAR(100),
                        CITY VARCHAR(30),
                        COUNTY_NAME VARCHAR(40),
                        CATEGORY_NUM INT,
                        CATEGORY_NAME VARCHAR(40),
                        VENDOR_NUM INT,
                        VENDOR_NAME VARCHAR(40),
                        ITEM_NUM INT,
                        ITEM_DESC VARCHAR(40),
                        PACK INT,
                        VOLUME INT,
                        STATE_BOTTLE_COST FLOAT,
                        STATE_BOTTLE_RETAIL FLOAT,
                        BOTTLE_SOLD INT,
                        SALE_DOLLORS FLOAT,
                        VOLUME_SOLD_GALLONS FLOAT
                        );"""
                      )

#%%
# Upload csv to Snowflake

file_path = os.path.abspath("cleaned_iowa_dataset.csv")
table_name = "HY_VEE"
cursor.execute("use schema SCHEMA_IOWA")
cursor.execute(f"put file://{file_path}* @%{table_name}")
cursor.execute(f"copy into {table_name} ON_ERROR = 'continue';")

#%%
# Not required
for row in df1.to_records(index=False):
    cursor.execute(" Insert Into SCHEMA_IOWA.HY_VEE(INVOICE_NUM) "
                   "VALUES (?);", row[0])
print("uploaded")

#%%
# Not required
for row in df1.to_records(index=False):
    cursor.execute(" Insert Into SCHEMA_IOWA.HY_VEE(INVOICE_NUM) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?):",
                   ((row[0]),(row[1]),(row[2]),(row[3]),(row[4]),(row[5]),(row[6]),(row[7]),(row[8]),(row[9]),(row[10]),(row[11]),(row[12]),(row[13]),(row[14]),(row[15]),(row[16]),(row[17]),(row[18]),(row[19])))
print("uploaded")

#%%
print(df1.dtypes)
#%%
df1["Store_Num"].sort_values()
print(df1["Store_Num"].unique())

print(df1.dtypes)

#%%
# Mrudang don't graph

#Grouped_Store=df1.groupby("Store_Num")

# Sort df1 based on bottles sold
# df1.sort_values("Bottle_Sold",ascending=False)

# Top 10 stores
aggr_Store_Num=df1.groupby("Store_Num")["Bottle_Sold"].sum()
#df1.sort

# Group the top 50 store by store number
Top_50_stores = aggr_Store_Num.sort_values(ascending=False).head(50)

# Plot the bar graph
plt.bar(Top_50_stores.index, Top_50_stores.values)
plt.xlabel("Store Number")
plt.ylabel("Total Bottles Sold")
plt.title("Top 50 Stores by Sale")
plt.show()
#%%
# Grapg generation and logic for top 50 stores by number of bottle sold

# Group the data by store number and sum the bottle sold
aggr_Store_Num = df1.groupby("Store_Num")["Bottle_Sold"].sum()

# Sort the groups by the sum of bottle sold and select the top 50 stores
Top_50_stores = aggr_Store_Num.sort_values(ascending=False).head(50)

# Plot the bar graph
cmap = plt.get_cmap("coolwarm")
c = cmap(np.linspace(0,1,len(Top_50_stores)))

fig, ax = plt.subplots(figsize=(20,5)) # Increase figure size

ax.bar(Top_50_stores.index,Top_50_stores.values,color=c)
ax.set_xlabel("Store Number")
ax.set_ylabel("Total Bottles Sold")
ax.set_title("Top 50 Stores by Sale")
ax.set_xticks(Top_50_stores.index) # Set x-tick labels to all store numbers
ax.tick_params(axis='x', rotation=90) # Rotate x-tick labels by 90 degrees
plt.tight_layout()  # adjust layout to prevent overlapping labels

plt.show()

#%%

# Graph generation and logic - Top 10 Categories Bottle Sold across all Stores

# Group the data by store number and sum the bottle sold

Prod_data = df1.groupby(["Store_Num","Category_Num"])["Bottle_Sold"].sum()
Prod_data = Prod_data.reset_index()
Prod_data = Prod_data.sort_values(by="Bottle_Sold", ascending=False).head(10)

# Combine Store Number & Category Name
Prod_data["Group"] = Prod_data["Category_Num"].astype(str) + "-" + Prod_data["Store_Num"].astype(str)

# Plot the graph
fig,ax=plt.subplots(figsize=(12,8))

# Create the bar plot
ax.bar(x=Prod_data["Group"],height=Prod_data["Bottle_Sold"])

# Add lables and title
ax.set_title("Top 10 Categories Bottle Sold across all Stores")
ax.set_xlabel("Category Number - Store Number")
ax.set_ylabel("Bottle Sold")

# Make it readable
plt.xticks(rotation=360)

# Show the plot
plt.show()

#%%

# Graph generation and logic - For sale at 2633

# Group the data for store number = 2633
Store_2633 = (df1.loc[df1["Store_Num"] == (2633)])

# Group the data based on Category & generate Bottle sold
Grp_2633 = df1.groupby(["Category_Num"])["Bottle_Sold"].sum()

# Sort the data in descending order
Grp_2633 = Grp_2633.reset_index()
# Convert the Bottle_Sold column to integer data type
Grp_2633 = Grp_2633.sort_values(by="Bottle_Sold",ascending=False).head(25)
print(Grp_2633.dtypes)

# Convert Bottle_Sold to integer type
Grp_2633 = Grp_2633.astype({"Bottle_Sold": int}) 

# Plot the graph
fig,ax=plt.subplots(figsize=(16,24))

# Create the bar plot
ax.bar(x=Grp_2633["Category_Num"].astype(str),height=round(Grp_2633["Bottle_Sold"]))

# Add lables and title
ax.set_title("Top 25 Categories Bottle Sold at Store 2633")
ax.set_xlabel("Category Number")
ax.set_ylabel("Bottle Sold")

# Add formatter to y-axis to display actual count of bottles sold
formatter = mtick.StrMethodFormatter('{x:,.0f}')
ax.yaxis.set_major_formatter(formatter)

# Make it readable
plt.xticks(rotation=90)

# Show the plot
plt.show()

#%%



# Graph generation and logic - Top 10 Categories Bottle Sold across all Stores

# Group the data by store number and sum the bottle sold

Prod_data = df1.groupby(["Store_Num","Item_Num"])["Bottle_Sold"].sum()
Prod_data = Prod_data.reset_index()
Prod_data = Prod_data.sort_values(by="Bottle_Sold", ascending=False).head(10)

# Combine Store Number & Category Name
Prod_data["Group"] = Prod_data["Item_Num"].astype(str) + "-" + Prod_data["Store_Num"].astype(str)

# Plot the graph
fig,ax=plt.subplots(figsize=(12,8))

# Create the bar plot
ax.bar(x=Prod_data["Group"],height=Prod_data["Bottle_Sold"])

# Add lables and title
ax.set_title("Top 10 Categories Bottle Sold based on date across all Stores")
ax.set_xlabel("Item Number - Store Number")
ax.set_ylabel("Bottle Sold")

# Make it readable
plt.xticks(rotation=360)

# Show the plot
plt.show()

#%%
# Top 5 selling products by date
#df1 = df1.set_index("Date")
#df1 = df1.sort_index()
#print(df1["Date"].dtype)
df1['Date'] = pd.to_datetime(df1['Date'])
#df1["Date"]=df1.rename_axis("Date_Time",axis="Date")
#df1 = df1.dropna(subset=['Date_Time'])
#print(top_50_sales_date.dtypes)
# df1["Date_Time"] = df1["Date_Time"].dt.strftime('%m/%d/%Y')
#df1['Date'] = pd.to_datetime(df1['Date'], format='%m/%d/%Y')
print(df1['Date'].dtype)

date_categ_sales = df1.groupby(["Date","Category_Num"])["Bottle_Sold"].sum().sort_values(ascending=False)
top_50_sales_date = (date_categ_sales.sort_values(ascending=False).head(5))
top_50_sales_date = top_50_sales_date.reset_index()
top_50_sales_date["Grp"] = top_50_sales_date["Date"].astype(str) + "-" + top_50_sales_date["Category_Num"].astype(str)

#top_50_sales_date["Grp"] = top_50_sales_date["Date"].astype(datetime)+"-"+top_50_sales_date["Category_Name"].astype(str)

# Create a bar plot
fig,ax= plt.subplots(figsize=(12,24))

# Create a bar plot
ax.bar(x=top_50_sales_date["Grp"].astype(str),height=top_50_sales_date["Bottle_Sold"])

# Add labels and titles
ax.set_title("Top 5 selling products by date")
ax.set_xlabel("Date - Category Num")
ax.set_ylabel("Bottles Sold")
#ax.tick_params(axis='x', rotation=90)

plt.show()

#%%

# Top 10 selling Item by date

df1['Date'] = pd.to_datetime(df1['Date'])

date_categ_sales = df1.groupby(["Date","Item_Num"])["Bottle_Sold"].sum().sort_values(ascending=False)
top_50_sales_date = (date_categ_sales.sort_values(ascending=False).head(10))
top_50_sales_date = top_50_sales_date.reset_index()
top_50_sales_date["Grp"] = top_50_sales_date["Date"].astype(str) + "-" + top_50_sales_date["Item_Num"].astype(str)

#top_50_sales_date["Grp"] = top_50_sales_date["Date"].astype(datetime)+"-"+top_50_sales_date["Category_Name"].astype(str)

# Create a bar plot
fig,ax= plt.subplots(figsize=(12,24))

# Create a bar plot
ax.bar(x=top_50_sales_date["Grp"].astype(str),height=top_50_sales_date["Bottle_Sold"])

# Add labels and titles
ax.set_title("Top 10 selling products by date")
ax.set_xlabel("Date - Item Num")
ax.set_ylabel("Bottles Sold")
#ax.tick_params(axis='x', rotation=90)

plt.show()

#%%
# Sort the groups by the sum of bottle sold and select the top 50 stores
Top_50_stores = aggr_Store_Num.sort_values(ascending=False).head(50)

# Plot the bar graph
cmap = plt.get_cmap("coolwarm")
c = cmap(np.linspace(0,1,len(Top_50_stores)))

fig, ax = plt.subplots(figsize=(20,5)) # Increase figure size

ax.bar(Top_50_stores.index,Top_50_stores.values,color=c)
ax.set_xlabel("Store Number")
ax.set_ylabel("Total Bottles Sold")
ax.set_title("Top 50 Stores by Sale")
ax.set_xticks(Top_50_stores.index) # Set x-tick labels to all store numbers
ax.tick_params(axis='x', rotation=90) # Rotate x-tick labels by 90 degrees
plt.tight_layout()  # adjust layout to prevent overlapping labels

plt.show()


#%%
df1.set_index("Date")
# df1.reset_index()
df2 = df1[["Store_Num","Item_Num","Bottle_Sold","Date"]]

df2= df2.groupby(["Item_Num",pd.Grouper(key="Date",freq="M"),"Store_Num"])["Bottle_Sold"].sum()

df2 = df2.to_frame().reset_index()
df2 = df2.sort_values("Bottle_Sold", ascending=False)

df3 = df2.head(20)

df3 = pd.Series(df3["Date"], index=df3.index)
df2.reset_index()
# df3.squeeze(axis = 0)
#pd.DataFrame()
# df2 = df2.pivot(index="Date",columns="Item_Num",values="Bottle_Sold")
# Compute the sum of bottle sold for each brand of liquor across all stores
#df2_total=pd.DataFrame(df2.sum(axis=0),columns=["Total_Bottle_Sold"])

# Compute the type of liquor getting sold the most for each month
#max_type = df2.idxmax(axis=1)

# Plot the graph
df3.plot(figsize=(20,12))
plt.title("Monthly Bottle Sale by Item")
plt.xlabel("Bottel Sold")
plt.ylabel("Year")
plt.tick_params(rotation = 0)
plt.show()
        
#%%
sales_forecast_df = df1[["Date","Bottle_Sold","Store_Num"]]

# Need to assign the output of set_index to sales_forecast_df
sales_forecast_df.set_index("Date", inplace=True)

from statsmodels.tsa.stattools import adfuller

# Define a function to check for stationarity using the ADF test
def check_stationarity(data):
    # Perform the ADF test
    result = adfuller(data)

    # Extract the p-value from the test results
    p_value = result[1]

    # Check if the data is stationary based on the p-value
    if p_value < 0.05:
        return True
    else:
        return False

# Define a function to make the data stationary by taking the first difference
def make_stationary(data):
    # Take the first difference of the data to make it stationary
    diff_data = data.diff().dropna()

    return diff_data

# Apply the make_stationary function to each store's sales data and store the 
# stationary data in a dictionary
stationary_sales = list()
for store in sales_forecast_df['Store_Num'].unique():
    store_sales = sales_forecast_df.loc[sales_forecast_df['Store_Num'] == store, 'Bottle_Sold']
    try:
        if check_stationarity(store_sales):
            print(f'Store {store}: The data is stationary')
            stationary_sales.append(store) # Store the original data in the dictionary
        else:
            print(f'Store {store}: The data is not stationary, making it stationary...')
            # Apply the make_stationary function
            stationary_data = make_stationary(store_sales)
            # Store the stationary data in the dictionary
            stationary_sales.append(store) 
            if check_stationarity(stationary_data):
                stationary_sales.append(store)
                print(f'Store {store}: The data is now stationary')
            else:
                print(f'Store {store}: The data is still not stationary')
    except ValueError:
        print(f"sample size is too short to use selected regression component: {store}")
        continue


#%%

from statsmodels.tsa.arima.model import ARIMA

#store_nums = [2633, 2512, 2670]

def arima_predict(Bottle_Sold):
    # Test for stationarity and make the data stationary
    diff1 = Bottle_Sold.diff(1).dropna()

    # Choose the values of p, d, and q using the AIC criterion
    p_values = range(0, 5)
    d_values = range(1, 3)
    q_values = range(0, 5)

    best_aic = np.inf
    best_params = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(diff1, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)

                except:
                    continue

    # Fit the ARIMA model using the best parameters
    model = ARIMA(diff1, order=best_params)
    results = model.fit()

    # Make a prediction for the next month
    forecast = results.forecast()
    forecast = pd.Series(forecast, index=[Bottle_Sold.index[-1]]).iloc[0]

    # Convert the prediction back to the original scale
    forecast = diff1.iloc[-1] + forecast
    forecast = Bottle_Sold.iloc[-1] + forecast

    if pd.isna(forecast):
        forecast = 0

    return forecast

#%%

prediction = dict()
for store_num in stationary_sales:
    sales_forecast_df = df1.loc[df1['Store_Num'] == store_num,["Date","Bottle_Sold","Store_Num"]].reset_index()
    sales_forecast_df.index = pd.DatetimeIndex(sales_forecast_df.index).to_period('M')
    # Group by date and store number and get the maximum value of bottle sold
    store_sales_max_date = sales_forecast_df.groupby(['Date','Store_Num']).agg({'Bottle_Sold': 'max'})
    store_sales_max_date = store_sales_max_date.reset_index()
    store_sales_max_date.index = pd.DatetimeIndex(store_sales_max_date['Date']).to_period('M')
    
    # Make a prediction using the date with the maximum sales
    max_date = store_sales_max_date.loc[store_sales_max_date['Bottle_Sold'].idxmax(),'Date']
    
    # Make a prediction for the next month using the data up to the maximum sales date
    data_up_to_max_date = store_sales_max_date.loc[store_sales_max_date.index.get_level_values(0) <= max_date]
#    data_up_to_max_date = store_sales_max_date.loc[store_sales_max_date.index.get_level_values(0) <= max_date]
#    data_up_to_max_date = store_sales_max_date.loc[store_sales_max_date.index <= max_date]
#    data_up_to_max_date = store_sales_max_date.reset_index().loc[store_sales_max_date['Date'] <= max_date]
#    data_up_to_max_date = store_sales_max_date.loc[store_sales_max_date['Date'] <= max_date]
    try:
        if not data_up_to_max_date.empty:
            for store_num in store_nums:
                store_data = data_up_to_max_date.loc[data_up_to_max_date['Store_Num'] == store_num, 'Bottle_Sold']
                if not store_data.empty:
                    prediction[store_num] = arima_predict(store_data)
#                    prediction.update[store_nums] = arima_predict(store_data)
                    print(f"Predicted sales for Store_Num = {store_num} for the month after {max_date}: {prediction[store_nums][0]:.2f}")
                else:
                    print(f"No data for Store_Num = {store_num} before maximum sales date")
        else:
            print(f"No data for Store_Num = {store_num} before maximum sales date")
            
    except TypeError:
        continue


#%% Yash code
prediction = dict()
for store_nums in stationary_sales:
    sales_forecast_df = df1.loc[df1['Store_Num'] == store_nums,["Date","Bottle_Sold","Store_Num"]].reset_index()
    sales_forecast_df.index = pd.DatetimeIndex(sales_forecast_df.index).to_period('M')
    # Group by date and store number and get the maximum value of bottle sold
    store_sales_max_date = sales_forecast_df.groupby(['Date','Store_Num']).agg({'Bottle_Sold': 'max'})
    
    # Make a prediction using the date with the maximum sales
    max_date = store_sales_max_date.loc[store_sales_max_date['Bottle_Sold'].idxmax(),'Date']
    
    # Make a prediction for the next month using the data up to the maximum sales date
    data_up_to_max_date = store_sales_max_date.loc[store_sales_max_date['Date'] <= max_date]
    try:
        if not data_up_to_max_date.empty:
            for store_num in store_nums:
                store_data = data_up_to_max_date.loc[data_up_to_max_date['Store_Num'] == store_num, 'Bottle_Sold']
                if not store_data.empty:
                    prediction.update[store_nums] = arima_predict(store_data)
                    print(f"Predicted sales for Store_Num = {store_num} for the month after {max_date}: {prediction[store_num]:.2f}")
                else:
                    print(f"No data for Store_Num = {store_num} before maximum sales date")
        else:
            print(f"No data for Store_Num = {store_num} before maximum sales date")
            
    except TypeError:
        continue

#%%
from statsmodels.tsa.arima.model import ARIMA
sales_forecast_df = df1.loc[df1['Store_Num'] == 2633 , ["Date","Bottle_Sold","Store_Num"]]

# Group by date and store number and get the maximum value of bottle sold
store_sales_max_date = sales_forecast_df.groupby(['Date','Store_Num']).agg({'Bottle_Sold': 'max'}).reset_index()

def arima_predict(Bottle_Sold):
    # Test for stationarity and make the data stationary
    diff1 = Bottle_Sold.diff(1).dropna()

    # Choose the values of p, d, and q using the AIC criterion
    p_values = range(0, 5)
    d_values = range(1, 3)
    q_values = range(0, 5)

    best_aic = np.inf
    best_params = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(diff1, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)

                except:
                    continue

    # Fit the ARIMA model using the best parameters
    model = ARIMA(diff1, order=best_params)
    results = model.fit()

    # Make a prediction for the next month
    forecast = results.forecast()
    forecast = pd.Series(forecast, index=[Bottle_Sold.index[-1]]).iloc[0]

    # Convert the prediction back to the original scale
    forecast = diff1.iloc[-1] + forecast
    forecast = Bottle_Sold.iloc[-1] + forecast

    if pd.isna(forecast):
        forecast = 0

    return forecast


# Make a prediction using the date with the maximum sales
max_date = store_sales_max_date.loc[store_sales_max_date['Bottle_Sold'].idxmax(),'Date']

# Make a prediction for the next month using the data up to the maximum sales date
data_up_to_max_date = store_sales_max_date.loc[store_sales_max_date['Date'] <= max_date]
if not data_up_to_max_date.empty:
    prediction = arima_predict(data_up_to_max_date['Bottle_Sold'])

    # Calculate the actual sales for the month after the maximum sales date
    actual_sales = sales_forecast_df.loc[sales_forecast_df['Date'] == max_date, 'Bottle_Sold'].sum()

    # Calculate the Mean Absolute Percentage Error (MAPE)
    mape = abs((actual_sales - prediction) / actual_sales) * 100
    
    # Print the prediction
    print(f"Predicted sales for Store_Num = 2633 for the month after {max_date}: {prediction:.2f}")
    print(f"MAPE: {mape:.2f}%")
else:
    print(f"No data for Store_Num = 2633 before maximum sales date")

#%%
# Make a prediction using the date with the maximum sales
max_date = store_sales_max_date.loc[store_sales_max_date['Bottle_Sold'].idxmax(),'Date']

# Make a prediction for the next month using the data up to the maximum sales date
data_up_to_max_date = store_sales_max_date.loc[store_sales_max_date['Date'] <= max_date]
if not data_up_to_max_date.empty:
    prediction = arima_predict(data_up_to_max_date['Bottle_Sold'])

    # Print the prediction
    print(f"Predicted sales for Store_Num = 2633 for the month after {max_date}: {prediction:.2f}")
else:
    print(f"No data for Store_Num = 2633 before maximum sales date")

#%%
def arima_predict(Bottle_Sold):
    # Test for stationarity and make the data stationary
    diff1 = Bottle_Sold.diff(1).dropna()

    # Choose the values of p, d, and q using the AIC criterion
    p_values = range(0, 5)
    d_values = range(1, 3)
    q_values = range(0, 5)

    best_aic = np.inf
    best_params = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(diff1, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)

                except:
                    continue

    # Fit the ARIMA model using the best parameters
    model = ARIMA(diff1, order=best_params)
    results = model.fit()

    # Make a prediction for the next month
    forecast = results.forecast()
    forecast = pd.Series(forecast, index=[Bottle_Sold.index[-1]]).iloc[0]

    # Convert the prediction back to the original scale
    forecast = diff1.iloc[-1] + forecast
    forecast = Bottle_Sold.iloc[-1] + forecast

    if pd.isna(forecast):
        forecast = 0

    return forecast

# Group by date, store number, and item number and get the maximum value of bottle sold
store_sales_max_date = sales_forecast_df.groupby(["Date","Store_Num", "Item_Num"]).agg({'Bottle_Sold': 'max'}).reset_index()

# Make a prediction using the date with the maximum sales
max_date = store_sales_max_date.loc[store_sales_max_date['Bottle_Sold'].idxmax(),'Date']

# Make a prediction for the next month using the data up to the maximum sales date
data_up_to_max_date = store_sales_max_date.loc[store_sales_max_date['Date'] <= max_date]
if not data_up_to_max_date.empty:
    prediction = arima_predict(data_up_to_max_date['Bottle_Sold'])

    # Get the item number for the store
    item_num = data_up_to_max_date.loc[data_up_to_max_date['Bottle_Sold'].idxmax(),'Item_Num']

    # Print the prediction
    print(f"Predicted sales for Store_Num = 2633 and Item_Num = {item_num} for the month after {max_date}: {prediction:.2f}")
else:
    print(f"No data for Store_Num = 2633 before maximum sales date")


#%%
sales_forecast_df = df1.loc[df1['Store_Num'] == 2633, ["Date","Bottle_Sold","Store_Num"]]

# Group by date and store number and get the maximum value of bottle sold
store_sales_max_date = sales_forecast_df.groupby(['Date','Store_Num']).agg({'Bottle_Sold': 'max'}).reset_index()

def arima_predict(Bottle_Sold):
    # Test for stationarity and make the data stationary
    diff1 = Bottle_Sold.diff(1).dropna()

    # Choose the values of p, d, and q using the AIC criterion
    p_values = range(0, 5)
    d_values = range(1, 3)
    q_values = range(0, 5)

    best_aic = np.inf
    best_params = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(diff1, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)

                except:
                    continue

    # Fit the ARIMA model using the best parameters
    model = ARIMA(diff1, order=best_params)
    results = model.fit()

    # Make a prediction for the next month
    forecast = results.forecast()
    forecast = pd.Series(forecast, index=[Bottle_Sold.index[-1]])[0]

    # Convert the prediction back to the original scale
    forecast = diff1.iloc[-1] + forecast
    forecast = Bottle_Sold.iloc[-1] + forecast

    if pd.isna(forecast):
        forecast = 0

    return forecast

# Define a list of stores to make predictions for
store_list = store_sales_max_date['Store_Num'].unique()

# Create an empty DataFrame to store the predictions
df_predictions = pd.DataFrame(columns=['Store_Num', 'Date', 'Sales'])

# Loop through each store and make a prediction using the date with the maximum sales
for store in store_list:
    # Filter the data to include only the data for the current store
    data = store_sales_max_date[store_sales_max_date['Store_Num'] == store]

    # Reset the index of the DataFrame to a column
    data = data.reset_index()

    # Use regular indexing to slice the DataFrame
    max_date = data.loc[data['Bottle_Sold'].idxmax(),'Date']

    # Make a prediction for the next month using the data up to the maximum sales date
    data_up_to_max_date = data.loc[data['Date'] <= max_date]
    if not data_up_to_max_date.empty:
        prediction = arima_predict(data_up_to_max_date['Bottle_Sold'])

        # Append the prediction to the DataFrame
        df_predictions = df_predictions.append({'Store_Num': store, 'Date': max_date, 'Sales': prediction}, ignore_index=True)
    else:
        print(f"No data for store {store} before maximum sales date")

# Print the DataFrame of predictions
print(df_predictions)




#%%
# Define a function to check for stationarity using the ADF test
def check_stationarity(data):
    # Perform the ADF test
    result = adfuller(data)

    # Extract the p-value from the test results
    p_value = result[1]

    # Print the test results
    print('ADF Statistic: {:.2f}'.format(result[0]))
    print('p-value: {:.2f}'.format(p_value))
    print('')

    # Check if the data is stationary based on the p-value
    if p_value < 0.05:
        print('The data is stationary')
    else:
        print('The data is not stationary')

# Select the 'Sale_Dollors' column of the dataset for a particular store
store_sales = sales_forecast_df[sales_forecast_df['Store_Num'] == 2633]['Bottle_Sold']
# store_sales = sales_forecast_df['Bottle_Sold']

# Check for stationarity using the ADF test
check_stationarity(store_sales)

# Define a function to fit an ARIMA model and make a prediction for a given brand of liquor
def arima_predict(Bottle_Sold, Store_Num):
    # Test for stationarity and make the data stationary
    diff1 = sales_forecast_df.diff(1).dropna()

    # Choose the values of p, d, and q using the AIC criterion
    p_values = range(0, 5)
    d_values = range(1, 3)
    q_values = range(0, 5)

    best_aic = np.inf
    best_params = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(diff1[brand], order=(p, d, q))
                    results = model.fit()
                    aic = results.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)

                except:
                    continue

    # Fit the ARIMA model using the best parameters
    model = ARIMA(diff1[brand], order=best_params)
    results = model.fit()

    # Make a prediction for the next month
    forecast = results.forecast()[0]

    # Convert the prediction back to the original scale
    forecast = diff1[brand].iloc[-1] + forecast
    forecast = data[brand].iloc[-1] + forecast

    return forecast

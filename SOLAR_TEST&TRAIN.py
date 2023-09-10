

############################################################
# USING MACHINE LEARNING TO FORECAST SOLAR ENERGY PRODUCTION
############################################################

#libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')



df = pd.read_csv('/Users/bayramsaygili/Desktop/MIUUL DATA SCIENTIST 12TH TERM JUN-SEP23/CAPSTONE_DATASET/EMHIRESPV_TSh_CF_Country_19862015.csv')

baslangic_tarihi = pd.to_datetime('01-01-1986 00:00', format='%d-%m-%Y %H:%M')
bitis_tarihi = pd.to_datetime('31-12-2015 23:59', format='%d-%m-%Y %H:%M')

#Saatlik aralıkla tarihleri oluşturun
tarihler = pd.date_range(start=baslangic_tarihi, end=bitis_tarihi, freq='H')

#'Tarih' sütunu ile yeni bir veri çerçevesi oluşturun
df['date'] = tarihler


df.head(25)



#re-assign df value


df = df[["DE","date"]]


df.head()

# Converting date column to index

df = df.set_index('date')
df.index = pd.to_datetime(df.index)




# visualization of time series

df.plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='Solar Energy Production in Germany 1986-2015')
plt.show()


###################
#Outlier Analysis and re-assignment with thresholds
###################

# to define threshold values (low & up)

def outlier_thresholds(dataframe, col_name, q1=0.16, q3=0.84):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, "DE")


# outlier_thresholds : (-0.39198558219787294, 0.6533093036631216)



#####################
# are there outliers or not?
#####################

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df, "DE")

# answer : TRUE !!


#####################
# visualize outliers
#####################

df.query('DE > 0.6533093036631216')['DE'] \
    .plot(style='.',
          figsize=(15, 5),
          color=color_pal[5],
          title='Outliers')



###################
# accessing the outliers themselves
###################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


grab_outliers(df, "DE")

grab_outliers(df, "DE", True)

age_index = grab_outliers(df, "DE", True)

outlier_thresholds(df, "DE")
check_outlier(df, "DE")
grab_outliers(df, "DE", True)


#DatetimeIndex(['1986-05-02 11:00:00', '1987-05-09 11:00:00', '1988-04-14 11:00:00',
#               '1988-04-14 12:00:00', '1988-05-12 11:00:00', '1990-05-01 11:00:00',
#               '1990-05-03 11:00:00', '1992-05-18 11:00:00', '1994-05-03 11:00:00',
#               '1998-05-14 11:00:00', '1999-08-11 11:00:00', '2002-03-28 11:00:00',
#               '2002-03-28 12:00:00', '2003-04-14 11:00:00', '2003-04-14 12:00:00',
#               '2003-04-15 11:00:00', '2003-04-16 11:00:00', '2003-04-17 11:00:00',
#               '2003-04-17 12:00:00', '2007-03-26 11:00:00', '2007-04-30 11:00:00',
#               '2010-04-17 11:00:00', '2012-05-25 11:00:00', '2012-05-26 11:00:00',
#               '2015-04-20 11:00:00', '2015-04-21 11:00:00'],
#              dtype='datetime64[ns]', name='date',length=26, freq=None)




###################
# re-assignment outliers with thresholds
###################

low, up = outlier_thresholds(df, "DE")

df[((df["DE"] < low) | (df["DE"] > up))]["DE"]

df.loc[((df["DE"] < low) | (df["DE"] > up)), "DE"]

df.loc[(df["DE"] > up), "DE"] = up

df.loc[(df["DE"] < low), "DE"] = low


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# check the result below;

df.loc['2003-04-14 11:00:00']  #up threshold : 0.6533093036631216

#####################
# Train / Test Split
#####################

train = df.loc[df.index < '31-12-2009']
test = df.loc[df.index >= '31-12-2009']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('31-12-2009', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()


#year of the data

#1986
df.loc[(df.index > '01-01-1986') & (df.index < '31-12-1986')] \
    .plot(figsize=(15, 5), title='Year Of Data')
plt.show()

#####################
#Feature Creation
#####################

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)


#####################
#Visualize our Feature / Target Relationship
#####################


#HOUR
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='hour', y='DE')
ax.set_title('Production by Hour')
plt.show()



#MONTH

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='DE', palette='Blues')
ax.set_title('Production by Month')
plt.show()


#YEAR  2003 ve 2011'de artış var,sebepleri araştırmak ?
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='year', y='DE')
ax.set_title('Production by Year')
plt.show()


#Create our Model


train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'DE'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

#our model has been trained with xgboost !!!

#####################
#Feature Importance
#####################

fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()

###################
#Forecast on Test
###################

test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['DE']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.show()


#############
#Score (RMSE)
#############


score = np.sqrt(mean_squared_error(test['DE'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')

#RMSE Score on Test set: 0.07



################
#Calculate Error
################

#Look at the worst and best predicted days


#BEST
test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
error_by_date = test.groupby(test.index.date)['error'].mean().sort_values(ascending=True).head(10)
print(error_by_date)


#WORST
test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
error_by_date = test.groupby(test.index.date)['error'].mean().sort_values(ascending=False).head(10)
print(error_by_date)






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
 plt.show()


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


##################################
# Time Series Cross Validation
##################################
from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()

fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

fold = 0
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]
    train['DE'].plot(ax=axs[fold],
                          label='Training Set',
                          title=f'Data Train/Test Split Fold {fold}')
    test['DE'].plot(ax=axs[fold],
                         label='Test Set')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    fold += 1
plt.show()


##################################
# Forecasting Horizon Explained (TAHMİNLEME UFKU)
##################################

#ÖNEMLİ NOT:::
#Tahmin ufku, tahminlerin hazırlanacağı geleceğe yönelik zamanın uzunluğudur.
#Bunlar genellikle kısa vadeli tahmin ufuklarından (üç aydan az) uzun vadeli
#tahmin ufuklarına (iki yıldan fazla) kadar değişir.



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



##############
#Lag Features
##############

def add_lags(df):
    target_map = df['DE'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df

df = add_lags(df)

df.head()


#                    weekofyear  lag1  lag2  lag3
#date
#1986-01-01 00:00:00           1   NaN   NaN   NaN
#1986-01-01 01:00:00           1   NaN   NaN   NaN
#1986-01-01 02:00:00           1   NaN   NaN   NaN
#1986-01-01 03:00:00           1   NaN   NaN   NaN
#1986-01-01 04:00:00           1   NaN   NaN   NaN


df.tail()

#                     weekofyear  lag1  lag2  lag3
#date
#2015-12-31 19:00:00          53 0.000 0.000 0.000
#2015-12-31 20:00:00          53 0.000 0.000 0.000
#2015-12-31 21:00:00          53 0.000 0.000 0.000
#2015-12-31 22:00:00          53 0.000 0.000 0.000
#2015-12-31 23:00:00          53 0.000 0.000 0.000



################################
#Train Using Cross Validation
###############################


tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()


fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    train = create_features(train)
    test = create_features(test)

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year',
                'lag1','lag2','lag3']
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

    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)

#OUTPUT

#[02:22:47] WARNING: /Users/runner/work/xgboost/xgboost/python-package/build/temp.macosx-10.9-x86_64-cpython-38/xgboost/src/objective/regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
#[0]	validation_0-rmse:0.42560	validation_1-rmse:0.42051
#[100]	validation_0-rmse:0.17200	validation_1-rmse:0.16771
#[200]	validation_0-rmse:0.09369	validation_1-rmse:0.09173
#[300]	validation_0-rmse:0.07560	validation_1-rmse:0.07613
#[400]	validation_0-rmse:0.07107	validation_1-rmse:0.07404
#[500]	validation_0-rmse:0.06943	validation_1-rmse:0.07353
#[600]	validation_0-rmse:0.06857	validation_1-rmse:0.07338
#[700]	validation_0-rmse:0.06796	validation_1-rmse:0.07311
#[800]	validation_0-rmse:0.06749	validation_1-rmse:0.07286
#[900]	validation_0-rmse:0.06709	validation_1-rmse:0.07261
#[999]	validation_0-rmse:0.06677	validation_1-rmse:0.07235
#[02:26:35] WARNING: /Users/runner/work/xgboost/xgboost/python-package/build/temp.macosx-10.9-x86_64-cpython-38/xgboost/src/objective/regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
#[0]	validation_0-rmse:0.42540	validation_1-rmse:0.42436
#[100]	validation_0-rmse:0.17196	validation_1-rmse:0.16975
#[200]	validation_0-rmse:0.09374	validation_1-rmse:0.08851
#[300]	validation_0-rmse:0.07570	validation_1-rmse:0.06901
#[400]	validation_0-rmse:0.07126	validation_1-rmse:0.06520
#[500]	validation_0-rmse:0.06964	validation_1-rmse:0.06424
#[600]	validation_0-rmse:0.06879	validation_1-rmse:0.06382
#[700]	validation_0-rmse:0.06817	validation_1-rmse:0.06360
#[800]	validation_0-rmse:0.06769	validation_1-rmse:0.06343
#[900]	validation_0-rmse:0.06729	validation_1-rmse:0.06330
#[999]	validation_0-rmse:0.06695	validation_1-rmse:0.06326
#[02:30:33] WARNING: /Users/runner/work/xgboost/xgboost/python-package/build/temp.macosx-10.9-x86_64-cpython-38/xgboost/src/objective/regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
#[0]	validation_0-rmse:0.42537	validation_1-rmse:0.43117
#[100]	validation_0-rmse:0.17185	validation_1-rmse:0.17678
#[200]	validation_0-rmse:0.09356	validation_1-rmse:0.09602
#[300]	validation_0-rmse:0.07552	validation_1-rmse:0.07642
#[400]	validation_0-rmse:0.07116	validation_1-rmse:0.07226
#[500]	validation_0-rmse:0.06953	validation_1-rmse:0.07083
#[600]	validation_0-rmse:0.06867	validation_1-rmse:0.07036
#[700]	validation_0-rmse:0.06806	validation_1-rmse:0.07006
#[800]	validation_0-rmse:0.06759	validation_1-rmse:0.06983
#[900]	validation_0-rmse:0.06719	validation_1-rmse:0.06957
#[999]	validation_0-rmse:0.06687	validation_1-rmse:0.06948
#[02:33:55] WARNING: /Users/runner/work/xgboost/xgboost/python-package/build/temp.macosx-10.9-x86_64-cpython-38/xgboost/src/objective/regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
#[0]	validation_0-rmse:0.42557	validation_1-rmse:0.42400
#[100]	validation_0-rmse:0.17189	validation_1-rmse:0.16791
#[200]	validation_0-rmse:0.09350	validation_1-rmse:0.08695
#[300]	validation_0-rmse:0.07544	validation_1-rmse:0.06808
#[400]	validation_0-rmse:0.07113	validation_1-rmse:0.06477
#[500]	validation_0-rmse:0.06955	validation_1-rmse:0.06383
#[600]	validation_0-rmse:0.06872	validation_1-rmse:0.06365
#[700]	validation_0-rmse:0.06812	validation_1-rmse:0.06350
#[800]	validation_0-rmse:0.06765	validation_1-rmse:0.06336
#[838]	validation_0-rmse:0.06749	validation_1-rmse:0.06333
#[02:35:28] WARNING: /Users/runner/work/xgboost/xgboost/python-package/build/temp.macosx-10.9-x86_64-cpython-38/xgboost/src/objective/regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
#[0]	validation_0-rmse:0.42552	validation_1-rmse:0.42304
#[100]	validation_0-rmse:0.17179	validation_1-rmse:0.16737
#[200]	validation_0-rmse:0.09332	validation_1-rmse:0.08699
#[300]	validation_0-rmse:0.07527	validation_1-rmse:0.06884
#[400]	validation_0-rmse:0.07098	validation_1-rmse:0.06578
#[500]	validation_0-rmse:0.06941	validation_1-rmse:0.06495
#[600]	validation_0-rmse:0.06857	validation_1-rmse:0.06464
#[700]	validation_0-rmse:0.06797	validation_1-rmse:0.06438
#[800]	validation_0-rmse:0.06751	validation_1-rmse:0.06417
#[888]	validation_0-rmse:0.06717	validation_1-rmse:0.06417



print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')

#Score across folds 0.0665
#Fold scores:[0.07235148182109119, 0.06324105580856021,
#             0.06948118609081526, 0.06330820862856405,
#             0.06413656231072157]


###############
#Predicting the Future
###############


# 1) Tüm veriler üzerinde yeniden eğitim yap

# 2)Geleceği tahmin etmek için gelecekteki tarih aralıklarına yönelik boş bir veri çerçevesine ihtiyacımız var.

# 3)Bu tarihleri feature creation code + lag creation ile çalıştır

# Tüm veriler üzerinde yeniden eğitim yap

df = create_features(df)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
            'lag1','lag2','lag3']
TARGET = 'DE'

X_all = df[FEATURES]
y_all = df[TARGET]

reg = xgb.XGBRegressor(base_score=0.5,
                       booster='gbtree',
                       n_estimators=500,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_all, y_all,
        eval_set=[(X_all, y_all)],
        verbose=100)

#[02:51:55] WARNING: /Users/runner/work/xgboost/xgboost/python-package/build/temp.macosx-10.9-x86_64-cpython-38/xgboost/src/objective/regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
#[0]	validation_0-rmse:0.42544
#[100]	validation_0-rmse:0.17169
#[200]	validation_0-rmse:0.09318
#[300]	validation_0-rmse:0.07511
#[400]	validation_0-rmse:0.07090
#[499]	validation_0-rmse:0.06935
#Out[59]:
#XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
#             colsample_bylevel=None, colsample_bynode=None,
#             colsample_bytree=None, early_stopping_rounds=None,
#             enable_categorical=False, eval_metric=None, feature_types=None,
#             gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
#             interaction_constraints=None, learning_rate=0.01, max_bin=None,
#             max_cat_threshold=None, max_cat_to_onehot=None,
#             max_delta_step=None, max_depth=3, max_leaves=None,
#             min_child_weight=None, missing=nan, monotone_constraints=None,
#             n_estimators=500, n_jobs=None, num_parallel_tree=None,
#             objective='reg:linear', predictor=None, ...)


df.index.max()

# Timestamp('2015-12-31 23:00:00')

## Gelecekteki veri çerçevesini oluştur ( BİZ BURADA 1 YILLIK BİR ÇERÇEVE OLUŞTURDUK)

future = pd.date_range('2015-12-31','2016-12-31', freq='1h')
future_df = pd.DataFrame(index=future)
future_df['isFuture'] = True
df['isFuture'] = False
df_and_future = pd.concat([df, future_df])
df_and_future = create_features(df_and_future)
df_and_future = add_lags(df_and_future)


future_w_features = df_and_future.query('isFuture').copy()


###################
#Predict the future
###################

future_w_features['pred'] = reg.predict(future_w_features[FEATURES])

future_w_features['pred'].plot(figsize=(10, 5),
                               color=color_pal[4],
                               ms=1,
                               lw=1,
                               title='Future Predictions')
plt.show()


#########################
#Saving Model For later
#########################


# Save model
reg.save_model('model.json')

!ls -lh

reg_new = xgb.XGBRegressor()
reg_new.load_model('model.json')
future_w_features['pred'] = reg_new.predict(future_w_features[FEATURES])
future_w_features['pred'].plot(figsize=(10, 5),
                               color=color_pal[4],
                               ms=1, lw=1,
                               title='Future Predictions')



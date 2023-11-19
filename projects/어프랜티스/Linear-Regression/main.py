import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

total_df = pd.read_csv('./elec_power.csv')

"""
데이터 수집 및 탐색
"""
# total_df.plot(kind='hist',  y='AT')
# total_df.plot(kind='hist',  y='V')
# total_df.plot(kind='hist',  y='AP')
# total_df.plot(kind='hist',  y='RH')
# total_df.plot(kind='hist',  y='EP')
# plt.show()

data_X = total_df.loc[:, ['AT', 'V', 'AP', 'RH']]
data_Y = total_df["EP"]

# fig, ax = plt.subplots(1, 4, figsize=(15, 5))
# correlation = total_df['AT'].corr(total_df['V'])
# print('correlation between AT and V: ', correlation)
# correlation = total_df['AT'].corr(total_df['AP'])
# print('correlation between AT and AP: ', correlation)
# correlation = total_df['AT'].corr(total_df['RH'])
# print('correlation between AT and RH: ', correlation)
# correlation = total_df['AT'].corr(total_df['EP'])
# print('correlation between AT and EP: ', correlation)
# ax[0].scatter(total_df['AT'], total_df['V'], label='AT vs. V')
# ax[0].legend()
# ax[1].scatter(total_df['AT'], total_df['AP'], label='AT vs. AP')
# ax[1].legend()
# ax[2].scatter(total_df['AT'], total_df['RH'], label='AT vs. RH')
# ax[2].legend()
# ax[3].scatter(total_df['AT'], total_df['EP'], label='AT vs. EP')
# ax[3].legend()

# correlation = total_df['V'].corr(total_df['AT'])
# print('correlation between V and AT: ', correlation)
# correlation = total_df['V'].corr(total_df['AP'])
# print('correlation between V and AP: ', correlation)
# correlation = total_df['V'].corr(total_df['RH'])
# print('correlation between V and RH: ', correlation)
# correlation = total_df['V'].corr(total_df['EP'])
# print('correlation between V and EP: ', correlation)
# ax[0].scatter(total_df['V'], total_df['AT'], label='V vs. AT')
# ax[0].legend()
# ax[1].scatter(total_df['V'], total_df['AP'], label='V vs. AP')
# ax[1].legend()
# ax[2].scatter(total_df['V'], total_df['RH'], label='V vs. RH')
# ax[2].legend()
# ax[3].scatter(total_df['V'], total_df['EP'], label='V vs. EP')
# ax[3].legend()

# correlation = total_df['AP'].corr(total_df['AT'])
# print('correlation between AP and AT: ', correlation)
# correlation = total_df['AP'].corr(total_df['V'])
# print('correlation between AP and V: ', correlation)
# correlation = total_df['AP'].corr(total_df['RH'])
# print('correlation between AP and RH: ', correlation)
# correlation = total_df['AP'].corr(total_df['EP'])
# print('correlation between AP and EP: ', correlation)
# ax[0].scatter(total_df['AP'], total_df['AT'], label='AP vs. AT')
# ax[0].legend()
# ax[1].scatter(total_df['AP'], total_df['V'], label='AP vs. V')
# ax[1].legend()
# ax[2].scatter(total_df['AP'], total_df['RH'], label='AP vs. RH')
# ax[2].legend()
# ax[3].scatter(total_df['AP'], total_df['EP'], label='V vs. EP')
# ax[3].legend()

# correlation = total_df['RH'].corr(total_df['AT'])
# print('correlation between RH and AT: ', correlation)
# correlation = total_df['RH'].corr(total_df['V'])
# print('correlation between RH and V: ', correlation)
# correlation = total_df['RH'].corr(total_df['AP'])
# print('correlation between RH and AP: ', correlation)
# correlation = total_df['RH'].corr(total_df['EP'])
# print('correlation between RH and EP: ', correlation)
# ax[0].scatter(total_df['RH'], total_df['AT'], label='RH vs. AT')
# ax[0].legend()
# ax[1].scatter(total_df['RH'], total_df['V'], label='RH vs. V')
# ax[1].legend()
# ax[2].scatter(total_df['RH'], total_df['AP'], label='RH vs. AP')
# ax[2].legend()
# ax[3].scatter(total_df['RH'], total_df['EP'], label='RH vs. EP')
# ax[3].legend()
# plt.show()

"""
데이터 프리프로세싱
"""
# plt.figure(figsize=(11, 11))
# sns.pairplot(total_df.drop(columns=['EP']));
# plt.show()

sk_lin_model = LinearRegression()
my_model = sk_lin_model.fit(data_X, data_Y)
#
# r2_score = my_model.score(data_X, data_Y)

# print('R2 score is ', r2_score)
# print('intercept (b0) ', my_model.intercept_)
# coef_names = ['b1', 'b2', 'b3', 'b4']
# print(pd.DataFrame({'Predictor': data_X.columns,
#                     'coefficient Name': coef_names,
#                     'coefficient Value': my_model.coef_}))


# import statsmodels.api as sm
#
# X = sm.add_constant(total_df.loc[:, ['AT', 'V', 'AP', 'RH']])
# lin_model = sm.OLS(total_df['EP'], X)
# my_model = lin_model.fit()
# print(my_model.summary())
# print(my_model.params)


"""
데이터 분할
"""
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(data_X, data_Y, \
                                                      test_size=0.3, random_state=88, shuffle=True)

lin_model = LinearRegression()
my_model = lin_model.fit(X_train, Y_train)
print('R2 score is ', my_model.score(X_valid, Y_valid))
print('model coefficients:\n', my_model.coef_, '\nintercept: ', my_model.intercept_)


test_df = Y_valid.reset_index(drop=True).to_frame()
test_df.plot(label="test")

pred_df = pd.DataFrame(my_model.predict(X_valid), columns=['Predict_EP'])
pred_df.plot(label="predict")

(test_df["EP"] - pred_df["Predict_EP"]).to_frame().plot(label="diff")
plt.xlabel("index")
plt.show()

RMSE = mean_squared_error(data_Y, my_model.predict(data_X), squared =False)
print('the root mean square error is ', RMSE)

# for coef, var in enumerate(total_df.columns[1:-1]):
#     print(var, '\t', round(my_model.coef_[coef], 5), '\twith range ', round(float(total_df[var].max() - total_df[var].min()), 2))

# # Visualize three distributions that have the same mean but different standard deviations
# col_num = total_df.shape[1]
# col = ["index", "AT", "V", "AP", "RH", "EP"]
# fig, ax = plt.subplots(1, col_num , figsize = (15, 3), sharey = True)
# for i in range(col_num):
#     ax[i].hist(total_df.iloc[:, i], bins = 50)
#     ax[i].set_title('variable ' + col[i])
#     ax[0].set_ylabel('count')
# plt.show()
#
# # Visualize three distributions with StandardScaler()
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# colnames = total_df.columns
# distributions = pd.DataFrame(scaler.fit_transform(total_df))
#
# distributions.columns = colnames
# fig, ax = plt.subplots(1, distributions.shape[1], figsize = (15, 3), sharey = True)
# for i in range(distributions.shape[1]):
#     _ = ax[i].hist(distributions.iloc[:, i], bins = 50)
#     _ = ax[i].set_title('variable ' + col[i])
#     _ = ax[0].set_ylabel('count')
# plt.show()

"""
데이터 스케일링
"""
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_scaled = scaler.fit_transform(X_train)
# X_scaled = pd.DataFrame(X_scaled)
# X_scaled.columns = total_df.columns[1:-1]
# X_scaled.describe().T
#
# lin_model = LinearRegression()
# my_model4 = lin_model.fit(X_scaled, Y_train)
# print('R2 score is ', my_model4.score(X_scaled, Y_train))
# print('model coefficients:\n', my_model4.coef_, '\nintercept: ', my_model4.intercept_)
# RMSE = mean_squared_error(data_Y, my_model4.predict(X_scaled), squared=False)
# print('the root mean square error is ', RMSE)
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 
 'B', 'LSTAT']
np.set_printoptions(precision = 3, suppress = True)
# d = repr(data[:5]) # Nice printable version of boston.data
# print(d)
bos = pd.DataFrame(data)
# bos.head()
bos.columns = feature_names
# bos.head()

crime_indus = bos['CRIM'].corr(bos['INDUS'])
pov_age = bos['LSTAT'].corr(bos['AGE'])
crime_tax = bos['CRIM'].corr(bos['TAX'])

print("corr crime/industry: %3.3f, corr poverty/age: %3.3f, corr crime/tax: %3.3f" 
      % (crime_indus, pov_age, crime_tax))

bos['PRICE'] = target
print("Correlation between property prices and poverty levels: %3.3f"
     % bos['PRICE'].corr(bos['LSTAT']))

X = bos['LSTAT'].values.reshape(-1, 1)
Y = bos['PRICE'].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state=0)
all_rows = X.shape[0]
test_rows = X_test.shape[0]
print("%%age of data used for test: %3.2f%%" % (test_rows / all_rows * 100.0))

lm = LinearRegression()
lm.fit(X_train, Y_train)
print("Coefficient: %3.4f, Intercept: %3.4f." % (lm.coef_, lm.intercept_))

Y_pred_train = lm.predict(X_train)
Y_pred_test = lm.predict(X_test)
train_mse = np.sqrt(metrics.mean_squared_error(Y_train, Y_pred_train))
test_mse = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_test))
print("RMSE for training data: %3.4f. RMSE for testing data: %3.4f." % (train_mse, test_mse))

train_r2 = metrics.r2_score(Y_train, Y_pred_train)
test_r2 = metrics.r2_score(Y_test, Y_pred_test)
print("R2 for training data: %3.4f. R2 for testing data: %3.4f." % (train_r2, test_r2))

price1 = lm.predict(np.array([[10.0]]))
price2 = lm.predict(np.array([[25.0]]))
print("Price at 10%% poverty level is $%3.2f. Price at 25%% poverty level is $%3.2f"
      %(price1 * 1000, price2 * 1000))

p1 = -0.9507 * 10 + 34.6372
p2 = -0.9507 * 25 + 34.6372
print("Price at 10%% poverty level is $%3.2f. Price at 25%% poverty level is $%3.2f"
      %(p1 * 1000, p2 * 1000))
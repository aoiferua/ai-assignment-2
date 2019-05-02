# Import some necessary python modules
import warnings

warnings.filterwarnings('ignore')
import sys
import matplotlib
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Above is a special style template for matplotlib, highly useful for visualizing time series data
from pylab import rcParams
from numpy.random import seed
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
import math
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD

# Setting seed value
seed(1)
set_random_seed(2)

# Allow Nike dataset to be displayed in full without being shortened
np.set_printoptions(threshold=sys.maxsize)

# Read the Nike stock data from .csv file using pandas
nike = pd.read_csv('stock-20050101-to-20171231/NKE_2006-01-01_to_2018-01-01.csv',
                   index_col='Date', parse_dates=['Date'])

# Display Nike data features
print(nike.columns)

# Preview the first few lines of the loaded data
print(nike.head())

# Summarise the Nike dataset
print(nike.describe())

# Look for empty data within the Nike dataset
null_columns = nike.columns[nike.isnull().any()]
print(nike[nike.isnull().any(axis=1)][null_columns])

# Data is missing on features Open and Low
# Get average from both columns and divide by two to fill empty values with
value_for_open = nike["Open"].mean()
value_for_low = nike["Low"].mean()
main_value = (value_for_open + value_for_low) / 2
main_value_used = round(main_value)

# Filling empty data with the average value
nike.fillna(main_value_used, inplace=True)

# Checking if empty data has been filled
null_columns = nike.columns[nike.isnull().any()]
print(nike[nike.isnull().any(axis=1)][null_columns])

# Visualising the nike data set from 2015 to 2017
nike['2015':'2017'].plot(subplots=True, figsize=(10, 12))
plt.title('nike stock attributes from 20015 to 2017')
plt.savefig('stocks.png')
plt.show()

# FEATURE SELECTION

# Data selection
x_data = nike.values[:, 0:5]
y_data = nike.values[:, 5]

# Univariate feature extraction
test = SelectKBest(score_func=chi2, k=3)
fit = test.fit(x_data, y_data)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(x_data)
# summarize selected features and show best 3
print(features[0:3, :])

# Feature extraction, PCA
pca = PCA(n_components=3)
fit = pca.fit(x_data)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

# feature importance
model = ExtraTreesClassifier()
model.fit(x_data, y_data)
print(model.feature_importances_)


# Best Algorithm Boxplots
def plot_models(model_results, model_names):
    # Box plots to compare algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(model_results)
    ax.set_xticklabels(model_names)
    plt.savefig("model_accuracy_assignment2.svg")


# Experimenting with a range of machine learning algorithm to discover the best performing algorithm
def selection_of_models():
    # Selecting data from "Low" column
    x = nike.values[:, 0:3]
    y = nike.values[:, 3]
    y = y.astype('int')

    # Setting validation size, to determine how to split the data
    validation_size = 0.08
    fixed_seed = 7
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size,
                                                                    random_state=fixed_seed)

    # 10 Fold cross validation will be used to estimate accuracy
    scoring = "accuracy"

    # Spot Check algorithms
    # Find which performs best
    models = []
    models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
    models.append(("LDA", LinearDiscriminantAnalysis()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier()))
    models.append(("NB", GaussianNB()))
    models.append(("SVM", SVC(gamma="auto")))

    # Evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=3, random_state=fixed_seed)
        cv_results = model_selection.cross_val_score(model, x_train, y_train,
                                                     cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    plot_models(results, names)

    # Make predictions on testing dataset
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_validation)
    ac = (accuracy_score(y_validation, predictions))

    print("\nKnn accuracy = {}.".format(ac))

    # Make predictions on testing dataset
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    predictions_lda = lda.predict(x_validation)
    ac = (accuracy_score(y_validation, predictions_lda))
    # cm = (confusion_matrix(y_validation, predictions))
    # cr = (classification_report(y_validation, predictions))

    print("Lda accuracy = {}.".format(ac))

    # Make predictions on testing dataset
    svm = SVC(gamma="auto")
    svm.fit(x_train, y_train)
    predictions = svm.predict(x_validation)
    ac = (accuracy_score(y_validation, predictions))

    print("Svm accuracy = {}.".format(ac))


selection_of_models()

# Predicting the nike stocks on feature High
# ARIMA Model
rcParams['figure.figsize'] = 16, 6
model = ARIMA(nike["High"].diff().iloc[1:].values, order=(2, 1, 0))
result = model.fit()
print(result.summary())
result.plot_predict(start=700, end=1000)
plt.show()
# ARIMA model root mean squared error
rmse = math.sqrt(mean_squared_error(nike["High"].diff().iloc[700:1001].values, result.predict(start=700, end=1000)))
print("The root mean squared error is {}.".format(rmse))


# Some functions to help out with when ploting dataset
def plot_predictions(test, predicted):
    plt.plot(test, color='red', label='Real Nike Stock Price')
    plt.plot(predicted, color='blue', label='Predicted Nike Stock Price')
    plt.title('Nike Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Nike Stock Price')
    plt.legend()
    plt.show()


# root mean squared error
def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))


# Checking for missing values
training_set = nike[:'2016'].iloc[:, 1:2].values
test_set = nike['2017':].iloc[:, 1:2].values

# We have chosen 'High' feature to test on.
nike["High"][:'2016'].plot(figsize=(16, 4), legend=True)
nike["High"]['2017':].plot(figsize=(16, 4), legend=True)
plt.legend(['Training set (Before 2017)', 'Test set (2017 and beyond)'])
plt.title('Nike stock price')
plt.show()

# Scaling the training set
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# create a data structure with 60 time steps and 1 output
X_train = []
y_train = []
for i in range(60, 2769):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping X_train for modelling
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
# Fitting to the training set
regressor.fit(X_train, y_train, epochs=50, batch_size=32)

# Now to get the test set ready in a similar way as the training set.
dataset_total = pd.concat((nike["High"][:'2016'], nike["High"]['2017':]), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# Preparing X_test and predicting the prices
X_test = []
for i in range(60, 311):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the results for LSTM
plot_predictions(test_set, predicted_stock_price)

# Evaluating our model
return_rmse(test_set, predicted_stock_price)

# The GRU architecture
regressorGRU = Sequential()
# First GRU layer with Dropout regularisation
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Second GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Third GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Fourth GRU layer
regressorGRU.add(GRU(units=50, activation='tanh'))
regressorGRU.add(Dropout(0.2))
# The output layer
regressorGRU.add(Dense(units=1))
# Compiling the RNN
regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')
# Fitting to the training set
regressorGRU.fit(X_train, y_train, epochs=50, batch_size=150)

# Preparing X_test and predicting the prices
X_test = []
for i in range(60, 311):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
GRU_predicted_stock_price = regressorGRU.predict(X_test)
GRU_predicted_stock_price = sc.inverse_transform(GRU_predicted_stock_price)

# Visualizing the results for GRU
plot_predictions(test_set, GRU_predicted_stock_price)

# Evaluating GRU
return_rmse(test_set, GRU_predicted_stock_price)

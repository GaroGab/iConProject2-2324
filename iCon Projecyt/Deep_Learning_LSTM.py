
from utils import *
import numpy as np
import pandas as pd
from pandas import *
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from keras import metrics
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def trainTestSplit(x_train,x_test,y_train,y_test):
    print("TRAIN-TEST SPLIT")
    x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2)


def loadTrainingData(ticker, START, TODAY):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    df = data
    print(df.head())
    df = df.drop(['Date', 'Adj Close'], axis=1)
    return df

# Define a function to load the dataset
def trainModel(data: DataFrame, days_before: int, optimizer, loss):
    # Splitting the dataset into training (70%) and testing (30%) set
    train, test = splitData(data, 0.70)
    print(train.shape)
    print(test.shape)

    train.head()

    # Using MinMax scaler for normalization of the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_close = train.iloc[:, 4:5].values
    test_close = test.iloc[:, 4:5].values
    print("TEST DATAFRAME:", test_close)
    print("days_before: ",days_before, "optimizer: ", optimizer, "loss: ", loss)
    data_training_array = scaler.fit_transform(train_close)

    x_train = []
    y_train = []

    for i in range(days_before, data_training_array.shape[0]):
        x_train.append(data_training_array[i - days_before: i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    print("x shape: ", x_train.shape)
    print("y shape: ", y_train.shape)

    # ML Model (LSTM)
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True
                   , input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=1))

    print(model.summary())
    # Training the model
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.MeanAbsoluteError()])
    callback = EarlyStopping(monitor='loss', patience=20)
    # This callback will stop the training when there is no improvement in
    # the loss for 20 consecutive epochs.
    model.fit(x_train, y_train, epochs=70, callbacks=callback)
    model_name = f'model_{optimizer}_{loss}_{days_before}.h5'
    model.save(model_name)

    past_x_days = pd.DataFrame(train_close[-days_before:])
    test_df = pd.DataFrame(test_close)
    # Defining the final dataset for testing by including last 'days_before' columns
    # of the training dataset to get the prediction from the 1st column of the testing dataset.
    final_df = pd.concat([past_x_days, test_df])
    final_df.head()
    input_data = scaler.fit_transform(final_df)
    print(input_data)
    print("input_data shape: ", input_data.shape)
    # Testing the model
    x_test = []
    y_test = []
    # x_test acquisisce diverse serie di numeri, ognuna grande quanto days_before, ed y_test,
    # per ognuna di esse, il valore del giorno 'i' dopo
    for i in range(days_before, input_data.shape[0]):
        x_test.append(input_data[i - days_before: i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    print(x_test.shape)
    print(y_test.shape)
    # Making prediction and plotting the graph of predicted vs actual values
    y_pred = testModel(input_data,model,days_before)

    #Evaluate the model
    evaluateAbsError(y_test,y_pred)
    show_r2_score(y_test,y_pred)
    scale_factor = 1 / 0.00041967
    y_pred = y_pred * scale_factor
    y_test = y_test * scale_factor
    plotComparingResults(y_test,y_pred)

def evaluateAbsError(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mae_percentage = (mae / np.mean(actual)) * 100
    print("Mean absolute error on test set: {:.2f}%".format(mae_percentage))

def show_r2_score(actual, predicted):
    r2 = r2_score(actual, predicted)
    print("R2 score:", r2)
    fig, ax = plt.subplots()
    ax.barh(0, r2, color='skyblue')
    ax.set_xlim([-1, 1])
    ax.set_yticks([])
    ax.set_xlabel('R2 Score')
    ax.set_title('R2 Score')
    # Adding the R2 score value on the bar
    ax.text(r2, 0, f'{r2:.2f}', va='center', color='black')
    plt.show()

    plt.scatter(actual, predicted)
    plt.plot([min(actual), max(actual)], [min(predicted), max(predicted)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'R2 Score: {r2:.2f}')
    plt.show()

'''
START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
data = loadTrainingData('AAPL',START,TODAY)
###print(trainModel(data,5,'adam','mean_squared_error'))
'''



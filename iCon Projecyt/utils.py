import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import yfinance as yf
def load_data(ticker, START, TODAY):
    data = yf.download(ticker, START, TODAY)
    data.index = [pd.to_datetime(str(s)[:10]) for s in data.index]
    return data

def splitData(data: pd.DataFrame, threshold : float) -> (pd.DataFrame, pd.DataFrame):
    i=int(len(data) * threshold)
    train_set = data.iloc[:i]
    test_set = data.iloc[(i+1):]
    return train_set, test_set

def testModel(input_data: pd.DataFrame, model, days_before: int):
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
    y_pred = model.predict(x_test)
    return y_pred

def plotComparingResults(original,predict):
    plt.figure(figsize=(12, 6))
    plt.plot(original, 'b', label="Original Price")
    plt.plot(predict, 'r', label="Predicted Price")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def modelPrediction(data: pd.DataFrame, days_before, model_name : str):
    print(data)
    model = load_model(model_name)
    x_test = data.iloc[:days_before]['Close']
    y_test = data.iloc[days_before+1]['Close']
    print(x_test)
    print("value that has to be predicted:", y_test)
    x_test, y_test = np.array([x_test.to_numpy()]), np.array(y_test)
    print(x_test.shape)
    print(y_test.shape)
    prediction = model.predict(np.array(x_test))
    print("the prediction is: ", prediction)
    
    
def predict(model, feature_values: pd.Series) -> float:
    '''
        Predict the stock price of values using the model and the feature_values.
        
        Parameters:
            model: keras model
            feature_values: pd.Series       # The feature values used to predict the new value (must be of the same length as the input of the model)
            
        Returns:
            float: the predicted stock price
    '''
    x_test = np.array([feature_values.to_numpy()])      # Convert the feature_values to a numpy array
    prediction = model.predict(np.array(x_test))        # Predict the stock price
    return float(prediction[0][0])                      # Return the predicted stock price
    

def predict_in_sample(model, data: pd.DataFrame, days_before: int, day_to_predict: str) -> float:
    '''
        Predict the stock price of the day_to_predict using the model and the data.
        
        Parameters:
            model: keras model
            data: pd.DataFrame
            days_before: int
            day_to_predict: str   # Format: 'YYYY-MM-DD'
        
        Returns:
            float: the predicted stock price
    '''
    x_test = data.loc[:pd.to_datetime(day_to_predict)].iloc[-days_before:]['Close']     # Get the last days_before days
    return predict(model, x_test)                                                       # Predict the stock price


def predict_all(model, data: pd.DataFrame, days_before: int) -> pd.Series:
    '''
        Predict the stock price of all the days in the data using the model.
        
        Parameters:
            model: keras model
            data: pd.DataFrame
            days_before: int
            
        Returns:
            pd.DataFrame: a DataFrame containing the predicted stock prices
    '''
    predictions = pd.Series([])
    for i in range(days_before, len(data)):
        x_test = data.iloc[i - days_before: i]['Close']
        x_test = np.array([x_test.to_numpy()])
        prediction = model.predict(np.array(x_test))
        predictions.loc[data.index[i]] = float(prediction[0][0])
    return predictions


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM
import streamlit as st
from PIL import Image
################    STREAMLIT    ##############
st.title('Crypto stocks prediction App')

cryptos = ('BTC', 'ETH','DOGE')
selected_crypto = st.selectbox('Select crypto coin ticker ', cryptos)

curr = ('USD', 'EUR')
selected_curr = st.selectbox('Select currency  ', curr)

model = ('LSTM', 'Linear regression',"Random Forest")
m = st.selectbox('Select model', model)
stock_data = yf.download(selected_crypto+'-'+selected_curr)
if(selected_crypto=="DOGE"):
    image = Image.open(selected_crypto+'.jpg')
else:
    image = Image.open(selected_crypto+'.png')

if(selected_crypto=="ETH"):
    cap="Ethereum logo"
if(selected_crypto=="DOGE"):
    cap="Doge logo"
else:
    cap="Bitcoin logo"
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image(image, caption=cap)
st.subheader("Stock Data for "+selected_crypto)
st.dataframe(stock_data)



################    LINEAR REGRESSION    ##############

if(m=="Linear regression"):

    # Create a new dataframe with just the 'Close' price
    df = pd.DataFrame(stock_data['Close'])

    # Create a new column for the days
    df['Days'] = np.arange(len(df))

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Create the linear regression model
    lr_model = LinearRegression()

    # Fit the model to the training data
    X_train = train_df[['Days']]
    y_train = train_df['Close']
    lr_model.fit(X_train, y_train)

    # Predict the BTC prices for the testing data
    X_test = test_df[['Days']]
    y_test = test_df['Close']
    y_pred = lr_model.predict(X_test)
    rmse = np.sqrt(np.mean(y_pred - y_test)**2)
    st.title('RMSE for '+selected_crypto)
    st.write(rmse)

    # Plot the predicted BTC prices
    fig=plt.figure()

    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.xlabel('Days')
    plt.ylabel(selected_crypto+'Price')
    plt.title(selected_crypto+' Price Prediction using Linear Regression')
    plt.legend()
    plt.show()
    
################    RANDOM FOREST    ##############

elif (m=="Random Forest"):

    # Create a new dataframe with just the 'Close' price
    df = pd.DataFrame(stock_data['Close'])

    # Create a new column for the days
    df['Days'] = np.arange(len(df))

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Create the Random Forest Regression model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model to the training data
    X_train = train_df[['Days']]
    y_train = train_df['Close']
    rf_model.fit(X_train, y_train)

    # Predict the BTC prices for the testing data
    X_test = test_df[['Days']]
    y_test = test_df['Close']
    y_pred = rf_model.predict(X_test)
    rmse = np.sqrt(np.mean(y_pred - y_test)**2)
    st.title('RMSE for '+selected_crypto)
    st.write(rmse)

    # Plot the predicted BTC prices
    fig=plt.figure()
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.xlabel('Days')
    plt.ylabel(selected_crypto+'Price')
    plt.title(selected_crypto+' Price Prediction using Random Forest Regression')
    plt.legend()
    plt.show()






    ################    LSTM        ##############
else:
    df = pd.DataFrame(stock_data['Close'])

    # Create a new column for the days
    df['Days'] = np.arange(len(df))

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_df[['Close']])
    test_data = scaler.transform(test_df[['Close']])

    # Create the training data and labels
    X_train = []
    y_train = []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the data to fit LSTM model input shape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = load_model(selected_crypto+".h5")


   
    # Predict the BTC prices for the testing data
    inputs = df['Close'][len(df) - len(test_df) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = []
    
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test= test_df['Close']

    rmse = np.sqrt(np.mean(y_pred[0] -y_test) )
    print(len(y_pred),len(y_test))
    st.title('RMSE for '+selected_crypto)
    st.write(rmse)


    # Plot the predicted BTC prices
    fig=plt.figure()
    plt.plot(test_df.index, test_df['Close'], label='Actual')
    plt.plot(test_df.index,y_pred, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel(selected_crypto+'Price')
    plt.title(selected_crypto+'Price Prediction using LSTM')
    plt.legend()
    plt.show()
        
st.title('Stock Plot for '+selected_crypto)
st.plotly_chart(fig)

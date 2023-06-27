from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
N=1000

model = Sequential()
model.add(LSTM(64, input_shape=(N,8),
               return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(6))
model.compile(loss='mse',optimizer='rmsprop')
model.summary()
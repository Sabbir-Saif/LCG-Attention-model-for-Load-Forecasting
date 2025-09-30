import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import copy
from pylab import rcParams
#import seaborn as sns
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM,GRU
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError,R2Score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

%matplotlib inline
rcParams['figure.figsize'] = 15, 10
warnings.filterwarnings('ignore')
# import and visualise data
parser = lambda dates: pd.datetime.strptime(dates,'%Y')
data = pd.read_csv("C:\\Users\\User\\Downloads\\load1.csv",
                   engine='python')
#data['Year']=pd.to_datetime(data['Date']).dt.year

# Avoiding Unnecessary Data
#data=data.iloc[:34200,:]
#data['Time'] = data['Time'].replace('24:00:00', '00:00:00')

# Outlier Detection and Trimming
mean = np.mean(data['load'])
std =np.std(data['load'])
upper=mean+3*std
lower=mean-3*std
data.loc[(data['load']>upper)|(data['load']<lower)]

#Outliers removed
data=data.loc[(data['load']<upper) & (data['load']>lower)]

# TSD
from statsmodels.tsa.seasonal import seasonal_decompose

data['resid']=seasonal_decompose(data['load'], model='additive', period=24).resid
data['trend']=seasonal_decompose(data['load'], model='additive', period=24).trend

# Incorporating Lag Features

data['t-lag_1_hour']=data['load'].shift(1)
data['t-lag_2_hour']=data['load'].shift(2)
data['t-lag_3_hour']=data['load'].shift(3)
data['t-lag_4_hour']=data['load'].shift(4)
data['t-lag_5_hour']=data['load'].shift(5)
data['t-lag_6_hour']=data['load'].shift(6)
data['t-lag_7_hour']=data['load'].shift(7)
data['t-lag_8_hour']=data['load'].shift(8)
data['t-lag_9_hour']=data['load'].shift(9)
data['t-lag_10_hour']=data['load'].shift(10)
data['t-lag_11_hour']=data['load'].shift(11)
data['t-lag_12_hour']=data['load'].shift(12)
data['t-lag_13_hour']=data['load'].shift(13)
data['t-lag_14_hour']=data['load'].shift(14)
data['t-lag_15_hour']=data['load'].shift(15)
data['t-lag_16_hour']=data['load'].shift(16)
data['t-lag_17_hour']=data['load'].shift(17)
data['t-lag_18_hour']=data['load'].shift(18)
data['t-lag_19_hour']=data['load'].shift(19)
data['t-lag_20_hour']=data['load'].shift(20)
data['t-lag_21_hour']=data['load'].shift(21)
data['t-lag_22_hour']=data['load'].shift(22)
data['t-lag_23_hour']=data['load'].shift(23)
data['t-lag_24_hour']=data['load'].shift(24)
data['t-lag_48_hour']=data['load'].shift(48)
data['t-lag_72_hour']=data['load'].shift(72)

data.fillna(0,inplace=True)
# Data preprocessing

scaler = MinMaxScaler()

df_for_training = data[['load','t-lag_1_hour','t-lag_2_hour','t-lag_24_hour','trend']]
#df_for_training=df_for_training.values.reshape(-1,1) #to obtain a 1D array must be reshape.(-1,1)
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)
y_scaler = scaler.fit(df_for_training[['load']])


#creating trainable arrays
trainX = []
trainY = []
n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 50  # Number of past days we want to use to predict the future.

#PROCESSING INTO TRAINABLE values

for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 1:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX,trainY=np.array(trainX),np.array(trainY)
print(trainX.shape,trainY.shape)
print(df_for_training)
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, GRU, Input, Concatenate, Attention, Reshape
from tensorflow.keras.models import Model, Sequential

# Define input shape
input_shape = (trainX.shape[1], trainX.shape[2])
# Define inputs
input1 = Input(shape=input_shape)
input2 = Input(shape=input_shape)
input3 = Input(shape=input_shape)
input4 = Input(shape=input_shape)
# Model 1: CNN
model1 = Sequential()
model1.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape))
model1.add(MaxPooling1D(pool_size=2))
model1.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model1.add(MaxPooling1D(pool_size=2))
model1.add(Conv1D(filters=48, kernel_size=3, activation='relu'))
model1.add(MaxPooling1D(pool_size=2))
model1.add(Flatten())
model1.add(Dropout(0.1))
# Call model1 on input1
output1 = model1(input1)

# Model 2: LSTM + GRU
model2 = Sequential()
model2.add(LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape))
model2.add(GRU(64, activation='relu'))
model2.add(Dropout(0.1))
# Call model2 on input2
output2 = model2(input2)

from MultiCustom_Attention import MultiHeadAttention
# Apply attention to each output
attention_output1 = Reshape((1, output1.shape[1]))(output1)
attention_output1 = MultiHeadAttention(80,4)(attention_output1)

attention_output2 = Reshape((1, output2.shape[1]))(output2)
attention_output2 = MultiHeadAttention(80,4)(attention_output2)

# Combine the attention outputs
combined_attention_output = Concatenate()([attention_output1, attention_output2])

# Flatten the combined output
flattened_attention_output = Flatten()(combined_attention_output)

# Fully connected layer
x = Dense(160, activation='linear')(flattened_attention_output)
x = Dropout(0.2)(x)
y = Dense(1, activation='linear')(x)

# Create the final model
model = Model(inputs=[input1, input2], outputs=y)

# Print the model summary
model.summary()
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse',metrics=[RootMeanSquaredError(),R2Score()])

#Fitting & Running

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

cp1 = ModelCheckpoint('model1_best.keras', save_best_only=True)
history = model.fit(x=[trainX[:27000],trainX[:27000]], y=trainY[:27000], epochs=100, batch_size=48, validation_split=0.2, verbose=1,callbacks=[cp1])
from tensorflow.keras.models import load_model
Model1 = load_model('model1_best.keras')
#print(Model1.save_spec)
train_predictions = Model1.predict([trainX[27000:],trainX[27000:]])
print(len(train_predictions),len(trainX[27000:]))
plt.plot(train_predictions[:100], label='Predictions')
plt.plot(trainY[27000:27100], label='Actual')
plt.legend()
original_predictions=y_scaler.inverse_transform(train_predictions)
test_y=y_scaler.inverse_transform(trainY[27000:])
r2_score = R2Score()
r2_score.update_state(test_y[:-10], original_predictions[:-10])
print("R2 Score:", r2_score.result().numpy())
print(mean_absolute_error(test_y[:-10],original_predictions[:-10]))
print(mean_squared_error(test_y[:-10],original_predictions[:-10]))
print(mean_absolute_percentage_error(test_y[:-10],original_predictions[:-10]))

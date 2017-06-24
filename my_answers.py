import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string
import re


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    i = 0
    while window_size < len(series) - i:
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
        i += 1

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
	model = Sequential()
	model.add(LSTM(5, input_shape=(window_size,1)))
	model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
	optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
	model.compile(loss='mean_squared_error', optimizer=optimizer)

	return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find the end of the text
    i = text.find('end of the project')
    text = text[:i]

    good_chars = string.ascii_lowercase + ' !,.:;?'
    chars_to_remove = set(text).difference(set(good_chars))

    # shorten any extra dead space created above
    for c in list(chars_to_remove):
        text = text.replace(c,' ')

    text = re.sub(' +', ' ', text)

	return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    i = 0
    while i + window_size < len(text):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
        i += step_size

    
    return inputs,outputs


# coding: utf-8

# In[1]:


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# In[2]:


# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


# In[3]:


# Load in STAT 110 for now; convert to lower case
filename = "cleaned_data/STAT/STAT 110.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()


# In[4]:


# Map unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# Prepare the dataset of input to output pairs encoded as integers
# TODO: try (padded) sentences instead of sequences of length 100?
# see textblob and https://keras.io/preprocessing/sequence/
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# Reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# Normalize
X = X / float(n_vocab)
# One hot encode the output variable
y = np_utils.to_categorical(dataY)


# In[5]:


# Define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define the checkpoint
filepath="keras_checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# In[ ]:


# Fit
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)


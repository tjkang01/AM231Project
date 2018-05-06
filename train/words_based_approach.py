
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

# Load in STAT 110 for now; convert to lower case
filename = "STAT 110.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()


# In[73]:

filtered_text = '\n'.join(filter(lambda x: len(x.split(' ')) > 10, raw_text.split('\n')))


# In[74]:

filtered_text


# In[77]:

words = sorted(list(set(filtered_text.split(' '))))
words_to_int = dict((c, i) for i,c in enumerate(words))


# In[80]:

n_words = len(filtered_text.split(' '))
n_vocab = len(words)


# In[81]:

print("Total words: ", n_words)
print("Total Vocab: ", n_vocab)


# In[89]:

seq_length = 3
dataX = []
dataY = []


# In[90]:

text_use = filtered_text.split(' ')


# In[91]:

for i in range(0, n_words - seq_length, 1):
    seq_in = text_use[i: i +seq_length]
    seq_out = text_use[i + seq_length]
    dataX.append([words_to_int[word] for word in seq_in])
    dataY.append(words_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)


# In[92]:

# Reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# Normalize
X = X / float(n_vocab)
# One hot encode the output variable
y = np_utils.to_categorical(dataY)


# Define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define the checkpoint
filepath="keras_checkpoints_words/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# In[ ]:


# Fit
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)



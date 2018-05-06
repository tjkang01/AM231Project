from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Input, Flatten, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
from six.moves import cPickle

import spacy
nlp = spacy.load('en_core_web_sm')

seq_length = 30 # sequence length
sequences_step = 1 #step to create sequences
in_filename = 'STAT 110.txt'

def create_wordlist(doc):
    wl = []
    for word in doc:
        if word.text not in ("\n","\n\n",'\u2009','\xa0'):
            wl.append(word.text.lower())
    return wl

wordlist = []
with codecs.open(in_filename, "r") as f:
    data = f.read()
#create sentences
doc = nlp(data)
wl = create_wordlist(doc)
wordlist = wordlist + wl

vocab_file = './words_vocab.pkl'

# count the number of words
word_counts = collections.Counter(wordlist)

# Mapping from index to word : that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]
# words = words[0:round(len(words) * scale)]

#size of the vocabulary
vocab_size = len(words)
print("vocab size: ", vocab_size)

#save the words and vocabulary
with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)

#create sequences
sequences = []
next_words = []
for i in range(0, len(wordlist) - seq_length, sequences_step):
    sequences.append(wordlist[i: i + seq_length])
    next_words.append(wordlist[i + seq_length])
#sequences = sequences[0:round(len(sequences) * 1/2)]
print('nb sequences:', len(sequences))

X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, vocab[word]] = 1
    y[i, vocab[next_words[i]]] = 1

def bidirectional_lstm_model(seq_length, vocab_size):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=learning_rate)
    callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    return model

rnn_size = 512 # size of RNN
batch_size = 128 # minibatch size
seq_length = 30 # sequence length
num_epochs = 75 # number of epochs
learning_rate = 0.001 #learning rate
sequences_step = 1 #step to create sequences

md = bidirectional_lstm_model(seq_length, vocab_size)
md.summary()

#fit the model

filepath = "./model2/my_model_gen_sentences_lstm.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=1, mode="auto")
callbacks = [checkpoint]
history = md.fit(X, y,
                 batch_size=batch_size,
                 shuffle=True,
                 epochs=num_epochs,
                 callbacks=callbacks,
                 validation_split=0.01)

md.save('./model2/my_model_gen_sentences_lstm.final.hdf5')


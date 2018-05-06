from six.moves import cPickle
vocab_file = "words_vocab.pkl" 
print("loading vocabulary...")
import numpy as np
with open(vocab_file, 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)
seq_length = 30
vocab_size = len(words)

from keras.models import load_model
# load the model
print("loading model...")
model = load_model('./model2/my_model_gen_sentences_lstm.final.hdf5')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
#initiate sentences
seed_sentences="best class ever"
generated = ''
sentence = []
for i in range (seq_length):
    sentence.append("a")

seed = seed_sentences.split()

for i in range(len(seed)):
    sentence[seq_length-i-1]=seed[len(seed)-i-1]

generated += ' '.join(sentence)
print('Generating text with the following seed: "' + ' '.join(sentence) + '"')

print ()

words_number = 200
#generate the text
for i in range(words_number):
    #create the vector
    x = np.zeros((1, seq_length, vocab_size))
    for t, word in enumerate(sentence):
        x[0, t, vocab[word]] = 1.
    #print(x.shape)

    #calculate next word
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 0.34)
    next_word = vocabulary_inv[next_index]

    #add the next word to the text
    generated += " " + next_word
    # shift the sentence by one, and and the next word at its end
    sentence = sentence[1:] + [next_word]

print(generated)
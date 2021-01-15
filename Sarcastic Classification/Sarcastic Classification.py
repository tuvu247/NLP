#!/usr/bin/env python
# coding: utf-8

# Mục tiêu: Xây dựng mô hình có độ chính xác trên tập validation khoảng 84%


import json
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding,GlobalAveragePooling1D

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000


# !wget --no-check-certificate \
#    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json -O /tmp/sarcasm.json


# Preparing Data

with open("Data.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

import numpy as np
labels = np.array(labels)


# print(len(sentences))
# print(sentences[2])


training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


len(testing_labels)



# Tokenizer

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)

tokenizer.fit_on_texts(training_sentences)

training_sequences = tokenizer.texts_to_sequences(training_sentences)

training_padded = pad_sequences(training_sequences, maxlen=max_length,padding = padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(testing_sequences, maxlen=max_length,padding = padding_type, truncating=trunc_type)


# testing_padded.shape
# training_padded.shape




# Building model


model = Sequential()


model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))

model.add(GlobalAveragePooling1D())

model.add(Dense(24,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.summary()


# Compile Optimizer và Loss function

model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')


# Training

model.fit(training_padded,training_labels,epochs=30,validation_data=(testing_padded,testing_labels))


# Predicting

test_sen = ["spicer denies that ending maternity care guarantee would mean women pay more for health care"]

test_sen_sequences = tokenizer.texts_to_sequences(test_sen)

test_sen_padded = pad_sequences(test_sen_sequences, maxlen=max_length,padding = padding_type, truncating=trunc_type)

y = model.predict(test_sen_padded)

print (y)

if y[0]>=0.5: 
    print('This meaning is 1')
else :
    print('This meaning is 0')




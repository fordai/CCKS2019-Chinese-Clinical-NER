#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import Sequential
from keras.layers import Bidirectional, Dense, Embedding, LSTM, TimeDistributed
from keras_contrib.layers.crf import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy


class BiLSTMCRF:
    def __init__(self, batch_size, epochs_num, embedding_dim, embedding_mat, validation_split,
                 tags, labels, vocab, model_filepath):
        self.batch_size = batch_size
        self.epochs_num = epochs_num
        self.embedding_dim = embedding_dim
        self.embedding_mat = embedding_mat
        self.validation_split = validation_split
        self.tags = tags
        self.labels = labels
        self.vocab = vocab
        self.model_filepath = model_filepath

        self.model = self.__build_model()

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y,
                       batch_size=self.batch_size,
                       epochs=self.epochs_num,
                       validation_split=self.validation_split)
        self.model.save(self.model_filepath)

    def __build_model(self):
        model = Sequential()

        embedding_layer = Embedding(input_dim=len(self.vocab) + 1,
                                    output_dim=self.embedding_dim,
                                    weights=[self.embedding_mat],
                                    trainable=False)
        model.add(embedding_layer)

        bilstm_layer = Bidirectional(LSTM(units=256, return_sequences=True))
        model.add(bilstm_layer)

        model.add(TimeDistributed(Dense(256, activation="relu")))

        crf_layer = CRF(units=len(self.tags), sparse_target=True)
        model.add(crf_layer)

        model.compile(optimizer="adam", loss=crf_loss, metrics=[crf_viterbi_accuracy])
        model.summary()

        return model

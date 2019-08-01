#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import string

import numpy as np
from gensim.models import word2vec, Word2Vec
from keras.preprocessing.sequence import pad_sequences


def build_tag2id(tags, tag2id_filepath):
    """
    Build the dictionary mapping from tag to id
    :param tags: the BIO tags
    :param tag2id_filepath: the output path of the dictionary
    """
    tag2id = {tag: idx for idx, tag in enumerate(tags)}

    with open(tag2id_filepath, "wb") as fw:
        pickle.dump(tag2id, fw)


def load_tag2id(tag2id_filepath):
    """
    Load the dictionary mapping from tag to id
    :param tag2id_filepath: the file path to the pre-built dictionary
    :return: the dictionary mapping from tag to id
    """
    with open(tag2id_filepath, "rb") as fr:
        tag2id = pickle.load(fr)

    return tag2id


def build_vocab(word2vec_model_filepath, vocab_filepath):
    embedding_dict = __get_embedding_dict(word2vec_model_filepath)
    vocab = {key: idx for idx, key in enumerate(sorted(embedding_dict.keys()), 1)}

    with open(vocab_filepath, "wb") as fw:
        pickle.dump(vocab, fw)


def load_vocab(vocab_filepath):
    """
    Load the dictionary mapping from word to id
    :param vocab_filepath: the file path to the pre-built dictionary
    :return: the dictionary mapping from word to id
    """
    with open(vocab_filepath, "rb") as fr:
        word2id = pickle.load(fr)

    return word2id


def train_word2vec_model(data_filepath, model_filepath, embedding_dim):
    seg_txt_list = __get_seg_txt_list(data_filepath)
    model = word2vec.Word2Vec(seg_txt_list, size=embedding_dim, window=5, min_count=1)
    model.save(model_filepath)


def get_embedding_matrix(model_filepath, word2id):
    """
    Get the embedding matrix of the word2vec model
    :param model_filepath: the file path to the pre-build word2vec model
    :param word2id: the directory mapping from word to id
    :return: the embedding matrix of the word2vec model
    """
    word2vec_model = Word2Vec.load(model_filepath)
    embeddings_dict = __get_embedding_dict(model_filepath)
    embedding_matrix = np.zeros((len(word2id) + 1, word2vec_model.vector_size))
    for word, idx in word2id.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    return embedding_matrix


def load_tagged_data(tagged_data_filepath, vocab, tag2id):
    """
    Load the input data to the model
    :param tagged_data_filepath: the file path to the tagged data file
    :param vocab: the dictionary mapping from word to id
    :param tag2id: the dictionary mapping from tag to id
    :return: Numpy arrays: `train_x, train_y`
    """
    seg_samples_list = __get_seg_sample_list(tagged_data_filepath, mode="tagged")

    words_list = [[word2tag[0] for word2tag in sample] for sample in seg_samples_list]
    sample2id = [[vocab.get(word, 0) for word in sample] for sample in words_list]
    max_seq_len = max(len(sample) for sample in sample2id)
    train_x = pad_sequences(sample2id, max_seq_len, padding="post", value=0)

    tags_list = [[word2tag[1] for word2tag in sample] for sample in seg_samples_list]
    tag2id = [[tag2id.get(tag, 0) for tag in sample] for sample in tags_list]
    train_y = pad_sequences(tag2id, max_seq_len, padding="post", value=0)
    train_y = np.expand_dims(train_y, 2)

    return train_x, train_y


def load_untagged_data(untagged_data_filepath, vocab):
    seg_samples_list = __get_seg_sample_list(untagged_data_filepath, mode="untagged")

    words_list = [[word for word in sample] for sample in seg_samples_list]
    sample2id = [[vocab.get(word, 0) for word in sample] for sample in words_list]
    max_seq_len = max(len(sample) for sample in sample2id)
    sample_seq_list = pad_sequences(sample2id, max_seq_len, padding="post", value=0)

    return sample_seq_list


def __get_seg_sample_list(data_filepath, mode="tagged", delimiter="\n"):
    with open(data_filepath, "r", encoding="utf-8-sig") as fr:
        seg_samples_list = [sample.split(delimiter) for sample in fr.read().split(delimiter + delimiter)]
        if mode is "tagged":
            seg_samples_list = [[word2tag.split(" ") for word2tag in sample] for sample in seg_samples_list]
        elif mode is "untagged":
            seg_samples_list = [[word for word in sample] for sample in seg_samples_list]

        seg_samples_list.pop()

    return seg_samples_list


def __get_seg_txt_list(data_filepath, delimiter="\n"):
    seg_sample_list = []

    with open(data_filepath, "r", encoding="utf-8-sig") as fr:
        for sample in fr.read().split(delimiter + delimiter):
            words = []
            for word in sample.split(delimiter):
                if word not in string.whitespace:
                    words.append(word.split()[0])
            seg_sample_list.append(" ".join(words))

    return seg_sample_list


def __get_embedding_dict(model_filepath):
    embedding_dict = {}
    word2vec_model = Word2Vec.load(model_filepath)
    vocab = [(word, word2vec_model.wv[word]) for word, vectors in word2vec_model.wv.vocab.items()]

    for i in range(len(vocab)):
        word = vocab[i][0]
        vectors = vocab[i][1]
        embedding_dict[word] = vectors

    return embedding_dict

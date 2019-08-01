#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

from keras.models import load_model
from keras_contrib.layers.crf import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

import sys

sys.path.append('../')

from word2vec_bilstm_crf.model import BiLSTMCRF
from word2vec_bilstm_crf.data_utils import build_tag2id, load_tag2id, train_word2vec_model, get_embedding_matrix, \
    build_vocab, load_vocab, load_tagged_data, load_untagged_data
from word2vec_bilstm_crf.eval_utils import get_true_tags, get_pred_tags, get_ori_txt, get_ori_txt_without_none, \
    save_results_as_json, evaluate

# The label types
LABELS_LIST = ["疾病和诊断",
               "解剖部位",
               "实验室检验",
               "影像检查",
               "手术",
               "药物"]

# The BIO tags
TAGS_LIST = ["O",
             "B-疾病和诊断", "I-疾病和诊断",
             "B-解剖部位", "I-解剖部位",
             "B-实验室检验", "I-实验室检验",
             "B-影像检查", "I-影像检查",
             "B-手术", "I-手术",
             "B-药物", "I-药物"]

# The JSON keys used in the original data files
JSON_ORI_TXT_KEY = "originalText"
JSON_ENTITIES_KEY = "entities"
JSON_START_POS_KEY = "start_pos"
JSON_END_POS_KEY = "end_pos"
JSON_LABEL_KEY = "label_type"
JSON_OVERLAP_KEY = "overlap"

# The file paths to the model configuration
MODEL_DIR = "./model/"
TAG2ID_FILENAME = "tag2id.pkl"
TAG2ID_FILEPATH = os.path.join(MODEL_DIR, TAG2ID_FILENAME)
VOCAB_FILENAME = "vocab.pkl"
VOCAB_FILEPATH = os.path.join(MODEL_DIR, VOCAB_FILENAME)
WORD2VEC_MODEL_FILENAME = "word2vec.model"
WORD2VEC_MODEL_FILEPATH = os.path.join(MODEL_DIR, WORD2VEC_MODEL_FILENAME)
MODEL_FILENAME = "word2vec_bilstm_crf.h5"
MODEL_FILEPATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# The file paths to the tagged data
PROC_DATA_DIR = "../data/processed_data/"
TRAIN_DATA_FILENAME = "train_data.txt"
TRAIN_DATA_FILEPATH = os.path.join(PROC_DATA_DIR, TRAIN_DATA_FILENAME)
TEST_DATA_FILENAME = "test_data.txt"
TEST_DATA_FILEPATH = os.path.join(PROC_DATA_DIR, TEST_DATA_FILENAME)
UNTAGGED_TEST_DATA_FILENAME = "untagged_test_data.txt"
UNTAGGED_TEST_DATA_FILEPATH = os.path.join(PROC_DATA_DIR, UNTAGGED_TEST_DATA_FILENAME)

# The file paths to prediction results
RESULTS_DIR = "./results/"
TEST_RESULTS_FILENAME = "test_results.json"
TEST_RESULTS_FILEPATH = os.path.join(RESULTS_DIR, TEST_RESULTS_FILENAME)
TRUE_RESULTS_FILENAME = "true_results.json"
TRUE_RESULTS_FILEPATH = os.path.join(RESULTS_DIR, TRUE_RESULTS_FILENAME)
EVAL_RESULTS_FILENAME = "eval_results.txt"
EVAL_RESULTS_FILEPATH = os.path.join(RESULTS_DIR, EVAL_RESULTS_FILENAME)
PRED_RESULTS_FILENAME = "pred_results.json"
PRED_RESULTS_FILEPATH = os.path.join(RESULTS_DIR, PRED_RESULTS_FILENAME)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test',
                        help="'Train' or 'Test' or 'Predict': 'Train' and 'Test' are based on tagged data,"
                             " 'predict' uses untagged data")
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size')
    parser.add_argument('--epoch_num', type=int, default=200,
                        help='The number of epoch to train the model')
    parser.add_argument('--embedding_dim', type=int, default=200,
                        help='The dimension of the embedding layer')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of the training data to be used as validation data')
    args = parser.parse_args()

    if args.mode == 'train':
        # Construct the tag2id dictionary
        build_tag2id(tags=TAGS_LIST, tag2id_filepath=TAG2ID_FILEPATH)
        tag2id = load_tag2id(tag2id_filepath=TAG2ID_FILEPATH)

        # Train the word2vec model
        train_word2vec_model(TRAIN_DATA_FILEPATH, WORD2VEC_MODEL_FILEPATH, args.embedding_dim)

        # Construct the word2id dictionary
        build_vocab(WORD2VEC_MODEL_FILEPATH, VOCAB_FILEPATH)
        vocab = load_vocab(vocab_filepath=VOCAB_FILEPATH)

        # Get the embedding matrix
        embedding_mat = get_embedding_matrix(WORD2VEC_MODEL_FILEPATH, vocab)

        # Construct the model
        model = BiLSTMCRF(
            batch_size=args.batch_size,
            epochs_num=args.epoch_num,
            embedding_dim=args.embedding_dim,
            embedding_mat=embedding_mat,
            validation_split=args.val_split,
            tags=TAGS_LIST,
            labels=LABELS_LIST,
            vocab=vocab,
            model_filepath=MODEL_FILEPATH,
        )

        # Train the model
        train_x, train_y = load_tagged_data(TRAIN_DATA_FILEPATH, vocab, tag2id)
        model.train(train_x, train_y)

    elif args.mode == 'test':
        tag2id = load_tag2id(tag2id_filepath=TAG2ID_FILEPATH)
        vocab = load_vocab(vocab_filepath=VOCAB_FILEPATH)

        # Load the model
        custom_objects = {"CRF": CRF,
                          "crf_loss": crf_loss,
                          "crf_viterbi_accuracy": crf_viterbi_accuracy}
        model = load_model(MODEL_FILEPATH, custom_objects=custom_objects)

        # Test the model
        test_x, test_y = load_tagged_data(TEST_DATA_FILEPATH, vocab, tag2id)
        pred_y = model.predict(test_x)

        # Process the prediction results
        pred_y = get_pred_tags(pred_y, tag2id)
        true_y = get_true_tags(test_y, tag2id)

        # Process the test texts
        ori_x = get_ori_txt(test_x, vocab)
        ori_x_without_none = get_ori_txt_without_none(test_x, vocab)

        # Save the results
        save_results_as_json(ori_x, ori_x_without_none, pred_y, TEST_RESULTS_FILEPATH)
        save_results_as_json(ori_x, ori_x_without_none, true_y, TRUE_RESULTS_FILEPATH)

        # Evaluate the results
        evaluate(TEST_RESULTS_FILEPATH, TRUE_RESULTS_FILEPATH, EVAL_RESULTS_FILEPATH, LABELS_LIST, TAGS_LIST)

    elif args.mode == 'predict':
        tag2id = load_tag2id(tag2id_filepath=TAG2ID_FILEPATH)
        vocab = load_vocab(vocab_filepath=VOCAB_FILEPATH)

        # Load the model
        custom_objects = {"CRF": CRF,
                          "crf_loss": crf_loss,
                          "crf_viterbi_accuracy": crf_viterbi_accuracy}
        model = load_model(MODEL_FILEPATH, custom_objects=custom_objects)

        # Test the model
        test_x = load_untagged_data(UNTAGGED_TEST_DATA_FILEPATH, vocab)
        pred_y = model.predict(test_x)

        # Process the prediction results
        pred_y = get_pred_tags(pred_y, tag2id)

        # Process the test texts
        ori_x = get_ori_txt(test_x, vocab)
        ori_x_without_none = get_ori_txt_without_none(test_x, vocab)

        # Save the results as json file
        save_results_as_json(ori_x, ori_x_without_none, pred_y, PRED_RESULTS_FILEPATH)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import argparse
import json
import os

import jieba.posseg as pseg
import numpy as np

# The JSON keys used in the original data files
JSON_ORI_TXT_KEY = "originalText"
JSON_ENTITIES_KEY = "entities"
JSON_START_POS_KEY = "start_pos"
JSON_END_POS_KEY = "end_pos"
JSON_LABEL_KEY = "label_type"
JSON_OVERLAP_KEY = "overlap"

ORI_DATA_DIR = "../data/original_data/"
TAGGED_DATA_DIR = "tagged_data/"
ORI_TAGGED_DATA_FILEPATH = os.path.join(ORI_DATA_DIR, TAGGED_DATA_DIR)
UNTAGGED_DATA_DIR = "untagged_data/"
ORI_UNTAGGED_DATA_FILEPATH = os.path.join(ORI_DATA_DIR, UNTAGGED_DATA_DIR)

PROC_DATA_DIR = "../data/processed_data/"
TRAIN_DATA_FILENAME = "train_data.txt"
TRAIN_DATA_FILEPATH = os.path.join(PROC_DATA_DIR, TRAIN_DATA_FILENAME)
TEST_DATA_FILENAME = "test_data.txt"
TEST_DATA_FILEPATH = os.path.join(PROC_DATA_DIR, TEST_DATA_FILENAME)
UNTAGGED_TEST_DATA_FILENAME = "untagged_test_data.txt"
UNTAGGED_TEST_DATA_FILEPATH = os.path.join(PROC_DATA_DIR, UNTAGGED_TEST_DATA_FILENAME)


def preprocess_tagged_data(ori_data_dir, train_data_filepath, test_data_filepath="", test_split=0):
    train_total_num = 0
    test_total_num = 0

    for data_filename in os.listdir(ori_data_dir):
        data_filepath = os.path.join(ori_data_dir, data_filename)
        samples_list = np.loadtxt(data_filepath,
                                  dtype="str", comments=None, delimiter="\r\n", encoding="utf-8-sig")
        test_sample_num = int(len(samples_list) * test_split)
        train_sample_num = int(len(samples_list) - test_sample_num)
        train_total_num += train_sample_num
        test_total_num += test_sample_num

        if len(test_data_filepath) > 0 and test_split > 0:
            __preprocess_tagged_data(samples_list[0:train_sample_num], train_data_filepath)
            __preprocess_tagged_data(samples_list[train_sample_num:], test_data_filepath)
        else:
            __preprocess_tagged_data(samples_list, train_data_filepath)

    print("Training samples: {}, Testing samples: {}".format(train_total_num, test_total_num))


def preprocess_untagged_data(ori_data_dir, test_data_filepath):
    train_total_num = 0
    for data_filename in os.listdir(ori_data_dir):
        data_filepath = os.path.join(ori_data_dir, data_filename)
        samples_list = np.loadtxt(data_filepath,
                                  dtype="str", comments=None, delimiter="\r\n", encoding="utf-8-sig")
        train_total_num += len(samples_list)
        __preprocess_untagged_data(samples_list, test_data_filepath)

    print("Training samples: {}".format(train_total_num))


def __preprocess_tagged_data(samples_list, tagged_data_filepath, delimiter="\n"):
    with open(tagged_data_filepath, "a", encoding="utf-8-sig") as fw:
        for i in range(len(samples_list)):
            word2tag = []
            sample = json.loads(samples_list[i])

            original_text = sample[JSON_ORI_TXT_KEY]
            for sentence in original_text.split(chr(12290)):
                if len(sentence) < 1:
                    continue
                sentence = sentence + chr(12290)
                for words in pseg.cut(sentence):
                    for w in words.word:
                        word2tag.append([w, "O"])

            entities = sample[JSON_ENTITIES_KEY]
            for entity in entities:
                if len(entity) < 1:
                    continue
                start_pos = entity[JSON_START_POS_KEY]
                end_pos = entity[JSON_END_POS_KEY]
                label_type = entity[JSON_LABEL_KEY]
                word2tag[start_pos][1] = "B-" + label_type
                for j in range(start_pos + 1, end_pos):
                    word2tag[j][1] = "I-" + label_type

            for [word, tag] in word2tag:
                fw.write(word + " " + tag)
                fw.write(delimiter)

            fw.write(delimiter)


def __preprocess_untagged_data(samples_list, untagged_data_filepath, delimiter="\n"):
    with open(untagged_data_filepath, "a", encoding="utf-8-sig") as fw:
        for i in range(len(samples_list)):
            sample = json.loads(samples_list[i])
            original_text = sample[JSON_ORI_TXT_KEY]
            for sentence in original_text.split(chr(12290)):
                if len(sentence) < 1:
                    continue
                sentence = sentence + chr(12290)
                for words in pseg.cut(sentence):
                    for w in words.word:
                        fw.write(w)
                        fw.write(delimiter)

            fw.write(delimiter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tagged', type=ast.literal_eval, default='True',
                        help='Whether the data has been tagged')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction of tagged data used as testing data')
    args = parser.parse_args()

    if args.tagged:
        preprocess_tagged_data(ori_data_dir=ORI_TAGGED_DATA_FILEPATH,
                               train_data_filepath=TRAIN_DATA_FILEPATH,
                               test_data_filepath=TEST_DATA_FILEPATH,
                               test_split=args.test_split)
    else:
        preprocess_untagged_data(ori_data_dir=ORI_UNTAGGED_DATA_FILEPATH,
                                 test_data_filepath=UNTAGGED_TEST_DATA_FILEPATH)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from collections import OrderedDict

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 10)
pd.set_option('display.width', 1000)
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)

JSON_ORI_TXT_KEY = "originalText"
JSON_ENTITIES_KEY = "entities"
JSON_START_POS_KEY = "start_pos"
JSON_END_POS_KEY = "end_pos"
JSON_LABEL_KEY = "label_type"
JSON_OVERLAP_KEY = "overlap"
JSON_MICRO_AVERAGE_KEY = "微平均"


def get_pred_tags(data_y, tag2id):
    pred_tags_list = []
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    for i in range(data_y.shape[0]):
        sample_tags = [id2tag[idx] for idx in np.argmax(data_y[i], axis=1)]
        pred_tags_list.append(sample_tags)

    return pred_tags_list


def get_true_tags(data_y, tag2id):
    true_tags_list = []
    id2tag = {idx: tag for tag, idx in tag2id.items()}

    for i in range(data_y.shape[0]):
        sample_tags = [id2tag[j] for idx in data_y[i] for j in idx]
        true_tags_list.append(sample_tags)

    return true_tags_list


def get_ori_txt(data_x, vocab):
    ori_txt_list = []
    id2word = {idx: word for word, idx in vocab.items()}

    for i in range(data_x.shape[0]):
        words_list = [id2word[idx] if idx > 0 else "None" for idx in data_x[i]]
        ori_txt_list.append(words_list)

    return ori_txt_list


def get_ori_txt_without_none(data_x, vocab):
    ori_txt_list = []
    id2word = {idx: word for word, idx in vocab.items()}

    for i in range(data_x.shape[0]):
        words_list = [id2word[idx] for idx in data_x[i] if idx > 0]
        ori_txt_list.append(words_list)

    return ori_txt_list


def save_results_as_json(data_x, data_x_without_none, data_y, results_filepath, delimiter="\n"):
    with open(results_filepath, "w", encoding="utf-8") as fw:
        for i in range(len(data_x)):
            idx = 0
            line = {}
            entity = {}
            entities = []
            line[JSON_ORI_TXT_KEY] = "".join(data_x_without_none[i])

            for tag in data_y[i]:
                tag = str(tag)
                if tag.startswith("B"):
                    entity[JSON_LABEL_KEY] = tag[2:]
                    entity[JSON_START_POS_KEY] = idx
                elif tag.startswith("O") and entity:
                    entity[JSON_END_POS_KEY] = idx
                    entities.append(entity)
                    entity = {}
                idx += 1
            entities = __check_overlap(entities)
            line[JSON_ENTITIES_KEY] = entities

            fw.write(json.dumps(line, ensure_ascii=False))
            fw.write(delimiter)


def evaluate(pred_results_filepath, true_results_filepath, eval_results_filepath, labels, tags):
    """
    Evaluate the predictions based on precision, recall and F1 scores
    :param pred_results_filepath: the file path to the predictions
    :param true_results_filepath: the file path to the standard results
    :param eval_results_filepath: the file path of the evaluation results
    :param labels: the label types
    :param tags: the BIO tags
    """
    pred_total_dict = __calc_total(pred_results_filepath, labels, tags)
    print("Predictions", pred_total_dict)

    true_total_dict = __calc_total(true_results_filepath, labels, tags)
    print("Standards", true_total_dict)

    pred_corr_dict = dict(zip(labels, [0 for _ in range(len(tags))]))
    with open(true_results_filepath, "r", encoding="utf-8-sig") as true_fr, \
            open(pred_results_filepath, "r", encoding="utf-8-sig") as pred_fr:
        for true_line, pred_line in zip(true_fr.readlines(), pred_fr.readlines()):
            for pred_entity in json.loads(pred_line)[JSON_ENTITIES_KEY]:
                for true_entity in json.loads(true_line)[JSON_ENTITIES_KEY]:
                    if __is_strict_equal(pred_entity, true_entity):
                        label_type = pred_entity[JSON_LABEL_KEY]
                        pred_corr_dict[label_type] += 1
                        break
    print("Correct predictions", pred_corr_dict)

    precision_dict = __calc_precision(pred_corr_dict, pred_total_dict, labels)
    recall_dict = __calc_recall(pred_corr_dict, true_total_dict, labels)
    f1_score_dict = __calc_f1_score(precision_dict, recall_dict, labels)

    __save_evaluation_results(precision_dict, recall_dict, f1_score_dict, eval_results_filepath)


def __is_strict_equal(json_entity_1, json_entity_2):
    is_equal = True

    for i in [JSON_START_POS_KEY, JSON_END_POS_KEY, JSON_LABEL_KEY]:
        if json_entity_1[i] != json_entity_2[i]:
            is_equal = False

    return is_equal


def __calc_total(results_filepath, labels, tags):
    total_dict = dict(zip(labels, [0 for _ in range(len(tags))]))

    with open(results_filepath, "r", encoding="utf-8-sig") as fr:
        for line in fr.readlines():
            for entity in json.loads(line)[JSON_ENTITIES_KEY]:
                label_type = entity[JSON_LABEL_KEY]
                total_dict[label_type] += 1

    return total_dict


def __calc_precision(pred_corr_dict, pred_total_dict, labels):
    precision_dict = OrderedDict()
    pred_corr_total = 0
    pred_total = 0

    for label in labels:
        precision_dict[label] = pred_corr_dict[label] * 1.0 / pred_total_dict[label] if pred_total_dict[label] else 0
        pred_corr_total += pred_corr_dict[label]
        pred_total += pred_total_dict[label]
    precision_dict[JSON_MICRO_AVERAGE_KEY] = pred_corr_total * 1.0 / pred_total if pred_total else 0

    return precision_dict


def __calc_recall(pred_corr_dict, true_total_dict, labels):
    recall_dict = OrderedDict()
    pred_corr_total = 0
    standard_total = 0

    for label in labels:
        recall_dict[label] = pred_corr_dict[label] * 1.0 / true_total_dict[label] if true_total_dict[label] else 0
        pred_corr_total += pred_corr_dict[label]
        standard_total += true_total_dict[label]
    recall_dict[JSON_MICRO_AVERAGE_KEY] = pred_corr_total * 1.0 / standard_total if standard_total else 0

    return recall_dict


def __calc_f1_score(precision_dict, recall_dict, labels):
    f1_score_dict = OrderedDict()

    for label in labels:
        p_r_sum = precision_dict[label] + recall_dict[label]
        f1_score_dict[label] = 2 * precision_dict[label] * recall_dict[label] / p_r_sum if p_r_sum else 0
    p_r_sum = precision_dict[JSON_MICRO_AVERAGE_KEY] + recall_dict[JSON_MICRO_AVERAGE_KEY]
    f1_score_dict[JSON_MICRO_AVERAGE_KEY] = 2 * precision_dict[JSON_MICRO_AVERAGE_KEY] * recall_dict[
        JSON_MICRO_AVERAGE_KEY] / p_r_sum if p_r_sum else 0

    return f1_score_dict


def __save_evaluation_results(precision, recall, f1_score, eval_results_filepath):
    df = pd.DataFrame([precision, recall, f1_score], index=["Precision", "Recall", "F1-Score"])
    df.to_csv(eval_results_filepath)
    print(df)


def __check_overlap(entities):
    for i in range(len(entities) - 1):
        curr_entity = entities[i]
        next_entity = entities[i + 1]
        if curr_entity[JSON_END_POS_KEY] > next_entity[JSON_START_POS_KEY]:
            curr_entity[JSON_OVERLAP_KEY] = 1
        else:
            curr_entity[JSON_OVERLAP_KEY] = 0

    return entities

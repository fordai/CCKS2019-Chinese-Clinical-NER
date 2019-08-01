# CCKS2019 Chinese Clinical NER
The word2vec BiLSTM-CRF model for CCKS2019 Chinese clinical named entity recognition.

# Dependencies 
* python 3.6
* gensim 3.4.0
* jieba 0.39
* keras 2.2.4
* keras_contrib 2.0.8
* numpy 1.16.4
* pandas 0.24.2

# Dataset
The dataset is provided by the CCKS2019.

文本 | 疾病和诊断 | 影像检查 | 实验室检验 | 手术 | 药物 | 解剖部位 | 总数
------------ | ------------- | ------------ | -------------| ------------ | ------------- | ------------ | -------------
1000 | 2116 | 222 | 318 | 765 | 456 | 1486 | 5363

## Data directory structure
* ./data/
  * original_data/
    * tagged_data/
    * untagged_data/
  * processed_data/ 
    * test_data.txt
    * train_data.txt
    * untagged_test_data.txt
## Data format
* In the "original_data" directory:
  * Each data file in the "tagged_data" should be in the following format:
      * Each line is a JSON object, with "originalText" and "entities" as JSON keys;
      * The JSON value of "entities" is a list of JSON object, and each JSON object represents an entity with "entity_name", "start_pos", "end_pos", "label_type", "overlap" as its JSON keys;
  * Each data file in the "untagged_data" should be in the following format:
      * Each line is a JSON object, with "originalText" as the JSON key;
* In the "processed_data" directory: "train_data.txt" and "test_data.txt" should be in the following format:
```
患 O
者 O
罹 O
患 O
胃 B-疾病和诊断
癌 I-疾病和诊断

每 O
个 O
例 O
子 O
空 O
行 O
分 O
隔 O
```
* "untagged_test_data.txt" should be in the following format:
```
患
者
罹
患
胃
癌

每
个
例
子
空
行
分
隔
```

# Getting Started
## Data configuration
* Please download the dataset from CCKS2019 by yourself.
* Put the tagged data under the directory "/data/original_data/tagged_data/".
* Put the untagged data under the directory "/data/original_data/untagged_data/".

## Preprocess
```
python preprocess --tagged True
python preprocess --tagged False
```

## Train the model
```
python main.py --mode train
```

## Test the model
```
python main.py --mode test
```
The prediction, standard results and evaluation would be saved as "test_results.json", "true_results.json" and "eval_results.csv", respectively.

## Predict
```
python main.py --mode predict
```
The prediction would be saved as "pred_results.json".

# Performance
| | 疾病和诊断 | 影像检查 | 实验室检验 | 手术 | 药物 | 解剖部位 | 综合
------------- | ------------- | ------------ | -------------| ------------ | ------------- | ------------ | -------------
严格指标 |  0.49346 | 0.51851 | 0.41049 | 0.55263 | 0.46835 | 0.49975 | 0.49018
松弛指标 |  0.58800 | 0.58370 | 0.54920 | 0.67105 | 0.55485 | 0.55902 | 0.56851

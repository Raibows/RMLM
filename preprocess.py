import nltk
import os
import re
import csv
import random
from tqdm import tqdm
from collections import Counter
from tools import tools_json_dump, tools_get_logger, tools_json_load


def write_data_json(datas, labels, label_map, path):
    assert len(datas) == len(labels)
    tools_get_logger('preprocess').info(f"write data {len(datas)} {label_map} to {path}")
    obj = {
        'num': len(datas),
        'label_map': label_map,
        'data':  datas,
        'label': labels,
    }
    tools_json_dump(obj, path)

def clean_text(text:str, lower=True):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = ' '.join(nltk.word_tokenize(text)).strip()
    return text.lower() if lower else text

def preprocess_imdb(origin_dir, output_dir='./dataset'):
    label_map = {
        'negative': 0,
        'positive': 1
    }
    for s in ['train', 'test']:
        path = f"{origin_dir}/{s}"
        path_list = []
        dirs = os.listdir(path)
        for dir in dirs:
            if dir == 'pos' or dir == 'neg':
                file_list = os.listdir(os.path.join(path, dir))
                file_list = map(lambda x: os.path.join(path, dir, x), file_list)
                path_list += list(file_list)
        datas = []
        labels = []
        for p in tqdm(path_list, desc=f'imdb_{s}'):
            label = 0 if 'neg' in p else 1
            with open(p, 'r', encoding='utf-8') as file:
                datas.append(clean_text(file.readline().strip()))
                labels.append(label)

        write_data_json(datas, labels, label_map, f'{output_dir}/imdb.{s}.json')

def preprocess_sst(path, n_classes, output_dir='./dataset'):
    assert n_classes in {2, 5}
    judge_sentiment = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    def split_sst():
        with open(f"{path}/datasetSentences.txt", 'r') as sentences, open(f"{path}/datasetSplit.txt", 'r') as splits:
            train, valid, test = [], [], []
            temp = {'1': train, '2': test, '3': valid}
            sentences = sentences.readlines()
            splits = splits.readlines()
            for i in range(1, len(sentences)):
                t1 = sentences[i].replace("-LRB-", "(")
                t2 = t1.replace("-RRB-", ")")
                k = splits[i].strip().split(",")
                t = t2.strip().split('\t')
                temp[k[1]].append(t[1])
        return train, valid, test
    def assign_labels(data):
        processed_datas = []
        processed_labels = []
        with open(f"{path}/sentiment_labels.txt", 'r') as labels, open(f"{path}/dictionary.txt", 'r') as tables:
            labels = labels.readlines()
            tables = tables.readlines()
        text2id = {}
        for i in range(len(tables)):
            s = tables[i].strip().split("|")
            text2id[s[0]] = s[1]
        id2sentiment = {}
        for i in range(len(labels)):
            s = labels[i].strip().split("|")
            id2sentiment[s[0]] = s[1]
        dropped = 0
        for line in tqdm(data, desc=f'sst{n_classes}'):
            line = line.strip()
            if line not in text2id:
                dropped += 1
                continue
            score = float(id2sentiment[text2id[line]])
            for i in range(1, len(judge_sentiment)):
                if score >= judge_sentiment[i-1] and score <= judge_sentiment[i]:
                    processed_labels.append(i-1)
                    break
            processed_datas.append(clean_text(line))
            assert len(processed_datas) == len(processed_labels)
            assert processed_labels[-1] in {0, 1, 2, 3, 4}
        if n_classes == 2:
            delete = []
            for i in range(len(processed_labels)):
                if processed_labels[i] < 2: processed_labels[i] = 0
                elif processed_labels[i] > 2: processed_labels[i] = 1
                else:
                    delete.append(i)
            dropped += len(delete)
            for i in sorted(delete, reverse=True):
                del processed_datas[i]
                del processed_labels[i]

            label_map = {'low': 0, 'high': 1}
        else:
            label_map = {'very low': 0, 'low': 1, 'neutral': 2, 'high': 3, 'very high': 4}
        return processed_datas, processed_labels, label_map, dropped

    train, valid, test = split_sst()
    temp = {'train': train, 'valid': valid, 'test': test}
    for k, v in temp.items():
        processed_datas, processed_labels, label_map, dropped = assign_labels(v)
        tools_get_logger('preprocess').info(f'SST-{n_classes} {k} dropped data {dropped}')
        write_data_json(processed_datas, processed_labels, label_map, f'{output_dir}/sst{n_classes}.{k}.json')

def preprocess_agnews(path, output_dir='./dataset'):
    types = ['train', 'test']
    label_map = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci / Tech': 3}
    for t in types:
        datas = []
        labels = []
        with open(f"{path}/{t}.csv", 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            for line in tqdm(reader, desc=f'agnews_{t}'):
                labels.append(int(line[0]) - 1)
                line = line[1] + '. ' + line[2]
                datas.append(clean_text(line))
        write_data_json(datas, labels, label_map, f'{output_dir}/agnews.{t}.json')

def split_data(path, num, output, seed=0):
    source_data = tools_json_load(path)
    data = source_data['data']
    label = source_data['label']
    label_map = source_data['label_map']
    data_length = source_data['num']
    assert num <= data_length

    class_num = len(label_map.keys())
    keys = list(label_map.values())
    num4class = Counter(label)
    label4class_list = {}
    selected_num4class = {}
    for element in keys:
        label4class_list[element] = [i for i, x in enumerate(label) if x == element]

    count = 0
    for i in range(class_num):
        if i == class_num - 1:
            selected_num4class[keys[i]] = num - count
        else:
            each_num = int(num * num4class[keys[i]] / data_length)
            selected_num4class[keys[i]] = each_num
            count += each_num

    selected_data = []
    selected_label = []
    for i in range(class_num):
        random.seed(seed + i)
        sample_index = random.sample(label4class_list[keys[i]], selected_num4class[keys[i]])
        selected_label = selected_label + [label[i] for i in sample_index]
        selected_data = selected_data + [data[i] for i in sample_index]
    write_data_json(selected_data, selected_label, label_map, output)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    choices = ['imdb', 'sst2', 'sst5', 'agnews']
    parser.add_argument('--dataset', type=str, choices=choices + ['all'], nargs='+', required=True)
    parser.add_argument('--attack_set', type=int, default=1000)
    args = parser.parse_args()
    if 'all' in args.dataset:
        args.dataset = choices
    for d in args.dataset:
        if d == 'imdb':
            preprocess_imdb('./dataset/imdb', output_dir='./dataset')
        elif d == 'agnews':
            preprocess_agnews('./dataset/ag_news_csv', output_dir='./dataset')
        elif d == 'sst2':
            preprocess_sst('./dataset/stanfordSentimentTreebank', 2, output_dir='./dataset')
        elif d == 'sst5':
            preprocess_sst('./dataset/stanfordSentimentTreebank', 5, output_dir='./dataset')
        else:
            raise NotImplementedError(f"{d} not found")

        path = f'./dataset/{d}.test.json'
        attack_path = f'./dataset/{d}.attack.json'
        if os.path.exists(path):
            split_data(path, num=args.attack_set, output=attack_path)
        else:
            path = f'./dataset/{d}.valid.json'
            split_data(path, num=args.attack_set, output=attack_path)
    pass
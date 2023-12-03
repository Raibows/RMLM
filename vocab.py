import copy
import os
import random
import re
from config import config_dataset
from tools import tools_json_dump, tools_get_logger


class Vocab():
    def __init__(self, datas, config:config_dataset, model_config:dict):
        self.datas = datas
        self.config = config
        self.model_config = model_config
        self.word2idx = {}
        self.__build_vocab()
        if self.model_config:
            self.__prepare_pretrained_wv()
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        tools_get_logger('vocab').info(f'limit {config.vocab_size} real_size {len(self)} low_freq {config.vocab_low_freq} high_freq {config.vocab_high_freq}')

        assert '[PAD]' in self.word2idx and '[UNK]' in self.word2idx

    def tokenize(self, sen:str):
        idxs = []
        for i, word in enumerate(self.__word_split(sen)[:self.config.maxlen]):
            if word not in self.word2idx: word = '[UNK]'
            idxs.append(self.word2idx[word])
        return idxs, len(idxs)

    def __prepare_pretrained_wv(self):
        if self.model_config['pretrained_wv_path']:
            path = f"{self.model_config['pretrained_wv_path']}.{self.config.name}"
            if not os.path.exists(path):
                tools_get_logger('vocab').info(f'starting prepare glove {path}')
                random.seed(0)
                wv_dim = int(re.findall("\d+d", path)[0][:-1])
                wv = [[random.normalvariate(0.0, 0.2) for _ in range(wv_dim)] for _ in range(len(self.word2idx))]
                with open(self.model_config['pretrained_wv_path'], 'r') as file:
                    for line in file:
                        line = line.strip().split()
                        word = ' '.join(line[:-wv_dim])
                        if word in self.word2idx:
                            wv[self.word2idx[word]] = [float(t) for t in line[-wv_dim:]]
                tools_json_dump(wv, path)

    def __word_split(self, sen:str):
        return sen.strip().lower().split()

    def __build_vocab(self):
        self.word2idx = copy.deepcopy(self.config.special_map)
        counts = {}
        for line in self.datas:
            for word in self.__word_split(line):
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1
        counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
        counts = list(counts.keys())
        start = int(self.config.vocab_high_freq * len(counts))
        end = len(counts) - int(self.config.vocab_low_freq * len(counts))
        end = max(start+1, end)
        for i in range(start, end):
            if len(self.word2idx) == self.config.vocab_size: break
            self.word2idx[counts[i]] = len(self.word2idx)

    def __len__(self):
        return len(set(self.word2idx.values()))

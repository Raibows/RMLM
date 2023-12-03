from torch.utils.data import Dataset
from tools import tools_json_load, tools_get_logger
from transformers import AutoTokenizer
from vocab import Vocab
from config import config_dataset

class ClassificationDataset(Dataset):
    def __init__(self, path, config:config_dataset, model_name, model_config, vocab=None):
        self.logger = tools_get_logger('data')
        self.logger.info(f'loading data from {path}')
        temp = tools_json_load(path)
        self.data, self.label, self.label_map = temp['data'], temp['label'], temp['label_map']
        self.config = config

        bert_tokenizer_name = config.get_bert_config()['tokenizer_name'] if self.config.using_bert_vocab else None
        self.using_bert = 'bert' in model_name.lower()
        if self.config.using_bert_vocab:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_name)
            self.pad_token = (self.tokenizer.pad_token, self.tokenizer.pad_token_id)
            self.vocab = None
        else:
            if not vocab:
                self.vocab = Vocab(self.data, config, model_config)
            else:
                self.vocab = vocab
            self.pad_token = ('[PAD]', self.vocab.word2idx['[PAD]'])
            self.tokenizer = self.vocab.tokenize

        if 'vocab_size' in model_config:
            vocab_size = len(self.vocab) if self.vocab else len(self.tokenizer)
            self.logger.info(f"reset max vocab size {model_config['vocab_size']} to {vocab_size}")
            model_config['vocab_size'] = vocab_size

    def pack_samples(self):
        self.packed = []
        if self.config.using_bert_vocab:
            temp = self.tokenizer(self.data, return_length=True, max_length=self.config.maxlen, truncation=True, add_special_tokens=True)
            self.input_ids = temp['input_ids']
            # type split sentence pair [000011111]
            self.type_ids = temp['token_type_ids']
            # mask the padding token [11111000]
            self.mask_ids = temp['attention_mask']
            self.length = temp['length']
            self.packed = [[i, self.input_ids[i], self.type_ids[i], self.mask_ids[i], self.length[i], self.label[i]] for i in range(len(self.label))]
        else:
            self.input_ids = []
            self.length = []
            for line in self.data:
                idx, l = self.tokenizer(line)
                self.input_ids.append(idx)
                self.length.append(l)
            self.packed = [[i, self.input_ids[i], i, i, self.length[i], self.label[i]] for i in range(len(self.label))]
            

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.packed[item]


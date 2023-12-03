
class config_dataset():
    def __init__(self, name, train_path, valid_path, test_path, attack_path, load_path, using_bert_vocab):
        self.name = name
        self.n_classes = 0
        self.maxlen = 0
        self.vocab_size = 50000 if not using_bert_vocab else 30522
        self.vocab_low_freq = 0.1
        self.vocab_high_freq = 0.0
        self.using_bert_vocab = using_bert_vocab
        self.special_map = {'[PAD]': 0, '[UNK]': 1}
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.attack_path = attack_path
        self.syn_w2list_path = f"dataset/bert.synonyms.json"
        self.load_path = load_path
        self.pretrained_wv_path = 'glove/glove.840B.300d.txt' if not using_bert_vocab else None
        self.rmlm_config = None

    def get_lstm_config(self):
        return {}

    def get_bilstm_config(self):
        return {}

    def get_wordcnn_config(self):
        return {}

    def get_bert_config(self):
        return {}

    def get_load_path(self, model_name):
        return self.load_path[model_name]

    def get_model_config(self, model_name):
        if 'bert' in model_name.lower():
            return self.get_bert_config()
        temp = {'lstm': self.get_lstm_config(), 'bilstm': self.get_bilstm_config(), 'wordcnn': self.get_wordcnn_config(),}[model_name]
        if self.using_bert_vocab:
            temp['embed_size'] = 128
        return temp

    def get_adv_data(self, model_name):
        num = {'imdb': 1500, 'agnews': 1500, 'sst2': 2000}[self.name]
        path = {}
        for t in ['pwws', 'textfooler']:
            path[t] = f"./dataset/{self.name}.train.adv{num}.{t}.{model_name}.json"
        return num, path

    def get_amda_beta_dist_alpha(self):
        return {'imdb': 8.0, 'sst2': 0.4, 'agnews': 2.0}[self.name]

class config_victim_imdb(config_dataset):
    def __init__(self, using_bert_vocab=False):
        train_path = './dataset/imdb.train.json'
        valid_path = test_path = './dataset/imdb.test.json'
        attack_path = './dataset/imdb.attack.json'
        load_path = {
            'lstm': 'configure the real ckpt .pt path',
            'wordcnn': 'configure the real ckpt .pt path',
            'bert': 'configure the real ckpt .pt path',
            'lm_bert_rmlm': 'configure the real ckpt .pt path',
            'lm_bert_mlm': 'configure the real dir path'
        }
        super(config_victim_imdb, self).__init__('imdb', train_path, valid_path, test_path, attack_path, load_path, using_bert_vocab)
        self.maxlen = 300
        self.n_classes = 2

    def get_lstm_config(self):
        return {'n_classes': self.n_classes, 'vocab_size': self.vocab_size, 'embed_size': 300, 'layer_num': 2, 'hidden_size': 300, 'bid': False, 'pretrained_wv_path':self.pretrained_wv_path, 'update_wv': False, 'dropout': 0.3}

    def get_bilstm_config(self):
        return {'n_classes': self.n_classes, 'vocab_size': self.vocab_size, 'embed_size': 300, 'layer_num': 2, 'hidden_size': 128, 'bid': True, 'pretrained_wv_path': self.pretrained_wv_path, 'update_wv': False, 'dropout': 0.3}

    def get_wordcnn_config(self):
        return {'n_classes': self.n_classes, 'vocab_size': self.vocab_size, 'embed_size': 300, 'kernel_size':[3, 4, 5], 'channel_size':[100, 100, 100], 'mode': 'static', 'pretrained_wv_path': self.pretrained_wv_path, 'dropout': 0.5}

    def get_bert_config(self):
        return {'n_classes': self.n_classes, 'bert_name': 'bert-base-uncased', 'dropout': 0.5, 'tokenizer_name': 'bert-base-uncased'}

class config_victim_agnews(config_dataset):
    def __init__(self, using_bert_vocab=False):
        train_path = './dataset/agnews.train.json'
        valid_path = test_path = './dataset/agnews.test.json'
        attack_path = './dataset/agnews.attack.json'
        load_path = {
            'lstm': 'configure the real ckpt .pt path',
            'wordcnn': 'configure the real ckpt .pt path',
            'bert': 'configure the real ckpt .pt path',
            'lm_bert_rmlm': 'configure the real ckpt .pt path',
            'lm_bert_mlm': 'configure the real dir path'
        }
        super(config_victim_agnews, self).__init__('agnews', train_path, valid_path, test_path, attack_path, load_path, using_bert_vocab)
        self.maxlen = 70
        self.n_classes = 4


    def get_lstm_config(self):
        return {'n_classes': self.n_classes, 'vocab_size': self.vocab_size, 'embed_size': 300, 'layer_num': 2, 'hidden_size': 300, 'bid': False, 'pretrained_wv_path':self.pretrained_wv_path, 'update_wv': False, 'dropout': 0.3}

    def get_bilstm_config(self):
        return {'n_classes': self.n_classes, 'vocab_size': self.vocab_size, 'embed_size': 300, 'layer_num': 2, 'hidden_size': 128, 'bid': True, 'pretrained_wv_path': self.pretrained_wv_path, 'update_wv': False, 'dropout': 0.3}

    def get_wordcnn_config(self):
        return {'n_classes': self.n_classes, 'vocab_size': self.vocab_size, 'embed_size': 300, 'kernel_size':[3, 4, 5], 'channel_size':[100, 100, 100], 'mode': 'static', 'pretrained_wv_path': self.pretrained_wv_path, 'dropout': 0.5}

    def get_bert_config(self):
        return {'n_classes': self.n_classes, 'bert_name': 'bert-base-uncased', 'dropout': 0.5, 'tokenizer_name': 'bert-base-uncased'}

class config_victim_sst2(config_dataset):
    def __init__(self, using_bert_vocab=False):
        train_path = './dataset/sst2.train.json'
        valid_path = './dataset/sst2.valid.json'
        test_path = './dataset/sst2.test.json'
        attack_path = './dataset/sst2.attack.json'
        load_path = {
            'lstm': 'configure the real ckpt .pt path',
            'wordcnn': 'configure the real ckpt .pt path',
            'bert': 'configure the real ckpt .pt path',
            'lm_bert_rmlm': 'configure the real ckpt .pt path',
            'lm_bert_mlm': 'configure the real dir path'
        }
        super(config_victim_sst2, self).__init__('sst2', train_path, valid_path, test_path, attack_path, load_path, using_bert_vocab)
        self.maxlen = 32
        self.n_classes = 2

    def get_lstm_config(self):
        return {'n_classes': self.n_classes, 'vocab_size': self.vocab_size, 'embed_size': 300, 'layer_num': 2, 'hidden_size': 300, 'bid': False, 'pretrained_wv_path':self.pretrained_wv_path, 'update_wv': False, 'dropout': 0.3}

    def get_bilstm_config(self):
        return {'n_classes': self.n_classes, 'vocab_size': self.vocab_size, 'embed_size': 300, 'layer_num': 2, 'hidden_size': 128, 'bid': True, 'pretrained_wv_path': self.pretrained_wv_path, 'update_wv': False, 'dropout': 0.3}

    def get_wordcnn_config(self):
        return {'n_classes': self.n_classes, 'vocab_size': self.vocab_size, 'embed_size': 300, 'kernel_size':[3, 4, 5], 'channel_size':[100, 100, 100], 'mode': 'static', 'pretrained_wv_path': self.pretrained_wv_path, 'dropout': 0.5}

    def get_bert_config(self):
        return {'n_classes': self.n_classes, 'bert_name': 'bert-base-uncased', 'dropout': 0.5, 'tokenizer_name': 'bert-base-uncased'}

class config_rmlm_best_loads:
    imdb = {
            'lstm': 'configure the real ckpt .pt path',
            'wordcnn': 'configure the real ckpt .pt path',
            'bert': 'configure the real ckpt .pt path',
    }
    agnews = {
            'lstm': 'configure the real ckpt .pt path',
            'wordcnn': 'configure the real ckpt .pt path',
            'bert': 'configure the real ckpt .pt path',
    }
    sst2 = {
            'lstm': 'configure the real ckpt .pt path',
            'wordcnn': 'configure the real ckpt .pt path',
            'bert': 'configure the real ckpt .pt path',
    }

    loads = {'imdb': imdb, 'agnews': agnews, 'sst2': sst2}

    infer_hyper = {
        'imdb': {
            'lstm': {
                'rate': 0.25,
                'syn': 32,
                'threshold': 0.22,
            },
            'wordcnn': {
                'rate': 0.25,
                'syn': 32,
                'threshold': 0.37,
            },
            'bert': {
                'rate': 0.1,
                'syn': 32,
                'threshold': 0.38,
            },
        }
    }

    for da in infer_hyper.values():
        for vi in da.values():
            vi['mode'] = 'gumbel'
            vi['update'] = 'no'
            vi['maskop'] = 'rmlm'
            vi['using_for'] = 'attack'


    def get_load_path(self, dataset, victim):
        assert dataset in self.loads
        assert victim in self.loads[dataset]
        return self.loads[dataset][victim], self.infer_hyper[dataset][victim]

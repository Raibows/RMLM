import copy
import torch
from tools import tools_to_gpu
import numpy as np
import OpenAttack
from tqdm import tqdm

def list_to_tensor(*args, dtype=torch.int32):
    return [torch.tensor(arg, dtype=dtype) for arg in args]

class padding_collator():
    def __init__(self, pad_id, using_bert_tokenizer):
        self.pad_id = pad_id
        self.using_bert = using_bert_tokenizer

    def __call__(self, batch):
        # ids, input_ids, type_ids, mask_ids, lengths, labels
        batch = copy.deepcopy(batch)
        ids, input_ids, type_ids, mask_ids, lengths, labels = map(list, zip(*batch))

        maxlen = max(lengths)
        for i in range(len(ids)):
            pad_len = maxlen - lengths[i]
            if pad_len > 0:
                input_ids[i] += [self.pad_id] * pad_len
                if self.using_bert:
                    type_ids[i] += [0] * pad_len
                    mask_ids[i] += [0] * pad_len

        ids, input_ids, type_ids, mask_ids = list_to_tensor(ids, input_ids, type_ids, mask_ids, dtype=torch.int32)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        return ids, input_ids, type_ids, mask_ids, lengths, labels

class OAClassifier(OpenAttack.Classifier):
    def __init__(self, model, train_dataset, rank, using_rmlm, dataset_name, model_name):
        self.model = model
        self.model.eval()
        self.tokenizer = OATensor(train_dataset, rank, 'wordcnn' in model_name)
        self.limit_batch_size = 256
        if using_rmlm:
            self.limit_batch_size = {
                'imdb': 64,
                'agnews': 256,
                'sst2': 512,
            }[dataset_name]
        self.using_detection = False if not using_rmlm else model.using_detection
        self.return_mask = False

    def get_pred_prob(self, input_):
        self.return_mask = True
        temp, mask = self.get_prob(input_)
        self.return_mask = False
        pred = temp.argmax(axis=-1)
        if self.using_detection:
            pred[mask] = -1
        return pred, temp

    def get_pred(self, input_):
        self.return_mask = True
        temp, mask = self.get_prob(input_)
        self.return_mask = False
        pred = temp.argmax(axis=-1)
        if self.using_detection:
            pred[mask] = -1
        return pred

    @torch.no_grad()
    def get_prob(self, input_):
        probs = []
        masks = []
        s = 0
        while s < len(input_):
            end = min(s + self.limit_batch_size, len(input_))
            temp = self.tokenizer.to_tensor(input_[s:end])
            if self.using_detection:
                logits, mask = self.model.forward(*temp, return_mask=True)
                masks.append(mask.to('cpu').detach())
            else:
                logits = self.model.forward(*temp)
            probs.append(logits.to('cpu').detach())
            s += self.limit_batch_size
            del logits, temp
        probs = torch.cat(probs, dim=0)

        if self.using_detection:
            masks = torch.cat(masks, dim=0)
            masks = masks.numpy()

        probs = torch.softmax(probs, dim=-1)
        probs = probs.numpy()
        if self.return_mask:
            if not self.using_detection:
                masks = None
            return probs, masks

        return probs

class OATensor():
    def __init__(self, train_dataset, rank, is_wordcnn):
        self.tokenize_func = train_dataset.tokenizer
        if train_dataset.vocab:
            self.vocab = train_dataset.vocab
        else:
            self.vocab = None
        self.device = torch.device(rank)
        self.pad_id = train_dataset.pad_token[1]
        self.config = train_dataset.config
        self.using_bert_vocab = self.config.using_bert_vocab
        self.is_wordcnn = is_wordcnn

    def to_tensor(self, input) -> list:
        if self.using_bert_vocab:
            if self.vocab: input = [self.vocab.sem_encode(s) for s in input]
            temp = self.tokenize_func(input, return_length=True, max_length=self.config.maxlen, truncation=True)
            input_ids = temp['input_ids']
            type_ids = temp['token_type_ids']
            mask_ids = temp['attention_mask']
            length = temp['length']
        else:
            input_ids, length = [], []
            for item in input:
                ids, l = self.tokenize_func(item)
                input_ids.append(ids)
                length.append(l)
            type_ids = -1
            mask_ids = -1

        maxlen = max(length)
        # max kernel size of wordcnn is 5, so your sentence must pad to at least 5
        if self.is_wordcnn: maxlen = max(maxlen, 5)
        for i in range(len(length)):
            pad_len = maxlen - length[i]
            if pad_len > 0:
                input_ids[i] += [self.pad_id] * pad_len
                if self.using_bert_vocab:
                    type_ids[i] += [0] * pad_len
                    mask_ids[i] += [0] * pad_len

        input_ids, type_ids, mask_ids = list_to_tensor(input_ids, type_ids, mask_ids, dtype=torch.int32)
        length = torch.tensor(length, dtype=torch.int64)
        input_ids, type_ids, mask_ids, length = tools_to_gpu(self.device, input_ids, type_ids, mask_ids, length)
        return [input_ids, type_ids, mask_ids, length]

def process_correct_classified(dataset, victim:OAClassifier, logger, num=None):
    import random
    random.seed(0)
    original_classified_correct_datas = []
    x_map_prob = {}
    idx = [i for i in range(len(dataset))]
    if isinstance(num, int):
        random.shuffle(idx)
        logger.info(f"select specific {num} data from {len(dataset)}")
    else:
        num = len(dataset)

    with tqdm(desc='predicting', total=len(dataset)) as pbar:
        cnt = 0
        for i in idx:
            pbar.update(1)
            x, y = dataset.data[i], dataset.label[i]
            pred, prob = victim.get_pred_prob([x])
            if pred[0] == y:
                original_classified_correct_datas.append({'x': x, 'y': y})
                x_map_prob[x] = prob[0]
            cnt += 1
            if cnt == num: break


    logger.info(f'only the correct classified num {len(original_classified_correct_datas)}/{cnt} acc {len(original_classified_correct_datas)/cnt:.5f} will under attack!')
    return original_classified_correct_datas, x_map_prob, cnt
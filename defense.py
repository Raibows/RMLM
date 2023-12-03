import os
import torch
import torch.nn as nn
from tools import tools_json_load, tools_get_logger
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertTokenizerFast
from config import config_dataset
from victims import build_model
import os
import random

class RockSolidDefender(nn.Module):
    def __init__(self, config:config_dataset, backbone_model, backbone_model_config, backbone_path, device, tokenizer, rmlm_config, using_detection=False):
        super(RockSolidDefender, self).__init__()
        device = int(device)
        assert rmlm_config['update'] in {'no', 'last', 'whole'}
        assert rmlm_config['mode'] in {'argmax', 'multinomial', 'gumbel'}
        assert rmlm_config['maskop'] in {'rmlm', 'no', 'mlm'}
        assert isinstance(rmlm_config['threshold'], float) or rmlm_config['threshold'] is None
        assert config.using_bert_vocab, "backbone model needs to use bert_vocab"

        self.tokenizer:BertTokenizerFast = tokenizer
        self.backbone = build_model(backbone_model, backbone_model_config, config, backbone_path, device)
        self.mode = rmlm_config['mode']
        self.update = rmlm_config['update']
        self.threshold = rmlm_config['threshold']
        self.backbone_name = backbone_model
        self.using_detection = using_detection
        self.device = device
        self.maskop = rmlm_config['maskop']
        lm_path = config.get_load_path('lm_bert_rmlm')
        if self.maskop == 'rmlm':
            self.mask_policy = {
                'keep': 0.1,
                'rand': 0.1,
                'unk': 0.1,
                'mask': 0.2,
                'syn': 0.5,
            }
            self.mlm_prob = rmlm_config['rate']
            self.syn_max_num = rmlm_config['syn']  # save your gpu memory
            self.pad_num = self.syn_max_num // 5
            self.synonym_matrix = [[] for _ in self.tokenizer.vocab.keys()]
            self.__prepare_synonym_matrix(config.syn_w2list_path)
            self.synonym_matrix = torch.tensor(self.synonym_matrix, dtype=torch.int32, requires_grad=False, device='cpu')
        elif self.maskop == 'mlm':
            lm_path = config.get_load_path('lm_bert_mlm')
            self.mlm_prob = rmlm_config['rate']
            self.mask_policy = {
                'keep': 0.1,
                'rand': 0.1,
                'mask': 0.8
            }
        else:
            self.maskop = None

        if lm_path and os.path.exists(lm_path):
            self.detector: BertForMaskedLM = BertForMaskedLM.from_pretrained(lm_path)
        else:
            tools_get_logger('defense').warning('you are not specifying the fine-tuned rmlm BERT, so it is initialized with pretrained bert-base-uncased')
            self.detector = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.detector.to(device)
        self.detector.eval()

    def __prepare_synonym_matrix(self, bert_synonym_json):
        bert_synonym_json = tools_json_load(bert_synonym_json)
        word2idx = self.tokenizer.vocab
        for k, i in word2idx.items():
            self.synonym_matrix[i] = [word2idx[t] for t in bert_synonym_json[k][:self.syn_max_num]]
            pad_len = self.syn_max_num - len(self.synonym_matrix[i])
            offset = max(pad_len - self.pad_num, 0)
            random_words = [random.randint(0, len(self.tokenizer))] * int(0.2 * offset)
            unk_words = [self.tokenizer.unk_token_id] * int(0.1 * offset)
            mask_words = [self.tokenizer.mask_token_id] * (offset - len(random_words) - len(unk_words))
            self.synonym_matrix[i] = self.synonym_matrix[i] + random_words + unk_words + mask_words
            self.synonym_matrix[i] += [-100] * (pad_len - offset)

    def get_optimized_params(self):
        no_decay = {"bias", "LayerNorm.weight"}
        backbone_params = [
                 {"params": [p for n, p in self.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
                  "weight_decay": 1e-3,},
                 {"params": [p for n, p in self.backbone.named_parameters() if any(nd in n for nd in no_decay)],
                  'weight_decay': 0.0,}
        ] if 'bert' not in self.backbone_name else [
            {"params": [p for n, p in self.backbone.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": 1e-3, 'lr': 3e-5},
            {"params": [p for n, p in self.backbone.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': 3e-5}
        ]
        if self.update == 'no':
            return backbone_params
        elif self.update == 'last':
            params = [
                {"params": [p for n, p in self.detector.cls.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": 1e-3, 'lr': 1e-5},
                {"params": [p for n, p in self.detector.cls.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0, 'lr': 1e-5},
            ]
        elif self.update == 'whole':
            params = [
                {"params": [p for n, p in self.detector.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": 1e-3, 'lr': 1e-6},
                {"params": [p for n, p in self.detector.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0, 'lr': 1e-6},
            ]
        else:
            raise RuntimeError()
        params += backbone_params
        return params

    @torch.no_grad()
    def lm_forward_no_grad(self, X, types, masks):
        logits = self.detector.forward(X, attention_mask=masks, token_type_ids=types, return_dict=True)['logits']
        bsz = X.shape[0]
        if self.mode == 'argmax':
            tokens = torch.argmax(logits, dim=-1)
        elif self.mode == 'multinomial':
            tokens = torch.multinomial(logits.reshape(-1, logits.shape[-1]).softmax(dim=-1), num_samples=1).reshape(bsz, -1)
            tokens = torch.masked_fill(tokens, ~masks.bool(), self.tokenizer.pad_token_id)
        elif self.mode == 'gumbel':
            tokens = F.gumbel_softmax(logits, tau=1, hard=not self.training)
        else:
            raise NotImplementedError(f'{self.mode} is error')
        return tokens

    def lm_forward_with_grad(self, X, types, masks):
        logits = self.detector.forward(X, attention_mask=masks, token_type_ids=types, return_dict=True)['logits']
        bsz = X.shape[0]
        if self.mode == 'argmax':
            tokens = torch.argmax(logits, dim=-1)
        elif self.mode == 'multinomial':
            tokens = torch.multinomial(logits.reshape(-1, logits.shape[-1]).softmax(dim=-1), num_samples=1).reshape(bsz, -1)
            tokens = torch.masked_fill(tokens, ~masks.bool(), self.tokenizer.pad_token_id)
        elif self.mode == 'gumbel':
            tokens = F.gumbel_softmax(logits, tau=1, hard=not self.training)
        else:
            raise NotImplementedError(f'{self.mode} is error')
        return tokens

    def get_logits(self, *inputs):
        X, types, masks, X_len = inputs
        if self.maskop: X, masks = self.__rmlm(X, masks)
        if self.update != 'no':
            tokens = self.lm_forward_with_grad(X, types, masks)
            logits = self.backbone.forward(tokens, types, masks, X_len, gumbel=self.mode == 'gumbel')
        else:
            tokens = self.lm_forward_no_grad(X, types, masks)
            logits = self.backbone.forward(tokens.detach(), types.detach(), masks.detach(), X_len.detach(), gumbel=self.mode == 'gumbel')
        return logits

    
    @torch.no_grad()
    def get_entropy(self, logits_0, logits_1):
        detected:torch.Tensor = logits_0.argmax(dim=-1) != logits_1.argmax(dim=-1)
        if torch.any(detected):
            p0 = torch.softmax(logits_0, dim=-1)
            p1 = torch.softmax(logits_1, dim=-1)
            e0 = torch.sum(-p0[detected] * torch.log(p0[detected]), dim=-1)
            e1 = torch.sum(-p1[detected] * torch.log(p1[detected]), dim=-1)
            clean = torch.logical_or(e0 > self.threshold, e1 > self.threshold)
            if torch.any(clean):
                replace = torch.logical_and(e1 < e0, clean)
                logits_0[detected.clone()] = logits_0[detected].masked_scatter(replace.unsqueeze(-1), logits_1[detected][replace])
                detected[detected.clone()] = detected.masked_select(detected).masked_fill(clean, False)

        return logits_0, detected

    @torch.no_grad()
    def forward_with_threshold(self, *inputs, return_mask=False):
        if not self.using_detection:
            logits = self.get_logits(*inputs)
            if return_mask:
                return logits, torch.full([logits.shape[0]], False)
            return logits
        else:
            logits_0 = self.get_logits(*inputs)
            logits_1 = self.get_logits(*inputs)
            logits_0, detected = self.get_entropy(logits_0, logits_1)

            if return_mask:
                return logits_0, detected
            else:
                return logits_0


    def forward(self, *inputs, return_mask=False):
        if not self.using_detection:
            logits = self.get_logits(*inputs)
            if return_mask:
                return logits, torch.full([logits.shape[0]], False)
            else:
                return logits
        else:
            if isinstance(self.threshold, float):
                return self.forward_with_threshold(*inputs, return_mask=return_mask)
            else:
                with torch.no_grad():
                    logits_0 = self.get_logits(*inputs)
                    logits_1 = self.get_logits(*inputs)
                    detected = logits_0.argmax(dim=-1) != logits_1.argmax(dim=-1)
                    if return_mask:
                        return logits_0, detected
                    else:
                        return logits_0

        assert False

    def __rmlm(self, input_ids:torch.Tensor, mask):
        input_ids = input_ids.to('cpu')
        mask = mask.to('cpu')
        prob = torch.full(input_ids.shape, self.mlm_prob,).masked_fill(~mask.bool(), 0.0)
        masked_indices = torch.bernoulli(prob).bool()

        for k, prob in self.mask_policy.items():
            if k == 'keep': continue
            if k == 'rand':
                indices = torch.bernoulli(torch.full(input_ids.shape, prob)).bool() & masked_indices
                random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.int32)
                input_ids[indices] = random_words[indices]
            elif k == 'unk':
                indices = torch.bernoulli(torch.full(input_ids.shape, prob)).bool() & masked_indices
                input_ids[indices] = self.tokenizer.unk_token_id
            elif k == 'mask':
                indices = torch.bernoulli(torch.full(input_ids.shape, prob)).bool() & masked_indices
                input_ids[indices] = self.tokenizer.mask_token_id
            elif k == 'syn':
                indices = torch.bernoulli(torch.full(input_ids.shape, prob)).bool() & masked_indices
                words_idx = input_ids[indices].long()
                syn_for_words = self.synonym_matrix[words_idx]
                candidates = torch.full(syn_for_words.shape, 1 / self.syn_max_num).masked_fill(syn_for_words == -100, -1e9).softmax(dim=1)
                candidates_index = torch.multinomial(candidates, num_samples=1).squeeze()
                input_ids[indices] = syn_for_words[torch.arange(syn_for_words.shape[0]), candidates_index]
            masked_indices = masked_indices & ~indices

        return input_ids.to(self.device), mask.to(self.device)

if __name__ == '__main__':
    pass
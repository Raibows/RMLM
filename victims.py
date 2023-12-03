import os
import torch
import torch.nn as nn
from tools import tools_json_load
from transformers import BertModel
from tools import tools_get_logger
from config import config_dataset


def build_model(model_name, model_config, config:config_dataset, load_path, rank,):
    device = torch.device(rank)
    if not config.using_bert_vocab and model_config['pretrained_wv_path']:
        model_config['pretrained_wv_path'] = f"{model_config['pretrained_wv_path']}.{config.name}"

    if rank == 0:
        tools_get_logger('model').info(model_config)

    if model_name == 'lstm':
        assert model_config['bid'] == False
        model = LSTM(**model_config)
    elif model_name == 'bilstm':
        assert model_config['bid'] == True
        model = LSTM(**model_config)
    elif model_name == 'wordcnn':
        model = WordCNN(**model_config)
    elif 'bert' in model_name.lower():
        model = BERT(**model_config)
    else:
        raise NotImplementedError(f'not support {model_name}')
    if load_path:
        if load_path == 'config':
            load_path = config.get_load_path(model_name)
        load_dict = torch.load(load_path, map_location=device)
        model_dict = model.state_dict()
        states = {k: v for k, v in load_dict.items() if k in model_dict}
        not_loaded = [k for k in load_dict.keys() if k not in model_dict]
        need_loaded_missing = [k for k in model_dict.keys() if k not in load_dict]
        model_dict.update(states)
        if rank == 0:
            tools_get_logger('model').info(f'loading model from {load_path} but keys below are not loaded\n{not_loaded}\nneed load below keys but missing\n{need_loaded_missing}')
        model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model.to(device)
    model.eval()

    return model

def build_embedding_layer(pretrained_wv_path, vocab_size, embed_size, update_wv):
    if isinstance(pretrained_wv_path, str):
        assert os.path.exists(pretrained_wv_path), f"make sure {pretrained_wv_path} exists"
        wv = torch.tensor(tools_json_load(pretrained_wv_path))
        layer = nn.Embedding.from_pretrained(wv)
    else:
        layer = nn.Embedding(vocab_size, embed_size)
    layer.weight.requires_grad = update_wv
    return layer

class LSTM(nn.Module):
    def __init__(self, n_classes, vocab_size, embed_size, layer_num, hidden_size, bid=False, pretrained_wv_path=None, update_wv=True, dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bid = bid
        self.embedding_layer = build_embedding_layer(pretrained_wv_path, vocab_size, embed_size, update_wv)
        self.encoder = nn.LSTM(embed_size, hidden_size, layer_num, bidirectional=self.bid, batch_first=True)
        self.output_size = hidden_size * 2 if self.bid else hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.mlp = nn.Linear(self.output_size, n_classes)

    def encode_pack(self, embeds, X, X_len):
        sort = torch.sort(X_len, descending=True)
        sort_len, idx_sort = sort.values, sort.indices
        idx_reverse = torch.argsort(idx_sort)
        embeds = embeds.index_select(0, idx_sort)
        packed = nn.utils.rnn.pack_padded_sequence(embeds, sort_len.cpu(), batch_first=True)
        outs, (h, c) = self.encoder.forward(packed)
        outs, _ = nn.utils.rnn.pad_packed_sequence(outs, padding_value=0.0, batch_first=True)
        outs = outs.index_select(0, idx_reverse)
        outs = outs[torch.arange(X.size(0)), X_len - 1, :]
        return outs

    def forward(self, *inputs, return_sen_embed=False, gumbel=False):
        X, _, _, X_len = inputs
        if gumbel:
            embeds = X @ self.embedding_layer.weight
        else:
            embeds = self.embedding_layer.forward(X)  # [batch, sen_len, embed_size]
        embeds = self.dropout(embeds)
        outs = self.encode_pack(embeds, X, X_len)
        logits = self.mlp.forward(self.dropout(outs))
        if return_sen_embed:
            return logits, outs
        return logits


class WordCNN(nn.Module):
    def __init__(self, n_classes, vocab_size, embed_size, kernel_size:list, channel_size:list, mode, pretrained_wv_path=None, dropout=0.5):
        super(WordCNN, self).__init__()
        self.mode = mode
        if mode is None:
            self.embedding_layer = build_embedding_layer(None, vocab_size, embed_size, True)
        elif self.mode == 'static':
            assert pretrained_wv_path is not None
            self.embedding_layer = build_embedding_layer(pretrained_wv_path, vocab_size, embed_size, False)
        elif self.mode == 'update':
            assert pretrained_wv_path is not None
            self.embedding_layer = build_embedding_layer(pretrained_wv_path, vocab_size, embed_size, True)
        elif self.mode == 'dynamic':
            assert pretrained_wv_path is not None
            self.static_embedding_layer = build_embedding_layer(pretrained_wv_path, vocab_size, embed_size, False)
            self.update_embedding_layer = build_embedding_layer(pretrained_wv_path, vocab_size, embed_size, True)
            embed_size *= 2
        else:
            raise NotImplementedError('you have to choose mode from [None, static, update, dynamic]')
        self.embed_size = embed_size
        self.convs = nn.ModuleList()
        for c, k in zip(channel_size, kernel_size):
            self.convs.append(
                nn.Conv1d(in_channels=self.embed_size,
                          out_channels=c,
                          kernel_size=k)
            )
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.mlp = nn.Linear(in_features=sum(channel_size), out_features=n_classes)

    def __forward_embedding_layer(self, X, gumbel=False):
        if self.mode == 'dynamic':
            static_embeds = self.static_embedding_layer.forward(X) if not gumbel else X @ self.static_embedding_layer.weight
            update_embeds = self.update_embedding_layer.forward(X) if not gumbel else X @ self.update_embedding_layer.weight
            embeds = torch.cat([static_embeds, update_embeds], dim=2)
        else:
            embeds = self.embedding_layer.forward(X) if not gumbel else X @ self.embedding_layer.weight

        return self.dropout(embeds)

    def forward(self, *inputs, gumbel=False):
        X, _, _, X_len = inputs
        embeds = self.__forward_embedding_layer(X, gumbel)
        embeds = embeds.permute(0, 2, 1) # [batch, embed_size, sen_len]
        outs = torch.cat([self.pool(self.relu(conv(embeds))).squeeze(-1) for conv in self.convs], dim=1)
        logits = self.mlp(self.dropout(outs))

        return logits

class BERT(nn.Module):
    def __init__(self, n_classes, bert_name, dropout, **kwargs):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp = nn.Linear(768, n_classes)

    def forward(self, *inputs, gumbel=False):
        X, types, masks, X_len = inputs
        if gumbel:
            inputs_embeds = X @ self.bert.embeddings.word_embeddings.weight
            cls = self.bert.forward(None, masks, types, inputs_embeds=inputs_embeds, return_dict=True)['last_hidden_state'][:, 0, :]
        else:
            cls = self.bert.forward(X, masks, types, return_dict=True)['last_hidden_state'][:, 0, :]
        logits = self.mlp.forward(self.dropout(cls))

        return logits



if __name__ == '__main__':
    pass







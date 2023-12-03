"""
fine-tuning with proposed masking scheme
"""
import random
from transformers import BertForMaskedLM, AutoTokenizer, BertTokenizerFast
from transformers import DataCollatorForLanguageModeling, BatchEncoding
from transformers import Trainer, TrainingArguments
import os
import torch
from torch.utils.data import Dataset
from tools import *
from config import *
import math
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', default='imdb')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--eval_step', type=int, default=500)
parser.add_argument('--resume_ckpt', type=str, default=None)
parser.add_argument('--rmlm', type=lambda x: 'y' in x.lower(), default=True)
args = parser.parse_args()
tools_setup_seed(0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.device}"

logger = tools_get_logger('mlm')

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, datas: list, block_size: int):
        temp = tokenizer(datas, max_length=block_size, truncation=True)
        self.examples = temp["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

class DataCollatorRMLM():
    def __init__(self, tokenizer: BertTokenizerFast, bert_synonym_json):
        self.tokenizer = tokenizer
        self.mask_policy = {
            'keep': 0.1,
            'rand': 0.1,
            'unk': 0.1,
            'mask': 0.2,
            'syn': 0.5,
        }
        assert sum(self.mask_policy.values()) == 1.0
        self.mlm: bool = True
        self.mlm_probability: float = 0.25
        self.return_tensors: str = "pt"
        self.pad_to_multiple_of: str = None
        self.syn_max_num = 32 # save your gpu memory
        self.pad_num = 32 // 5
        self.synonym_matrix = [[] for _ in self.tokenizer.vocab.keys()]
        self._prepare_synonym_matrix(bert_synonym_json)
        self.synonym_matrix = torch.tensor(self.synonym_matrix, dtype=torch.long)



    def _prepare_synonym_matrix(self, bert_synonym_json):
        logger.info(f'starting preparing synonym matrix from {bert_synonym_json}')
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

    def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        import numpy as np
        import torch

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)

        # Check if padding is necessary.

        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0]:] = example
        return result

    def __call__(self, examples) -> dict:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": self._torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs, special_tokens_mask= None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        for k, prob in self.mask_policy.items():
            if k == 'keep': continue
            if k == 'rand':
                indices = torch.bernoulli(torch.full(labels.shape, prob)).bool() & masked_indices
                random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
                inputs[indices] = random_words[indices]
            elif k == 'unk':
                indices = torch.bernoulli(torch.full(labels.shape, prob)).bool() & masked_indices
                inputs[indices] = self.tokenizer.unk_token_id
            elif k == 'mask':
                indices = torch.bernoulli(torch.full(labels.shape, prob)).bool() & masked_indices
                inputs[indices] = self.tokenizer.mask_token_id
            elif k == 'syn':
                indices = torch.bernoulli(torch.full(labels.shape, prob)).bool() & masked_indices
                # get the selected word idx
                words_idx = inputs[indices]
                # get the corresponding synonyms of the selected words
                syn_for_words = self.synonym_matrix[words_idx]
                # sample the candidate synonyms, note that we have to mask the specific id -100 that means nothing
                candidates = torch.full(syn_for_words.shape, 1/self.syn_max_num).masked_fill(syn_for_words == -100, -1e9).softmax(dim=1)
                # multinomial sampling
                candidates_index = torch.multinomial(candidates, num_samples=1).squeeze()
                # implement the replacement
                inputs[indices] = syn_for_words[torch.arange(syn_for_words.shape[0]), candidates_index]
            # every turn the masked_indices need to update, mask the selected ops for the following ops
            masked_indices = masked_indices & ~indices

        return inputs, labels

logger.info(args.__dict__)
config:config_dataset = {
        'imdb': config_victim_imdb(using_bert_vocab=True),
        'agnews': config_victim_agnews(using_bert_vocab=True),
        'sst2': config_victim_sst2(using_bert_vocab=True),
    }[args.dataset]

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
train_data = tools_json_load(config.train_path)['data']
valid_data = tools_json_load(config.valid_path)['data']

if args.rmlm:
    data_collator = DataCollatorRMLM(tokenizer=tokenizer, bert_synonym_json=config.syn_w2list_path)
    output_dir = f"./checkpoint/{args.dataset}/mlm/{tools_get_time()}_rmlm"
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.25)
    output_dir = f"./checkpoint/{args.dataset}/mlm/{tools_get_time()}_mlm"

logger.info(f'output_dir is {output_dir}')
tools_copy_all_suffix_files(f"{output_dir}/codes")
tools_json_dump(args.__dict__, f"{output_dir}/args.json")

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    datas=train_data,
    block_size=config.maxlen,
)
valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    datas=valid_data,
    block_size=config.maxlen,
)
logger.info('starting training')

training_args = TrainingArguments(
    output_dir=output_dir,
    do_train=True,
    do_eval=True,
    evaluation_strategy='steps',
    overwrite_output_dir=True,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch,
    save_steps=args.eval_step,
    save_total_limit=3,
    prediction_loss_only=True,
    seed=0,
    load_best_model_at_end=True,
    logging_first_step=True,
    eval_steps=args.eval_step,
    fp16=torch.cuda.is_available(),
    fp16_backend='auto'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=args.resume_ckpt)
trainer.save_model(output_dir)
eval_results = trainer.evaluate()
logger.info(f"Perplexity: {math.exp(eval_results['eval_loss']):.5f}")
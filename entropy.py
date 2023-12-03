from config import *
from tools import *
from utils import *
from vocab import Vocab
from data import *
import torch
from tqdm import tqdm
import os
from defense import RockSolidDefender
import argparse


def test_get_entropy_forward(logits_0, logits_1):
    detected: torch.Tensor = logits_0.argmax(dim=-1) != logits_1.argmax(dim=-1)
    if torch.any(detected):
        p0 = torch.softmax(logits_0, dim=-1)
        p1 = torch.softmax(logits_1, dim=-1)
        e0 = torch.sum(-p0 * torch.log(p0), dim=-1)
        e1 = torch.sum(-p1 * torch.log(p1), dim=-1)
        clean = torch.logical_or(e0 > 0.3, e1 > 0.3)
        clean = torch.logical_and(detected, clean)
        replace = torch.logical_and(detected, e1 < e0)
        logits_0[replace] = logits_1[replace]
        detected[clean] = False

    return logits_0, detected

def get_entropy(logits):
    assert logits.dim() == 1
    assert torch.abs(torch.sum(logits) - 1.0) < 1e-6
    return (-logits @ torch.log(logits).T).item()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['imdb', 'agnews', 'sst2'])
parser.add_argument('--victim', choices=['lstm', 'bert', 'wordcnn'])
parser.add_argument('--device', default='0', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device}'
    device = 0
    tools_setup_seed(0)
    dataset = args.dataset
    victim = args.victim
    model_load_path, default_defender_config = config_rmlm_best_loads().get_load_path(args.dataset, args.victim)
    default_defender_config['threshold'] = None
    config: config_dataset = {
        'imdb': config_victim_imdb,
        'agnews': config_victim_agnews,
        'sst2': config_victim_sst2,
    }[dataset](using_bert_vocab=True)
    model_config = config.get_model_config(victim)
    model_config['pretrained_wv_path'] = None
    config.rmlm_config = default_defender_config
    if victim == 'wordcnn':
        model_config['mode'] = None

    path = config.train_path
    print(config.rmlm_config)

    train_dataset = ClassificationDataset(config.train_path, config, victim, model_config, vocab=None)
    attack_dataset = ClassificationDataset(path, config, victim, model_config, vocab=train_dataset.vocab)
    oatensor = OATensor(train_dataset, 0, 'wordcnn' in victim)

    model = RockSolidDefender(config, args.victim, model_config, None, args.device, train_dataset.tokenizer, config.rmlm_config, using_detection=True)
    model.load_state_dict(torch.load(model_load_path, map_location=torch.device(0)))
    model.to(device)
    model.eval()

    print('calculating average entropy on train dataset')
    correct = []
    incorrect = []
    clean_acc = 0
    entropy_help = []
    for i in tqdm(range(len(attack_dataset)), desc=f'{args.victim} {args.dataset } entropy on train set'):
        x, y = attack_dataset.data[i], attack_dataset.label[i]
        temp = oatensor.to_tensor([x])
        logits_0 = model.get_logits(*temp)
        logits_1 = model.get_logits(*temp)

        detected = logits_0.argmax(dim=-1) != logits_1.argmax(dim=-1)
        logits_0 = logits_0.softmax(dim=-1)
        logits_1 = logits_1.softmax(dim=-1)
        logits_0 = logits_0[0]
        logits_1 = logits_1[0]
        detected = detected[0]

        if not detected.item():
            correct.append(get_entropy(logits_0))
            correct.append(get_entropy(logits_1))
            pred = logits_0.argmax(dim=-1).item()
        else:
            e1, e2 = get_entropy(logits_0), get_entropy(logits_1)
            incorrect.append(e1)
            incorrect.append(e2)
            pred = -1

        if pred == y:
            clean_acc += 1

    print(f"acc {clean_acc / len(attack_dataset)}")

    correct_avg = 0.0
    for item in correct:
        correct_avg += item
    incorrect_avg = 0.0
    for item in incorrect:
        incorrect_avg += item

    print(f"average entropy of mis-detected samples{incorrect_avg / len(incorrect)}")

print(args.__dict__)
import time
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.multiprocessing as mp
import torch.distributed as dist
import re
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from argparse import ArgumentParser
from data import ClassificationDataset
from tools import *
from utils import *
from config import *
from metric import Metric
from victims import build_model
from defense import RockSolidDefender


parser = ArgumentParser()
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--optim', type=str, default='adamw', choices=['sgd', 'adamw', 'adam'])
parser.add_argument('--save_best', type=int, default=1)
parser.add_argument('--save_type', type=str, default='test')
parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'agnews', 'sst2'])
parser.add_argument('--model', type=str, default='lstm', choices=['wordcnn', 'lstm', 'bilstm', 'bert'])
parser.add_argument('--using_bert_vocab', type=lambda x: 'y' in x.lower(), default='no')
parser.add_argument('--only_eval', type=lambda x: 'y' in x.lower(), default='no')
parser.add_argument('--load', default=None)
parser.add_argument('--eval_path', default=None, help='choose from [train, valid, test, attack or a json file path]')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--metric_name', type=str, default='valid_acc')
parser.add_argument('--note', type=str, default=None)
parser.add_argument('--using_rmlm', type=lambda x: 'y' in x.lower(), default='no')
parser.add_argument('--rmlm_mode', type=str, default=None, choices=['gumbel', 'argmax', 'multinomial'])
parser.add_argument('--rmlm_update', type=str, default=None, choices=['no', 'last', 'whole'])
parser.add_argument('--rmlm_maskop', default=None, choices=['rmlm', 'no', 'mlm'])
parser.add_argument('--rmlm_mask_rate', type=float, default=0.25)
parser.add_argument('--rmlm_max_syn', type=int, default=32)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.device}"


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl" if torch.cuda.is_available() else 'gloo', rank=rank, world_size=world_size)

def save_model(metric, ep, args, ddp_model, save_dir):
    last_best_ep = list(metric.find_best(args.metric_name, lower_is_better=False, top=args.save_best).keys())
    best_ep = last_best_ep[0]
    best_metric = metric.records[best_ep][args.metric_name]

    if ep in last_best_ep:
        if len(metric.best_path) == args.save_best:
            os.remove(metric.best_path[args.save_best - 1])
            del metric.best_path[args.save_best - 1]

        mapped = {}
        for old_idx, old_path in metric.best_path.items():
            ep_t = int(re.findall("epoch\d+", old_path)[0].strip('epoch'))
            new_idx = last_best_ep.index(ep_t)
            new_path = old_path.replace(f'best{old_idx}', f'best{new_idx}')
            mapped[old_idx] = (new_idx, old_path, new_path)

        for old_idx, (new_idx, old_path, new_path) in mapped.items():
            os.rename(old_path, new_path)
            metric.best_path[new_idx] = new_path

        idx = last_best_ep.index(ep)
        save_path = f"{save_dir}/best{idx}_epoch{ep}.{args.metric_name}{metric.records[ep][args.metric_name]:.5f}.pt"
        torch.save(ddp_model.module.state_dict(), save_path)
        metric.best_path[idx] = save_path

    return best_ep, best_metric

def run(rank, world_size, args, port, save_dir, config:config_dataset, model_config:dict):
    setup(rank, world_size, port)
    logger = tools_get_logger(f'train-{save_dir}')
    if rank == 0:
        logger.info(f"{args}\n{model_config}")
    tools_setup_seed(0)
    metric = Metric(save_file=f"{save_dir}/metric.log")

    valid_path = config.valid_path
    if args.eval_path is not None:
        if args.eval_path in {'train', 'valid', 'test', 'attack'}:
            valid_path = eval(f'config.{args.eval_path}_path')
        else:
            valid_path = args.eval_path

    train_dataset = ClassificationDataset(config.train_path, config, args.model, model_config, vocab=None)
    valid_dataset = ClassificationDataset(valid_path, config, args.model, model_config, vocab=train_dataset.vocab)
    train_dataset.pack_samples()
    valid_dataset.pack_samples()
    collator = padding_collator(train_dataset.pad_token[1], using_bert_tokenizer=config.using_bert_vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=0), collate_fn=collator)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch, sampler=DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=0), collate_fn=collator)

    if args.using_rmlm:
        if args.only_eval:
            args.load, config.rmlm_config = config_rmlm_best_loads().get_load_path(args.dataset, args.model)
            model = RockSolidDefender(config, args.model, model_config, None, rank, train_dataset.tokenizer, config.rmlm_config, using_detection=False)
            model.load_state_dict(torch.load(args.load, map_location=torch.device(rank)))
        else:
            # here the load path is the backbone victim load path, for joint training
            if args.load is None:
                args.load = config.get_load_path(args.model)
            model = RockSolidDefender(config, args.model, model_config, args.load, rank, train_dataset.tokenizer, config.rmlm_config, using_detection=False)
    else:
        model = build_model(args.model, model_config, config, load_path=args.load, rank=rank)



    if args.device == 'cpu':
        temp = train_dataset.using_bert or (args.using_rmlm and args.rmlm_update == 'no')
        ddp_model = ddp(model, find_unused_parameters=temp)
    else:
        temp = train_dataset.using_bert or (args.using_rmlm and args.rmlm_update == 'no')
        ddp_model = ddp(model, device_ids=[rank], find_unused_parameters=temp)


    weight_decay = 1e-3
    no_decay = {"bias", "LayerNorm.weight"}


    if args.using_rmlm:
        params = model.get_optimized_params()
    elif train_dataset.using_bert:
        params = [
            {"params": [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": 1e-3, 'lr': 3e-5},
            {"params": [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': 3e-5}
        ]
    else:
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

    optimizer = {
        'sgd': optim.SGD(params, lr=args.lr, weight_decay=weight_decay),
        'adam': optim.Adam(params, lr=args.lr, weight_decay=weight_decay),
        'adamw': optim.AdamW(params, lr=args.lr, weight_decay=weight_decay)
    }[args.optim]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=6 if args.optim == 'sgd' else 2, min_lr=9e-5, verbose=True)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (0.01 + 0.99 * step / args.warmup) if step < args.warmup else 1.0)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    dist.barrier()

    if args.only_eval:
        if args.load is None and rank == 0:
            logger.warn(f"you are in only evaluate mode expecting load a fully trained model but load path is {args.load}")
        valid_loader.sampler.set_epoch(0)
        valid_loss, valid_acc = valid_process(ddp_model, valid_loader, rank, world_size, optimizer, criterion, metric, 0, save_dir)
        logger.info(f"{'---' * 10}\n"
                    f'evaluate {args.model} load from {args.load} done!\n'
                    f'evaluate data on {valid_path}\n'
                    f'accuracy {valid_acc:.5f} loss {valid_loss:.5f}\n'
                    f"{'---' * 10}\n")
        dist.destroy_process_group()
        return None

    global_step = 0
    for ep in range(args.epoch):
        train_loader.sampler.set_epoch(ep)
        valid_loader.sampler.set_epoch(ep)
        if rank == 0:
            start_time = time.perf_counter()
            logger.info(f"epoch {ep} starts train_process")

        train_loss, global_step = train_process(ddp_model, train_loader, rank, world_size, optimizer, criterion, metric, ep, global_step, warmup_scheduler,)
        valid_loss, valid_acc = valid_process(ddp_model, valid_loader, rank, world_size, optimizer, criterion, metric, ep, save_dir)

        if rank == 0:
            best_ep, best_metric = save_model(metric, ep, args, ddp_model, save_dir)
            logger.info(f"{'---' * 10}\n"
                        f"epoch {ep} done! cost {time.perf_counter() - start_time:.0f} seconds\n"
                        f"train_loss {train_loss:.4f} valid_loss {valid_loss:.4f} valid_acc {valid_acc:.4f}\n"
                        f"best epoch {best_ep} with {args.metric_name} {best_metric:.4f}\n"
                        f"{'---' * 10}\n")
            metric.save()

        if ep > args.warmup:
            scheduler.step(metric.records[ep][args.metric_name])

    dist.destroy_process_group()

def train_process(model, train_loader, rank, world_size, optimizer, criterion, metric:Metric, epoch, global_step, warmup_scheduler):
    loss_mean = 0.0
    model.train()
    with tqdm(desc=f"train {epoch}", disable=rank != 0, total=len(train_loader)) as pbar:
        for i, (ids, input_ids, type_ids, mask_ids, length, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                input_ids, type_ids, mask_ids, length, labels = tools_to_gpu(rank, input_ids, type_ids, mask_ids, length, labels)
            logits = model.forward(input_ids, type_ids, mask_ids, length)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f"loss {loss.item():.4f} step {global_step}")
            loss_mean += loss.item()
            pbar.update(1)
            if global_step < args.warmup:
                warmup_scheduler.step()
            global_step += 1

    loss_mean = torch.tensor(loss_mean / len(train_loader), dtype=torch.float, device=rank)
    dist.all_reduce(loss_mean, op=dist.ReduceOp.SUM)
    loss_mean = loss_mean.item() / world_size
    metric.add_record('train_loss', epoch, loss_mean)

    return loss_mean, global_step

@torch.no_grad()
def valid_process(model, valid_loader, rank, world_size, optimizer, criterion, metric:Metric, epoch, save_dir):
    optimizer.zero_grad()
    loss_mean = 0.0
    model.eval()
    save_data = {'ids': [], 'predicts': [], 'labels': []}
    with tqdm(desc=f"valid {epoch}", disable=rank != 0, total=len(valid_loader)) as pbar:
        for i, (ids, input_ids, type_ids, mask_ids, length, labels) in enumerate(valid_loader):
            if torch.cuda.is_available():
                input_ids, type_ids, mask_ids, length, labels = tools_to_gpu(rank, input_ids, type_ids, mask_ids, length, labels)

            logits = model.forward(input_ids, type_ids, mask_ids, length)
            predicts = logits.argmax(dim=-1).squeeze().tolist()
            save_data['ids'] += ids.tolist()
            if isinstance(predicts, int): predicts = [predicts]
            save_data['predicts'] += predicts
            save_data['labels'] += labels.tolist()

            loss = criterion(logits, labels)
            pbar.set_postfix_str(f"loss {loss.item():.4f}")
            loss_mean += loss.item()
            pbar.update(1)

    tools_json_dump(save_data, f"{save_dir}/{rank}.temp")
    dist.barrier()
    if rank == 0:
        results = {}
        for r in range(world_size):
            obj = tools_json_load(f"{save_dir}/{r}.temp")
            for i in range(len(obj['ids'])):
                if obj['ids'][i] not in results:
                    results[obj['ids'][i]] = 1 if obj['predicts'][i] == obj['labels'][i] else 0
            os.remove(f"{save_dir}/{r}.temp")
        acc = sum(results.values()) / len(results)
        acc = torch.tensor(acc, dtype=torch.float32, device=torch.device(rank))
    else:
        acc = torch.tensor(0.0, dtype=torch.float32, device=torch.device(rank))
    dist.barrier()
    dist.broadcast(acc, src=0)

    loss_mean = torch.tensor(loss_mean / len(valid_loader), dtype=torch.float, device=rank)
    dist.all_reduce(loss_mean, op=dist.ReduceOp.SUM)
    loss_mean = loss_mean.item() / world_size
    metric.add_record('valid_loss', epoch, loss_mean)
    metric.add_record('valid_acc', epoch, acc.item())

    return loss_mean, acc



if __name__ == '__main__':
    args, save_dir, config, gpu_num, model_config = prepare_before_train(args)
    mp.spawn(run, args=(gpu_num, args, tools_get_random_available_port(), save_dir, config, model_config), nprocs=gpu_num, join=True)
    pass
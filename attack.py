from tools import *
from utils import *
from config import *
from argparse import ArgumentParser
from data import ClassificationDataset
from defense import RockSolidDefender
from victims import build_model
import OpenAttack
import tensorflow as tf


parser = ArgumentParser()
parser.add_argument('--dataset', choices=['imdb', 'agnews', 'sst2'], default='imdb')
parser.add_argument('--victim', choices=['lstm', 'bilstm', 'bert', 'wordcnn'], default='bert')
parser.add_argument('--attack', choices=['pwws', 'textfooler', 'bertattack'], default='textfooler')
parser.add_argument('--device', default=None, type=str)
parser.add_argument('--load_path', default=None)
parser.add_argument('--using_bert_vocab', type=lambda x: 'y' in x.lower(), default='no')
parser.add_argument('--eval_path', default='attack', help='choose from [train, valid, test, attack or a json file path]')
parser.add_argument('--note', default=None, type=str)
parser.add_argument('--using_rmlm', type=lambda x: 'y' in x.lower(), default='no')
args = parser.parse_args()

if 'bert' in args.victim: assert args.using_bert_vocab
if args.using_rmlm: assert args.using_bert_vocab
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
logger = tools_get_logger('attack')
using_tf = args.attack in {'textfooler'}
tools_setup_seed(0)

if using_tf:
    if args.device is None:
        torch_gpu, attack_gpu = tools_auto_specify_gpu(2)
    else:
        attack_gpu = tools_auto_specify_gpu(1, exclude=args.device)
        torch_gpu = args.device
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{torch_gpu},{attack_gpu}'
    cpu_d = tf.config.list_physical_devices('CPU')
    gpu_d = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpu_d[-1], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8*1024)])
    tf_all = [cpu_d[0], gpu_d[-1]]
    tf.config.set_visible_devices(tf_all)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device}'
args.device = 0  # for torch




if __name__ == '__main__':
    config:config_dataset = {
        'imdb': config_victim_imdb(using_bert_vocab=args.using_bert_vocab),
        'agnews': config_victim_agnews(using_bert_vocab=args.using_bert_vocab),
        'sst2': config_victim_sst2(using_bert_vocab=args.using_bert_vocab),
    }[args.dataset]
    if args.eval_path in {'train', 'valid', 'test', 'attack'}:
        attack_path = eval(f'config.{args.eval_path}_path')
    else:
        attack_path = args.eval_path
    if args.load_path:
        model_load_path = args.load_path
    elif args.using_rmlm:
        model_load_path, infer_config = config_rmlm_best_loads().get_load_path(args.dataset, args.victim)
        config.rmlm_config = infer_config
    else:
        model_load_path = config.get_load_path(args.victim)
    model_config = config.get_model_config(args.victim)
    if args.using_bert_vocab:
        # bert vocabulary
        if args.using_rmlm: prefix = f'{args.victim}_rmlm'
        else: prefix = f'{args.victim}_bert_vocab'
    else:
        # glove
        prefix = args.victim

    if args.note and len(args.note) > 0:
        output_path = f"checkpoint/{args.dataset}/attack/{prefix}.{args.attack}.{tools_get_time()}.{args.note}.json"
    else:
        output_path = f"checkpoint/{args.dataset}/attack/{prefix}.{args.attack}.{tools_get_time()}.json"

    if args.using_rmlm:
        model_config['pretrained_wv_path'] = None
        if args.victim == 'wordcnn':
            model_config['mode'] = None
            logger.info('you are using bert_vocab, so the mode of wordcnn will reset to None')

    logger.info(args.__dict__)
    logger.info(model_config)

    train_dataset = ClassificationDataset(config.train_path, config, args.victim, model_config, vocab=None)
    attack_dataset = ClassificationDataset(attack_path, config, args.victim, model_config, vocab=train_dataset.vocab)

    if args.using_rmlm:
        logger.info(f"using rmlm+{args.victim} load from {model_load_path}\n"
                    f"mlm config {config.rmlm_config}")
        model = RockSolidDefender(config, args.victim, model_config, None, args.device, train_dataset.tokenizer, config.rmlm_config, using_detection=True)
        model.load_state_dict(torch.load(model_load_path, map_location=torch.device(0)))
    else:
        model = build_model(args.victim, model_config, config, load_path=model_load_path, rank=args.device)

    victim = OAClassifier(model, train_dataset, rank=args.device, using_rmlm=args.using_rmlm, dataset_name=args.dataset, model_name=args.victim)
    correct_attack_dataset, x_map_prob, len_attack_set = process_correct_classified(attack_dataset, victim, logger, num=None,)

    TOKEN_UNK = '[UNK]'
    if args.attack == 'pwws':
        attacker = OpenAttack.attackers.PWWSAttacker(token_unk=TOKEN_UNK)
    elif args.attack == 'textfooler':
        attacker = OpenAttack.attackers.TextFoolerAttacker(token_unk=TOKEN_UNK)
    elif args.attack == 'bertattack':
        attacker = OpenAttack.attackers.BERTAttacker(mlm_path='bert-base-uncased', device=torch.device(0), max_length=config.maxlen)
    else:
        raise NotImplementedError(args.attack)

    attack_eval = OpenAttack.AttackEval(args.attack, attacker, victim, metrics=[OpenAttack.metric.Modification(None)])
    temp, attack_success_results = attack_eval.eval(correct_attack_dataset, visualize=True, progress_bar=True, x_map_prob=x_map_prob, total_attack_set_num_include_misclsf=len_attack_set, tqdm_prefix=f'attack-{args.victim}-{args.attack}-{args.dataset}')
    summary = {
        'dataset': args.dataset,
        'victim': args.victim,
        'load_path': model_load_path,
        'attacker': args.attack,
        'dataset_path': attack_path,
        'dataset_num': len_attack_set,
        'attack_num': temp['Total Attacked Instances'],
        'oom_fail_num': temp['oom_fail'],
        'attack_success_num': temp['Successful Instances'],
        'attack_success_rate': temp['Successful Instances'] / temp['Total Attacked Instances'],
        'ori_acc': len(correct_attack_dataset) / len_attack_set,
        'adv_acc': (len(correct_attack_dataset) - temp['Successful Instances']) / len_attack_set,
        'acc_shift': temp['Successful Instances'] / len_attack_set,
        'avg_adv_time': temp['Avg. Running Time'],
        'query_exceed_num': temp['Total Query Exceeded'],
        'avg_query': temp['Avg. Victim Model Queries'],
        'avg_word_modif_rate': temp['Avg. Word Modif. Rate'],
    }
    logger.info(summary)
    save_results = {
        'args': args.__dict__,
        'victim_config': model_config,
        'rmlm_config_infer': config.rmlm_config,
        'summary': summary,
        'attack_success_example': attack_success_results
    }

    if not os.path.exists(f"checkpoint/{args.dataset}/attack"):
        os.makedirs(f"checkpoint/{args.dataset}/attack", exist_ok=True)
    logger.info(f'done! detailed result is in {output_path}')
    tools_json_dump(save_results, output_path)
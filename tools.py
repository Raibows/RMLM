import json
import logging
import os
import shutil
from datetime import datetime
from config import *
from gpustat import new_query

logging.basicConfig(level=logging.INFO, datefmt="%m/%d/%Y %H:%M:%S", format='%(asctime)s - %(levelname)s - %(name)s\n%(message)s')

def prepare_before_train(args):
    logger = tools_get_logger('prepare')

    if 'bert' in args.model.lower():
        args.using_bert_vocab = True

    if args.using_rmlm:
        args.using_bert_vocab = True

    if args.only_eval:
        args.save_type = 'evaluate'
        assert args.eval_path is not None, "you are in evaluate mode, please specify the arg '--eval_path' dataset type from [train, valid, test, attack or a json file path]"
        assert args.load is not None, "you are in evaluate mode, specify the model load path"

    if args.using_rmlm:
        assert args.rmlm_mode and args.rmlm_update and args.rmlm_maskop is not None, "you are using rmlm, need to explicitly specify the mode and grad update strategy"

    config = {
        'imdb': config_victim_imdb(using_bert_vocab=args.using_bert_vocab),
        'agnews': config_victim_agnews(using_bert_vocab=args.using_bert_vocab),
        'sst2': config_victim_sst2(using_bert_vocab=args.using_bert_vocab),
    }[args.dataset]
    dir_model = f"{args.model}_bert_vocab" if (args.using_bert_vocab and 'bert' not in args.model.lower()) else args.model
    if args.using_rmlm:
        if args.rmlm_maskop != 'rmlm': args.rmlm_max_syn = 0
        dir_model = f"{args.model}_rmlm_mode{args.rmlm_mode}_update{args.rmlm_update}_maskop{args.rmlm_maskop}_rate{args.rmlm_mask_rate}_syn{args.rmlm_max_syn}"
        config.rmlm_config = {
            'mode': args.rmlm_mode,
            'update': args.rmlm_update,
            'maskop': args.rmlm_maskop,
            'rate': args.rmlm_mask_rate,
            'syn': args.rmlm_max_syn,
            'using_for': 'train',
        }

    if args.note and len(args.note) > 0:
        save_dir = f"./checkpoint/{args.dataset}/{args.save_type}/{dir_model}/{tools_get_time()}_{args.note}"
    else:
        save_dir = f"./checkpoint/{args.dataset}/{args.save_type}/{dir_model}/{tools_get_time()}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/codes", exist_ok=True)
    tools_copy_all_suffix_files(target_dir=f"{save_dir}/codes", source_dir='.', suffix=['.py', '.sh'])

    model_config = config.get_model_config(args.model)
    if config.using_bert_vocab:
        if args.model == 'wordcnn':
            model_config['mode'] = None
            logger.info('you are using bert_vocab, so the mode of wordcnn will reset to None')
        model_config['pretrained_wv_path'] = None
        logger.info(f"you are using bert_vocab, so the pretrained glove will not load")

    hyper_params = {'train': args.__dict__, 'model': model_config, 'rmlm': config.rmlm_config}
    tools_json_dump(hyper_params, f"{save_dir}/args.json")
    gpu_num = len(args.device.split(','))

    return args, save_dir, config, gpu_num, model_config

def tools_auto_specify_gpu(num, exclude=None):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if exclude is None: exclude = set()
    if isinstance(exclude, int): exclude = str(exclude)
    if isinstance(exclude, str): exclude = exclude.strip().split(',')
    if not isinstance(exclude, set): exclude = set(exclude)
    status = new_query().jsonify()['gpus']
    assert len(status) >= num + len(exclude)
    status = sorted(status, key=lambda item: item['memory.used'])
    set_device = []
    for item in status:
        if str(item['index']) not in exclude:
            set_device.append(item['index'])
            if len(set_device) == num: break
    return set_device if len(set_device) > 1 else set_device[0]

def tools_get_logger(name):
    return logging.getLogger(name)

def tools_to_gpu(device, *args):
    return [a.to(device) for a in args]

def tools_json_load(path):
    with open(path, 'r') as file:
        return json.load(file)

def tools_json_dump(obj, path):
    with open(path, 'w') as file:
        json.dump(obj, file, indent=4)

def tools_setup_seed(seed):
    import torch
    import numpy as np
    import random
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def tools_copy_all_suffix_files(target_dir, source_dir='.', suffix=['.py', '.sh']):
    os.makedirs(target_dir, exist_ok=True)
    src_files = os.listdir(source_dir)
    for file in src_files:
        for s in suffix:
            if file.endswith(s):
                shutil.copy(f'{source_dir}/{file}', f'{target_dir}/{file}')
                break

def tools_get_random_available_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def tools_get_time():
    return datetime.now().strftime("%y-%m-%d-%H_%M_%S")


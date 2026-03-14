import argparse
import json
import os
import sys
import torch


def get_args_to_config(base_dir, parser, is_evaluation: bool = False):
    try:
        add_other_args(parser)
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
        dict_args = vars(args)
        args_to_config = {key: val for key, val in dict_args.items() if val is not None}
    except Exception as e:
        print(e)
        # parser.print_help()
        sys.exit()

    config = ConfigParser(base_dir, args_to_config)
    return config.config

def load_TTE_best_configs(args: argparse.Namespace):
    # model specific settings
    if args.model_name == '':
        pass
    pass

def add_other_args(parser):
    data = other_arguments
    for arg in data:
        if data[arg]['type'] == 'int':
            parser.add_argument('--{}'.format(arg), type=int,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'bool':
            parser.add_argument('--{}'.format(arg), type=str2bool,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'float':
            parser.add_argument('--{}'.format(arg), type=str2float,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'str':
            parser.add_argument('--{}'.format(arg), type=str,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'list of int':
            parser.add_argument('--{}'.format(arg), nargs='+', type=int,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'list of str':
            parser.add_argument('--{}'.format(arg), nargs='+', type=str,
                                default=data[arg]['default'], help=data[arg]['help'])

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')

def str2float(s):
    if isinstance(s, float):
        return s
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError('float value expected.')
    return x

class ConfigParser(object):

    def __init__(self,base_dir, args_to_config=None):
        self.config = {}
        self._parse_basic_config(base_dir, args_to_config)
        self._load_config()

    def _parse_basic_config(self,base_dir, args_to_config=None):
        self.config['base_dir'] = base_dir
        if args_to_config is not None:
            for key in args_to_config:
                self.config[key] = args_to_config[key]
            # add the model to the relative task
            if self.config['model'] == 'MLP_REG':
                self.config['task'] = 'reg_finetune'
            elif self.config['model'] == 'MLP_TTE':
                self.config['task'] = 'tte_finetune'
            elif self.config['model'] == 'MLP_PR':
                self.config['task'] = 'pr_finetune'
            elif self.config['model'] == 'MLP_CLS':
                self.config['task'] = 'cls_finetune'
            elif self.config['model'] == 'MultiTaskModel':
                self.config['task'] = 'mt_finetune'

    def _parse_config_file(self, config_file):
        if config_file is not None:
            if os.path.exists('./{}.json'.format(config_file)):
                with open('./{}.json'.format(config_file), 'r') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]
            else:
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))

    def _load_config(self):
        config_dir = os.path.join(self.config['base_dir'],'config')

        with open(f'{config_dir}/configs.json', 'r') as f:
            task_config = json.load(f)
            task_config = task_config[self.config['task']]
            model = self.config['model']
            self.config['dataset_config'] = f'dataset/' + self.config['dataset'] + '/' + task_config[model]['dataset_config']
            self.config['executor_config'] = f'executor/' + task_config[model]['executor_config']
            self.config['model_config'] = f'model/' + task_config[model]['model_config']

            if self.config['peft'] is not '':
                self.config['adapter_config'] = f'adapter/' + self.config['peft']

                with open(f"{config_dir}/{self.config['adapter_config']}.json", 'r') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]

        for file_name in [self.config['dataset_config'],
                          self.config['executor_config'], self.config['model_config']]:
            with open(f'{config_dir}/{file_name}.json', 'r') as f:
                x = json.load(f)
                for key in x:
                    if key not in self.config:
                        self.config[key] = x[key]

    def get(self, key, default=None):
        return self.config.get(key, default)

    def convert_all_config_to_strs(self):
        strs = str()
        for idx, (key, value) in enumerate(self.config.items()):
            strs = strs + str(f"{key}: {value}; ")
            if idx != 0 and idx % 4 == 0:
                strs = strs + '\n'
        return strs


    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    def __iter__(self):
        return self.config.__iter__()


other_arguments = {
    "seed": {
        "type": "int",
        "default": None,
        "help": "random seed"
    },
    "batch_size": {
        "type": "int",
        "default": None,
        "help": "the batch size"
    },
    "learning_rate": {
        "type": "float",
        "default": None,
        "help": "learning rate"
    },
    "num_workers": {
        "type": "int",
        "default": None,
        "help": "num_workers for dataloader"
    },
    "collate_func":{
        "type": "str",
        "default": None,
        "help": "choose the collate function"
    },
    "backbone": {
        "type": "str",
        "default": None,
        "help": "specify the pretrained model as backbone"
    },
    "pretrained_path": {
        "type": "str",
        "default": None,
        "help": "specify the pretrained representation model's path, such as 'repository\saved_model\.。。\example.pkL'"
    },
    "test_model_load_dict": {
        "type": "str",
        "default": None,
        "help": "specify the evaluated model, default is None, which means loading the best model in the training process. If specify the path, such as 'repository\saved_model\.。。\example.pkL', will load the model in this path to evaluate."
    },
    "freeze": {
        "type": "bool",
        "default": None,
        "help": "choose if freezing the pretrained model"
    },
    "expert_num": {
        "type": "int",
        "default": None,
        "help": "expert_num for adapter_moe"
    },
    "num_top_k_expert": {
        "type": "int",
        "default": None,
        "help": "num_top_k_expert for adapter_moe"
    },
    "reduction_factor": {
        "type": "int",
        "default": None,
        "help": "reduction_factor for adapter_moe"
    },
    "num_runs": {
        "type": "int",
        "default": None,
        "help": "run times for model"
    },
    "patience":{
        "type": "int",
        "default": None,
        "help": "patience epoch for early stop"
    },
    "num_epochs": {
        "type": "int",
        "default": None,
        "help": "max epochs"
    },
    "test_interval_epochs": {
        "type": "int",
        "default": None,
        "help": "test after num of epochs"
    },
    "mlp_only_train": {
        "type": "bool",
        "default": None,
        "help": "choose if proccess mlp in the finetune model"
    }
}

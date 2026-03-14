import argparse
import os
import sys
import warnings
import wandb
import time
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_API_KEY'] = 'YOUR_API_KEY'

from utils.load_config import get_args_to_config
from train_model import train_model
from evaluate_model import test_model

# Press the green button in the gutter to run the script.1
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # arguments
    parser = argparse.ArgumentParser('Interface for the task')
    parser.add_argument('--model', type=str, default='MLP_TTE', choices=['MLP_PR','MLP_TTE','MLP_CLS','MLP_REG', 'MultiTaskModel'])
    parser.add_argument('--peft', type=str, default='adapter_p_moe_down', choices=['ft_all', 'ft_head', 'adapter_p_moe_down'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'resume', 'test'])
    parser.add_argument('--dataset', type=str, default='porto',choices=['chengdu', 'porto'])
    parser.add_argument('--identify', type=str, default='test')
    parser.add_argument('--gpu', type=int, default=0)

    # get arguments
    config = get_args_to_config(base_dir = os.path.dirname(os.path.abspath(__file__)), parser=parser, is_evaluation=False)

    print(f"{config['mode']} {config['peft']} load {config['model']} on {config['dataset']}. identify: {config['identify']}")

    # tell wandb to get started
    with wandb.init(project=f"{config['peft']}_{config['model']}_{config['dataset']}_{config['identify']}_{config['mode']}",name=f"{time.time()}", config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        if 'finetune' in config['task']:
            if config['mode'] == 'test':
                test_model(config)
            else:
                train_model(config)
    wandb.finish()
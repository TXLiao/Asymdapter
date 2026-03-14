from importlib import import_module

import torch
from torch import nn
import os
from models.SimPRL import TimeEncoder, TimeTupleEncoder
import loralib as lora


class MLP_PR(nn.Module):
    def __init__(self, config):
        super(MLP_PR, self).__init__()
        self.output_dim = config['output_dim']
        self.device = config['device']
        self.freeze = config['freeze']
        self.beackbone_out_dim =  config['beackbone_out_dim']

        config['pretrained_path'] = f"repository/saved_model/SimPRL/{config['dataset']}/SimPRL_rrl_pretrain_{config['dataset']}/SimPRL_rrl_pretrain_{config['dataset']}_seed0.pkl"

        self.pretrained_path = os.path.join(config['base_dir'], config['pretrained_path'])

        self.backbones = getattr(import_module(f"models.{config['backbone']}"),config['backbone'])(config)
        if config['mode'] != 'test':
            checkpoints = torch.load(self.pretrained_path, map_location=self.device)
            if 'model_state_dict' in checkpoints.keys():
                missing_keys, _ = self.backbones.load_state_dict(checkpoints['model_state_dict'], strict=False)
            else:
                missing_keys , _ = self.backbones.load_state_dict(checkpoints, strict=False)
            print(missing_keys)
            print(f"{config['backbone']}, {self.pretrained_path} backbones.load_state_dict load success!")

        self.segs_extract = nn.Linear(in_features=self.beackbone_out_dim, out_features=self.output_dim * 2)
        self.hid2out = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim // 2),
            nn.ReLU(),
            nn.Linear(self.output_dim // 2, self.output_dim // 8),
            nn.ReLU(),
            nn.Linear(self.output_dim // 8, 1)
        )
        self.dense = nn.Linear(in_features=self.beackbone_out_dim, out_features=self.beackbone_out_dim)
        self.head = nn.ModuleList([self.segs_extract, self.hid2out,  self.dense])

    def forward(self, features, config, data_feature):
        scaler = data_feature['sim_standard']

        outputs = self.backbones(features, config)
        # adapter_moe
        gate_weights_loss = outputs[-1]
        outputs = outputs[0]
        outputs = self.head[2](outputs)
    

        outputs = self.head[0](outputs)

        outputs = self.head[1](outputs)
        # adapter_moe
        return outputs, gate_weights_loss



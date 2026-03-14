from importlib import import_module

import torch
from torch import nn
import os
from models.SimPRL import TimeEncoder, TimeTupleEncoder
import loralib as lora

class MultiTaskModel(nn.Module):
    def __init__(self, config):
        super(MultiTaskModel, self).__init__()
        self.output_dim = config['output_dim']
        self.device = config['device']
        self.freeze = config['freeze']
        self.time_dim = config['time_dim']
        self.beackbone_out_dim = config['beackbone_out_dim']
        config['pretrained_path'] = f"repository/saved_model/SimPRL/{config['dataset']}/SimPRL_rrl_pretrain_{config['dataset']}/SimPRL_rrl_pretrain_{config['dataset']}_seed0.pkl"

        self.pretrained_path = os.path.join(config['base_dir'], config['pretrained_path'])

        self.backbones = getattr(import_module(f"models.{config['backbone']}"),config['backbone'])(config)
        if config['mode'] is not 'test':
            checkpoints = torch.load(self.pretrained_path, map_location=self.device)
            if 'model_state_dict' in checkpoints.keys():
                missing_keys, _ = self.backbones.load_state_dict(checkpoints['model_state_dict'], strict=False)
            else:
                missing_keys , _ = self.backbones.load_state_dict(checkpoints, strict=False)

            print(missing_keys)
            print(f"{config['backbone']}, {self.pretrained_path} backbones.load_state_dict load success!")
        # MLP_TTE
        if config['backbone'] == 'SimPRL' and config['mlp_only_train']:
            self.tte_head = nn.ModuleList([
                TimeTupleEncoder(time_dim=self.time_dim),
                nn.Linear(in_features=1216, out_features=self.output_dim * 2),
                nn.Sequential(
                    nn.Linear(self.output_dim * 2, self.output_dim // 2),
                    nn.LeakyReLU(),
                    nn.Linear(self.output_dim // 2, self.output_dim // 8),
                    nn.LeakyReLU(),
                    nn.Linear(self.output_dim // 8, 1)
                ),
                nn.Linear(config['hidden_size'] + 3 * config['time_dim'], config['out_dim'])])
        else:
            self.tte_head = nn.ModuleList([
                TimeTupleEncoder(time_dim=self.time_dim),
                nn.Linear(in_features=1216, out_features=self.output_dim * 2),
                nn.Sequential(
                    nn.Linear(self.output_dim * 2, self.output_dim // 2),
                    nn.LeakyReLU(),
                    nn.Linear(self.output_dim // 2, self.output_dim // 8),
                    nn.LeakyReLU(),
                    nn.Linear(self.output_dim // 8, 1)
                )])
        # MLP_CLS
        self.num_classes = 11
        if config['backbone'] == 'SimPRL':
            self.cls_head = nn.ModuleList([
                nn.Linear(in_features=768, out_features=self.num_classes),
                nn.Linear(in_features=768, out_features=768)])
        else:
            self.cls_head = nn.ModuleList([
                nn.Linear(in_features=768, out_features=self.num_classes)])

        # MLP_PR
        if config['backbone'] == 'SimPRL':
            self.pr_head = nn.ModuleList([
                nn.Linear(in_features=768, out_features=self.output_dim * 2),
                nn.Sequential(
                    nn.Linear(self.output_dim * 2, self.output_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.output_dim // 2, self.output_dim // 8),
                    nn.ReLU(),
                    nn.Linear(self.output_dim // 8, 1)
                ),
                nn.Linear(in_features=768, out_features=768)])
        else:
            self.pr_head = nn.ModuleList([
                nn.Linear(in_features=768, out_features=self.output_dim * 2),
                nn.Sequential(
                    nn.Linear(self.output_dim * 2, self.output_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.output_dim // 2, self.output_dim // 8),
                    nn.ReLU(),
                    nn.Linear(self.output_dim // 8, 1)
                )])
        # MLP_REG
        if config['backbone'] == 'SimPRL' and config['mlp_only_train']:
            self.reg_head = nn.ModuleList([
                TimeTupleEncoder(time_dim=config['time_dim']),
                nn.Linear(in_features=1216, out_features=self.output_dim * 2),
                nn.Sequential(
                    nn.Linear(self.output_dim * 2, self.output_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.output_dim // 2, self.output_dim // 8),
                    nn.ReLU(),
                    nn.Linear(self.output_dim // 8, 1)
                ),
                nn.Linear(config['hidden_size'] + 3 * config['time_dim'], config['out_dim'])])
        else:
            self.reg_head = nn.ModuleList([
                TimeTupleEncoder(time_dim=config['time_dim']),
                nn.Linear(in_features=1216, out_features=self.output_dim * 2),
                nn.Sequential(
                    nn.Linear(self.output_dim * 2, self.output_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.output_dim // 2, self.output_dim // 8),
                    nn.ReLU(),
                    nn.Linear(self.output_dim // 8, 1)
                )
            ])

    def forward(self, features, config, data_feature):
        task_id = features['task_id'][0][0]

        # SimPRL
        outputs = self.backbones(features, config)
        gate_weights_loss = torch.tensor(0.0, device=self.device)
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            last_item = outputs[-1]
            if torch.is_tensor(last_item) and last_item.dim() == 0:
                gate_weights_loss = last_item
                outputs = outputs[:-1]

        if task_id == 1: # tte
            outputs = outputs[0]
            if config['mlp_only_train']:
                outputs = self.tte_head[-1](outputs)
            outputs = self.tte_head[1](torch.cat([outputs,self.tte_head[0](features['departure_timestamp'])],dim=-1))
            outputs = self.tte_head[2](outputs)
            outputs = data_feature['time_standard'].inverse_transform(outputs).reshape(-1)
        elif task_id == 2: # cls
            outputs = outputs[1]
            outputs = self.cls_head[1](outputs)
            outputs = self.cls_head[0](outputs).reshape(-1, self.num_classes)
        elif task_id == 3: # pr
            outputs = outputs[0]
            outputs = self.pr_head[2](outputs)
            outputs = self.pr_head[0](outputs)
            outputs = self.pr_head[1](outputs)
            outputs = data_feature['sim_standard'].inverse_transform(outputs)
        elif task_id == 4: # reg
            outputs = outputs[0]
            if config['mlp_only_train']:
                outputs = self.reg_head[-1](outputs)
            outputs = self.reg_head[1](torch.cat([outputs, self.reg_head[0](features['departure_timestamp'])], dim=-1))
            outputs = self.reg_head[2](outputs)
            outputs = data_feature['spd_standard'].inverse_transform(outputs)

        return outputs, gate_weights_loss


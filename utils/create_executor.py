import os
import shutil

import torch.nn.functional as F
import torch
from torch.nn import SmoothL1Loss, MSELoss, CrossEntropyLoss, NLLLoss, LogSoftmax, HuberLoss
from torch.optim import Adam, SGD, AdamW
from importlib import import_module
import loralib as lora


def create_loss(config):
    loss_name = config['loss_name']
    if loss_name == 'rmse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            rmse = torch.sqrt(torch.mean(torch.pow(preds - labels, 2)))
            return rmse
    elif loss_name == 'mse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            # mse = torch.mean(torch.pow(preds - labels, 2))
            mse = MSELoss(reduction='mean').forward(preds.view(-1), labels)
            return mse
    elif loss_name == 'mape':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mape = torch.mean(torch.abs(preds - labels) / (labels + 0.1))
            return mape
    elif loss_name == 'mae':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mape = torch.mean(torch.abs(preds - labels))
            return mape
    elif loss_name == 'huberloss':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            loaavaL = HuberLoss(reduction='mean', delta = config['loss_beta']).forward(preds, labels)
            return loaavaL
    elif loss_name == 'smoothL1':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            smoothL1 = SmoothL1Loss(reduction='mean', beta = config['loss_beta']).forward(preds, labels)
            return smoothL1
    elif loss_name == 'CrossEntropyLoss':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth'].reshape(-1)
            return CrossEntropyLoss().forward(preds, labels)
    elif loss_name == 'NLLLoss':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth'].reshape(-1)
            preds = F.log_softmax(preds, dim=-1)
            return NLLLoss().forward(preds, labels)
    elif loss_name == 'multi_loss_CE':
        def loss(**kwargs):
            # map -> list
            kwargs['truth'] = list(kwargs['truth'])
            preds0 = kwargs['predict'][0]
            label0 = kwargs['truth'][0].reshape(-1)
            preds1 = kwargs['predict'][1]
            label1 = kwargs['truth'][1].reshape(-1)
            preds2 = kwargs['predict'][2]
            label2 = kwargs['truth'][2].reshape(-1)
            preds3 = kwargs['predict'][3]
            label3 = kwargs['truth'][3].reshape(-1)
            loss0 = CrossEntropyLoss().forward(preds0, label0)
            loss1 = CrossEntropyLoss().forward(preds1, label1)
            loss2 = CrossEntropyLoss().forward(preds2, label2)
            loss3 = CrossEntropyLoss().forward(preds3, label3)
            # losses = config['lr_bata_0'] * loss0 + config['lr_bata_1'] * loss1 / (loss1 / loss0 + 1e-5).detach() + config['lr_bata_3'] * loss3 / (loss3 / loss0 + 1e-5).detach()
            losses = config['lr_bata_0'] * loss0 + config['lr_bata_1'] * loss1 / (loss1 / loss0 + 1e-5).detach() + config['lr_bata_2'] * loss2 / (loss2 / loss0 + 1e-5).detach() + config['lr_bata_3'] * loss3 / (loss3 / loss0 + 1e-5).detach()
            return losses
    elif loss_name == '2_loss_CE':
        def loss(**kwargs):
            # map -> list
            kwargs['truth'] = list(kwargs['truth'])
            preds0 = kwargs['predict'][0]
            label0 = kwargs['truth'][0].reshape(-1)
            preds1 = kwargs['predict'][1]
            label1 = kwargs['truth'][1].reshape(-1)

            loss0 = CrossEntropyLoss().forward(preds0, label0)
            # m = LogSoftmax(dim=1)
            # loss1 = NLLLoss().forward(m(preds1), label1)
            loss1 = CrossEntropyLoss().forward(preds1, label1)

            # print(f'loss0: {loss0.detach()} loss1: {loss1.detach()} nums: {(loss0 / loss1 + 1e-5).detach()}')

            losses = config['lr_bata_0'] * loss0 / (loss0 / loss1 + 1e-5).detach() + config['lr_bata_1'] * loss1
            return losses
    elif loss_name == '2_loss_CE_sml1':
        def loss(**kwargs):
            # map -> list
            kwargs['truth'] = list(kwargs['truth'])
            preds0 = kwargs['predict'][0]
            label0 = kwargs['truth'][0].reshape(-1)
            preds1 = kwargs['predict'][1]
            label1 = kwargs['truth'][1].reshape(-1)
            label_index = torch.where(label1 != -100)[0]
            if len(preds1.shape) == 2:
                preds1 = torch.argmax(preds1, dim=-1).float()
            preds1 = preds1.detach().reshape(-1)
            preds1 = preds1[label_index]
            label1 = label1[label_index]

            loss0 = CrossEntropyLoss().forward(preds0, label0)
            smoothL1 = SmoothL1Loss(reduction='mean', beta=config['loss_beta']).forward(preds1, label1)
            # mae = torch.mean(torch.abs(preds1 - label1))

            losses = config['lr_bata_0'] * loss0 / (loss0 / smoothL1 + 1e-5).detach() + config['lr_bata_1'] * smoothL1
            return losses
    elif loss_name == 'MSELoss_sim':
        def loss(**kwargs):
            preds = kwargs['predict']
            sub_simi = kwargs['truth']
            pred_l1_simi = torch.cdist(preds, preds, 1)
            pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal=1) == 1]
            truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal=1) == 1]
            return MSELoss().forward(pred_l1_simi, truth_l1_simi)
    elif loss_name == 'loss_transmit':
        def loss(**kwargs):
            return kwargs['predict']
    elif loss_name == 'FocalLoss':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth'].reshape(-1)
            # Convert y_true to one-hot encoding
            num_classes = preds.size(1)
            valid_mask = (labels != -100)

            # Filter out the ignored indices in y_true and y_pred
            labels = labels[valid_mask]
            preds = preds[valid_mask]
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

            # Clip predictions to prevent log(0)
            preds = torch.clamp(preds, 1e-7, 1 - 1e-7)

            # Compute cross entropy
            cross_entropy = -labels_one_hot * torch.log(preds)

            # Compute focal loss
            pt = torch.sum(labels_one_hot * preds, dim=1)
            focal_loss = 1 * (1 - pt) ** 2  * torch.sum(cross_entropy, dim=1) # alpha用于平衡类别的不均衡，通常可以不设置（即默认为 1）。 gamma用于控制难易样本的关注度,通常设置为 2，但可以根据具体任务进行调整。

            return torch.mean(focal_loss)
    elif loss_name == 'multi_task':
        # 针对多任务场景，将 loss 随 task_id 切换
        smoothL1_tte = SmoothL1Loss(reduction='mean', beta=config.get('loss_beta'))
        mse_reg = MSELoss(reduction='mean')
        mse_pr = MSELoss(reduction='mean')
        ce_cls = CrossEntropyLoss()
        gate_alpha = config.get('gate_loss_alpha', 0.1)

        def loss(**kwargs):
            preds = kwargs['predict']
            labels_pack = kwargs['truth']
            task_id = int(labels_pack['task_id'])
            labels = labels_pack['labels']

            gate_loss = None
            if isinstance(preds, tuple):
                preds, gate_loss = preds

            if task_id == 1:  # TTE
                l = smoothL1_tte(preds.view(-1), labels.view(-1))
            elif task_id == 2:  # CLS
                l = ce_cls(preds, labels.reshape(-1))
            elif task_id == 3:  # PR
                l = mse_pr(preds.view(-1), labels.view(-1))
            elif task_id == 4:  # REG
                l = mse_reg(preds.view(-1), labels.view(-1))
            else:
                raise ValueError(f"Unknown task_id {task_id} for multi_task loss.")

            # if gate_loss is not None:
            #     l = l + gate_alpha * gate_loss
            return l


    else:
        print(NotImplementedError(f"Unknown loss function {loss_name}."))
    return loss

def create_model(config):
    # try:
    model_name = config['model']
    model_class = getattr(import_module(f"models.{model_name}"),model_name)
    return model_class(config)
    # except Exception as e:
    #     print(f"model augments initialization errors! check the model file")
    #     print(e)

def create_optim(model, config):
    optim_name = config['learner']
    if optim_name == 'Adam':
        return Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['lr_weight_decay'], eps=config['lr_eps'])
    elif optim_name == 'SGD':
        return SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['lr_weight_decay'], momentum=config['momentum'])
    elif optim_name == 'AdamW':
        return AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['lr_weight_decay'], eps=config['lr_eps'])
    else:
        print(NotImplementedError(f"Unknown optim function {optim_name}."))

def create_executor(config, seed = 0):
    start_epoch = 0
    start_run = 0
    config['save_model_name'] = f"{config['model']}_{config['task']}_{config['dataset']}_{config['identify']}"
    save_model_folder = os.path.join(config['base_dir'],'repository','saved_model', config['model'], config['dataset'], config['save_model_name'])
    config['save_model_folder'] = save_model_folder

    loss_func = create_loss(config)
    model = create_model(config).to(config['device'])
    optim = create_optim(model, config)

    if 'train' in config['mode']:
        # create model dir
        if os.path.exists(save_model_folder):
            shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

    elif 'resume' in config['mode']:
        # load model dir and load the final model
        final_model = torch.load(os.path.join(save_model_folder, 'final_model.pkl'), map_location=config['device'])
        start_epoch = final_model['epoch']
        start_run = final_model['run']
        model.load_state_dict(final_model['model_state_dict'], strict=True)
        optim.load_state_dict(final_model['optimizer_state_dict'])

    elif config['mode'] == 'test':
        if config.get('test_model_load_dict') != None:
            # print(model.state_dict().keys())
            # print(torch.load(config['test_model_load_dict'], map_location=config['device'])['model_state_dict'].keys())
            model.load_state_dict(torch.load(config['test_model_load_dict'], map_location=config['device'])['model_state_dict'], strict=True)
            print(f"{config['test_model_load_dict']} load test_model_load_dict success!")
        else:
            # load model dir and load the best model
            print(f"load {config['save_model_name']}_seed{seed} model in {save_model_folder}...")
            model.load_state_dict(torch.load(os.path.join(save_model_folder, f"{config['save_model_name']}_seed{seed}.pkl"),
                                         map_location=config['device'])['model_state_dict'],strict=True)
            print(f"{config['save_model_name']}_seed{seed} load model in {save_model_folder} success!")
    else:
        ValueError(f"create executor for {config['mode']} fail!")
    return model, loss_func, optim, start_epoch, start_run




import random
import torch
import torch.nn as nn
import numpy as np
import os
import time

from models.adapters.adapter_controller import AdapterController_MoE


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parameter_sizes(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: nn.Module
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def to_var(var, device=0):
    if torch.is_tensor(var):
        # var = Variable(var)
        if torch.cuda.is_available():
            var = var.to(device)
        return var
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key], device)
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x, device), var)
        return var


def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)


def binary2graycode(binary):
    gray = []
    gray.append(binary[0])
    for i in range(1, len(binary)):
        if binary[i - 1] == binary[i]:
            g = 0
        else:
            g = 1
        gray.append(g)
    return gray


def convert_lon2binary(lon, size, round_num):
    # 负数转换二进制后带负号，此处取绝对值，避免计算错误
    lon = abs(lon)
    lon = round(lon, round_num) * (10 ** round_num)
    bin_lon = '{0:b}'.format(int(lon)).zfill(size)  # lon_size
    bin_lon_list = [int(i) for i in bin_lon]

    # if self.configs.use_graycode:
    bin_lon_list = binary2graycode(bin_lon_list)

    assert len(bin_lon_list) == size, "ERROR"
    # return np.array(bin_lon_list,dtype=np.int8)
    return np.asarray(bin_lon_list)


def convert_lat2binary(lat, size, round_num):
    # 负数转换二进制后带负号，此处取绝对值，避免计算错误
    lat = abs(lat)
    lat = round(lat, round_num) * (10 ** round_num)
    bin_lat = '{0:b}'.format(int(lat)).zfill(size)  # lat_size
    bin_lat_list = [int(i) for i in bin_lat]

    # if self.configs.use_graycode:
    bin_lat_list = binary2graycode(bin_lat_list)

    assert len(bin_lat_list) == size, "ERROR"
    # return np.array(bin_lat_list,dtype=np.int8)
    return np.asarray(bin_lat_list)


def timestamp2timetuple(timestamp):
    timestamp = time.localtime(timestamp)

    return [timestamp.tm_wday, timestamp.tm_yday, timestamp.tm_hour * 60 + timestamp.tm_min]


def classify_time(time_info):
    # input: [timestamp.tm_wday,timestamp.tm_yday, timestamp.tm_hour * 60 + timestamp.tm_min]
    tm_wday, _, total_minutes = time_info

    # 判断是否是工作日
    is_weekday = 0 <= tm_wday <= 4

    # 判断是否是 Morning peak
    if is_weekday and 420 <= total_minutes < 540:
        return 0  # Morning peak

    # 判断是否是 Afternoon peak
    if is_weekday and 960 <= total_minutes < 1140:
        return 1  # Afternoon peak

    # 其他时间都是 Off-peak
    return 2  # Off-peak


def truncated_rand(mu=0, sigma=0.1, factor=0.001, bound_lo=-0.01, bound_hi=0.01):
    # using the defaults parameters, the success rate of one-pass random number generation is ~96%
    # gauss visualization: https://www.desmos.com/calculator/jxzs8fz9qr?lang=zh-CN
    while True:
        n = random.gauss(mu, sigma) * factor
        if bound_lo <= n <= bound_hi:
            break
    return n


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # 递归展平子列表
        else:
            flat_list.append(item)  # 直接添加非列表元素
    return flat_list


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def unfreeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = True


def freeze_model_params(model, config):

    if config['freeze'] and 'adapter' in config['peft']:
        freeze_params(model.backbones)
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (AdapterController_MoE)):
                for param_name, param in sub_module.named_parameters():
                    param.requires_grad = True

        # Unfreezes layer norms.
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.LayerNorm):
                if 'adapter' not in name:  # this will not consider layer norms inside adapters then.
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
    elif config['freeze']:
        print("---------------------------------")
        print('freeze!')
        print("---------------------------------")
        freeze_params(model.backbones)

def modify_model_after_init(logger, model, config, steps_pre_epoch = 0):
    # Freezes model parameters.
    freeze_model_params(model, config)

    total_backbones_params = sum(p.numel() for p in model.backbones.parameters())
    total_backbones_trainable_params = sum(p.numel() if p.requires_grad == True else 0 for p in model.backbones.parameters())
    if 'MultiTaskModel' in config['model']:
        total_head_params = sum(p.numel() for p in model.tte_head.parameters())
        total_head_params += sum(p.numel() for p in model.cls_head.parameters())
        total_head_params += sum(p.numel() for p in model.pr_head.parameters())
        total_head_params += sum(p.numel() for p in model.reg_head.parameters())
    else:
        total_head_params = sum(p.numel() for p in model.head.parameters())
    total_trainable_params = get_parameter_sizes(model)
    logger.info("***** Model Trainable Parameters {} *****".format(total_trainable_params))

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info("##### Parameter name %s", name)

    total_params = sum(p.numel() for p in model.parameters())
    base_model_params = config['base_model_params']
    total_adapter_paras = total_params - base_model_params - total_head_params
    total_trainable_bias_params = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and n.endswith(".bias"))
    total_trainable_layernorm_params = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and ".LayerNorm.weight" in n)

    logger.info("Total trainable parameters %s", total_trainable_params)
    logger.info("Total traianable bias parameters %s", total_trainable_bias_params)
    logger.info("Total trainable layernorm parameters %s", total_trainable_layernorm_params)
    logger.info("Total adapter parameters %s", total_adapter_paras)
    logger.info("Total backbone parameters %s", total_backbones_params)
    logger.info("Total backbone trainable parameters %s", total_backbones_trainable_params)
    logger.info("Total base model parameters %s", base_model_params)
    logger.info("Total head model parameters %s", total_head_params)
    logger.info("Total parameters %s", total_params)

    total_params_ratio = total_params / base_model_params
    total_trainable_params_percent = (total_trainable_params / base_model_params) * 100
    total_trainable_bias_params_percent = (total_trainable_bias_params / total_trainable_params) * 100
    total_trainable_layernorm_params_percent = (total_trainable_layernorm_params / total_trainable_params) * 100
    total_trainable_head_params_percent = (total_head_params / base_model_params) * 100

    logger.info("For adapters/prompt-tuning, total params %s", total_params_ratio)
    logger.info("For intrinsic, total params %s", total_params / base_model_params)
    logger.info("Total trainable params %s precent", total_trainable_params_percent)
    logger.info("Total trainable bias params %s precent", total_trainable_bias_params_percent)
    logger.info("Total trainable layernorm params %s precent", total_trainable_layernorm_params_percent)
    logger.info("Total head params %s precent", total_trainable_head_params_percent)

    if 'dolora' in config['peft']:
        model = Adaptive_Lora_Model_Wrapper(config, model, max_step=steps_pre_epoch * config['num_epochs'] * config['num_runs'])

    return model
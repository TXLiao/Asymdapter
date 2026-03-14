import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, r2_score
from scipy.stats import kendalltau, spearmanr
import pandas as pd
import torch.nn.functional as F


def calculate_regression_metrics(preds: torch.Tensor, labels: torch.Tensor):
    # preds and labels must be torch.tensor
    try:
        preds = preds.cpu().detach().reshape(-1)
        labels = labels.reshape(-1)
        mape = torch.mean(torch.abs(torch.divide(torch.sub(preds, labels), labels + 1e-5)))
        mse = torch.mean(torch.square(torch.sub(preds, labels)))
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(torch.sub(preds, labels)))
        pearsonrs = pearsonr(preds, labels)
        r2_scores = r2_score(preds, labels)

    except Exception as e:
        print(e)
        mae = 0
        mape = 0
        rmse = 0
        pearsonrs = (None, None)
        r2_scores = 0

    return {'MAE': float(mae), 'MAPE': float(mape), 'RMSE': float(rmse), 'PEARR':pearsonrs[0], 'PEARP': pearsonrs[1], 'r2_score':float(r2_scores)}

def calculate_multi_classification_acc_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: lists of tensor, shape [(num_samples, classes),...]
    :param labels: lists of tensor, shape [(num_samples, classes),...]
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    metrics = {}

    # label = labels[0]
    # pred = preds[0]

    labels = labels.reshape(-1)
    label_index = torch.where(labels != -100)[0]

    preds = preds.cpu().detach()
    preds = preds[label_index]
    labels = labels[label_index]

    if len(preds.shape) == 2:
        preds = torch.argmax(preds, dim=-1).reshape(-1)
    metrics[f'accuracy_scores'] = accuracy_score(y_true=labels, y_pred=preds)
    return metrics

def calculate_multi_classification_sum_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: lists of tensor, shape [(num_samples, classes),...]
    :param labels: lists of tensor, shape [(num_samples, classes),...]
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    metrics = {}

    # label = labels[0]
    # pred = preds[0]

    # labels = labels.reshape(-1)
    # label_index = torch.where(labels != -100)[0]
    #
    # preds = preds.cpu().detach()
    # preds = preds[label_index]
    # labels = labels[label_index]

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = F.softmax(preds, dim=-1)
    metrics['auc_ovo'] = roc_auc_score(labels, preds, multi_class='ovo')
    metrics['auc_ovr'] = roc_auc_score(labels, preds, multi_class='ovr')

    if len(preds.shape) == 2:
        preds = torch.argmax(preds, dim=-1).reshape(-1).float()
    metrics[f'f1_scores_macro'] = f1_score(y_true=labels, y_pred=preds, average='macro')
    metrics[f'f1_scores_micro'] = f1_score(y_true=labels, y_pred=preds, average='micro')
    metrics[f'balanced_accuracy'] = balanced_accuracy_score(y_true=labels, y_pred=preds)
    metrics[f'accuracy_scores'] = accuracy_score(y_true=labels, y_pred=preds)
    return metrics

def calculate_path_rank_pre_metrics(preds: torch.Tensor, labels: torch.Tensor):
    # preds and labels must be torch.tensor

    preds = preds.cpu().detach().reshape(-1)
    labels = labels.reshape(-1)

    mae = torch.mean(torch.abs(torch.sub(preds, labels)))
    mare = mae / torch.mean(torch.abs(labels))

    return {'MAE': float(mae), 'MARE': float(mare)}

def calculate_path_rank_sum_metrics(preds: torch.Tensor, labels: torch.Tensor, idxs=None):
    # preds and labels must be torch.tensor
    preds = torch.cat(preds, dim=0).reshape(-1)
    labels = torch.cat(labels, dim=0)
    idxs = torch.cat(idxs, dim=0).cpu()

    mae = torch.mean(torch.abs(torch.sub(preds, labels)))
    mare = mae / torch.mean(torch.abs(labels))

    # 创建 DataFrame
    df = pd.DataFrame(np.asarray(torch.stack([idxs, labels, preds], dim=1)), columns=['idx', 'ground_truth', 'predict'])

    # 按 idx 分组并计算相关系数
    kendall_tau_values = []
    spearman_corr_values = []

    for idx, group in df.groupby('idx'):
        tau, _ = kendalltau(group['ground_truth'], group['predict'])
        rho, _ = spearmanr(group['ground_truth'], group['predict'])

        kendall_tau_values.append(tau)
        spearman_corr_values.append(rho)

    # 计算均值
    mean_kendall_tau = pd.Series(kendall_tau_values).mean()
    mean_spearman_corr = pd.Series(spearman_corr_values).mean()

    return {'MAE': float(mae), 'MARE': float(mare), 'kendall_tau': float(mean_kendall_tau),'spearman_corr': float(mean_spearman_corr)}

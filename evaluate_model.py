import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import time
import os
import json
from tqdm import tqdm
import numpy as np
from importlib import import_module
from collections import defaultdict
from utils.DataLoader import get_feature_data, get_dataset
from utils.create_executor import create_executor
from utils.utils import get_parameter_sizes, to_var
from utils.utils import set_random_seed, flatten_list


def evaluate_multitask_model(model: nn.Module, evaluate_data_loader: DataLoader, criterion: nn.Module, config, data_feature, mode='val'):
    model.eval()
    loss_records = defaultdict(list)
    preds_buffer = defaultdict(list)
    labels_buffer = defaultdict(list)
    idx_buffer = defaultdict(list)
    with torch.no_grad():
        evaluate_data_loader_tqdm = tqdm(evaluate_data_loader, ncols=120, dynamic_ncols=True)
        for batch_idx, evaluate_data in enumerate(evaluate_data_loader_tqdm):
            (features, truth_data) = evaluate_data
            features = to_var(features, config['device'])
            truth_data = to_var(truth_data, config['device'])
            task_id = int(truth_data['task_id'])

            outputs = model(features, config, data_feature)
            gate_loss = None
            if isinstance(outputs, tuple):
                outputs, gate_loss = outputs
            loss = criterion(truth={'labels': truth_data['labels'], 'task_id': task_id},
                             predict=(outputs, gate_loss) if gate_loss is not None else outputs)
            loss_records[task_id].append(loss.item())

            if task_id == 2:
                truth_data = truth_data['labels'].cpu().reshape(-1)
                label_index = torch.where(truth_data != -100)[0]
                outputs = outputs.cpu().detach()
                preds_buffer[task_id].append(outputs[label_index])
                labels_buffer[task_id].append(truth_data[label_index])
            else:
                preds_buffer[task_id].append(outputs.cpu().detach())
                labels_buffer[task_id].append(truth_data['labels'].cpu().reshape(-1))
            if task_id == 3 and 'traj_idxs' in features:
                idx_buffer[task_id].append(features['traj_idxs'].cpu())
            evaluate_data_loader_tqdm.set_description(
                f"task {task_id} | {mode} for the {batch_idx + 1}-th batch, loss: {loss.item():.4f}")

    metrics_total = {}
    for task_id, preds in preds_buffer.items():
        if len(preds) == 0:
            continue
        metrics_key = config['metrics_eval'][str(task_id)] if isinstance(config['metrics_eval'], dict) else config['metrics_eval']
        metric_fn = getattr(import_module('utils.metrics'), metrics_key)
        labels = labels_buffer[task_id]
        idxs = idx_buffer.get(task_id, [])
        if metrics_key in ['calculate_path_rank_sum_metrics']:
            metrics = metric_fn(preds=preds, labels=labels, idxs=idxs if len(idxs) > 0 else None)
        elif metrics_key in ['calculate_multi_classification_sum_metrics']:
            metrics = metric_fn(preds=preds, labels=labels)
        else:
            metrics = metric_fn(preds=torch.cat(preds, dim=0), labels=torch.cat(labels, dim=0))
        for k, v in metrics.items():
            metrics_total[f"task{task_id}_{k}"] = v

    all_losses = [ls for task_ls in loss_records.values() for ls in task_ls]
    overall_loss = float(np.mean(all_losses)) if len(all_losses) > 0 else 0.0
    metrics_total['overall_loss'] = overall_loss
    return [overall_loss], [metrics_total]


def evaluate_model(model: nn.Module, evaluate_data_loader: DataLoader, criterion: nn.Module, config, data_feature, mode = 'val'):
    if config.get('task') == 'mt_finetune':
        return evaluate_multitask_model(model, evaluate_data_loader, criterion, config, data_feature, mode)
    model.eval()
    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_data_loader_tqdm = tqdm(evaluate_data_loader, ncols=120, dynamic_ncols=True)
        truths = []
        predicts = []
        idx = []

        for batch_idx, evaluate_data in enumerate(evaluate_data_loader_tqdm):
            (features, truth_data) = evaluate_data

            features = to_var(features, config['device'])
            # adapter_moe
            outputs, gate_weights_loss = model(features, config, data_feature)
            loss = criterion(truth=to_var(truth_data, config['device']), predict=outputs) + gate_weights_loss

            evaluate_losses.append(loss.item())
            # adapter_moe
            evaluate_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item():.4f}, gate_loss: {gate_weights_loss.item() if type(gate_weights_loss) != int else 0. :.4f}')
            if config['task'] == 'reg_finetune' or config['task'] == 'tte_finetune':
                evaluate_metrics.append(getattr(import_module('utils.metrics'),
                                            config['metrics_eval'])(preds=outputs, labels=truth_data))
            elif config['task'] == 'cls_finetune':
                truth_data = truth_data.reshape(-1)
                label_index = torch.where(truth_data != -100)[0]
                outputs = outputs.cpu().detach()
                predicts.append(outputs[label_index])
                truths.append(truth_data[label_index])
            elif config['task'] == 'pr_finetune' and 'PR' in config['model']:
                truth_data = truth_data.reshape(-1)
                outputs = outputs.cpu().detach()
                predicts.append(outputs)
                truths.append(truth_data)
                idx.append(features['traj_idxs'])
            else:
                outputs = outputs.cpu().detach()
                predicts.append(outputs)
                idx.append(features['traj_idxs'])
                truths = data_feature[f'{mode}_sims']

        if config['task'] == 'cls_finetune':
            evaluate_metrics.append(getattr(import_module('utils.metrics'), config['metrics_eval'])(preds=predicts, labels=truths))
        elif config['task'] == 'pr_finetune':
            evaluate_metrics.append(getattr(import_module('utils.metrics'), config['metrics_eval'])(preds=predicts, labels=truths, idxs = idx))

    return evaluate_losses, evaluate_metrics

def test_model(config):
    # get segment features
    data_feature = get_feature_data(config)

    # get dataloader of full data
    dataloader = get_dataset(config, data_feature)

    val_metric_all_runs, test_metric_all_runs, = [], []

    for run in range(0, config['num_runs']):
        set_random_seed(seed=run)
        seed = run
        model, criterion, _, _, _ = create_executor(config, seed)
        # model.eval()

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"{config['base_dir']}/logs/{config['model']}/{config['dataset']}/{config['save_model_name']}/{seed}/",
            exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"{config['base_dir']}/logs/{config['model']}/{config['dataset']}/{config['save_model_name']}/{seed}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {json.dumps(dict(config), indent=4, ensure_ascii=False)}')

        logger.info(f'model -> {model}')
        logger.info(f"model name: {config['model']}, #parameters: {get_parameter_sizes(model) * 4} B, "
                    f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.")

        # evaluate the best model
        logger.info(f"get final performance on dataset {config['dataset']}...")

        val_losses, val_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['val'],
                                                 criterion=criterion,
                                                 config=config, data_feature=data_feature)

        test_losses, test_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['test'],
                                                   criterion=criterion,
                                                   config=config, data_feature=data_feature, mode='final_test')

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < config['num_runs'] - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in
                                 val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in
                             test_metric_dict},
        }
        result_json = json.dumps(result_json, indent=4)

        # save_result_folder = f"./saved_results/{args.model_name}_{args.identify}/{args.dataset_name}"
        save_result_folder = os.path.join(config['base_dir'], 'repository', 'saved_results', config['model'],
                                          config['dataset'])
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{config['save_model_name']}_seed{seed}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f"metrics over {config['num_runs']} runs:")

    for metric_name in val_metric_all_runs[0].keys():
        logger.info(
            f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
        logger.info(
            f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
            f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(
            f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(
            f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
            f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

if __name__ == '__main__':
    pass

import json
import logging
import os
import sys
import wandb
import time

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from utils.DataLoader import get_feature_data, get_dataset
from utils.create_executor import create_executor
from utils.utils import get_parameter_sizes, to_var
from utils.EarlyStopping import EarlyStopping
from importlib import import_module
from evaluate_model import evaluate_model, evaluate_multitask_model
from utils.utils import set_random_seed, save_model, modify_model_after_init
import copy


def compute_mt_metrics(task_id, preds_buffer, labels_buffer, idx_buffer, metrics_map):
    metrics_key = metrics_map[str(task_id)] if isinstance(metrics_map, dict) else metrics_map
    metric_fn = getattr(import_module('utils.metrics'), metrics_key)
    preds = preds_buffer.get(task_id, [])
    labels = labels_buffer.get(task_id, [])
    idxs = idx_buffer.get(task_id, [])
    if len(preds) == 0:
        return {}
    if metrics_key in ['calculate_multi_classification_sum_metrics', 'calculate_path_rank_sum_metrics']:
        return metric_fn(preds=preds, labels=labels, idxs=idxs if len(idxs) > 0 else None)
    else:
        return metric_fn(preds=torch.cat(preds, dim=0), labels=torch.cat(labels, dim=0))


def train_multitask_model(config, data_feature, model, criterion, optimizer, start_epoch, start_run):
    val_metric_all_runs, test_metric_all_runs, = [], []
    step = 0
    current_epoch = 0
    seed = 0
    # 任务动态权重、归一化与学习率缩放
    task_ids = ['1', '2', '3', '4']
    task_weights = {tid: 1.0 / len(task_ids) for tid in task_ids}
    task_loss_stat = {tid: 1.0 for tid in task_ids}
    loss_balance_temp = config.get('loss_balance_temp', 1.0)
    task_lr_scale = config.get('task_lr_scale', {tid: 1.0 for tid in task_ids})
    wandb.watch(model, criterion, log="all", log_freq=500)
    try:
        for run in range(start_run, config['num_runs']):
            set_random_seed(seed=run)
            seed = run

            # get dataloader of full data
            dataloader = get_dataset(config, data_feature)

            # set up logger
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            os.makedirs(f"{config['base_dir']}/logs/{config['model']}/{config['dataset']}/{config['save_model_name']}/{seed}/", exist_ok=True)
            fh = logging.FileHandler(
                f"{config['base_dir']}/logs/{config['model']}/{config['dataset']}/{config['save_model_name']}/{seed}/{str(time.time())}.log")
            fh.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(ch)

            if config['peft'] is not '' and run == start_run  and 'Embed' not in config['model']:
                model = modify_model_after_init(logger, model, config, len(dataloader['train']))

            run_start_time = time.time()
            logger.info(f"********** Run {run + 1} starts. **********")
            logger.info(f'configuration is {json.dumps(dict(config), indent=4, ensure_ascii=False)}')
            logger.info(f'model -> {model}')
            logger.info(f"model name: {config['model']}, #parameters: {get_parameter_sizes(model) * 4} B, "
                        f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.")

            early_stopping = EarlyStopping(patience=config['patience'], save_model_folder=config['save_model_folder'],
                                           save_model_name=config['save_model_name'], logger=logger, model_name=config['model'], seed=seed)
            for epoch in range(start_epoch, config['num_epochs']):
                model.train()
                current_epoch = epoch

                train_loss_records = defaultdict(list)
                train_norm_loss_records = defaultdict(list)
                preds_buffer = defaultdict(list)
                labels_buffer = defaultdict(list)
                idx_buffer = defaultdict(list)

                current_time = time.time()
                train_data_loader_tqdm = tqdm(dataloader['train'], ncols=120, dynamic_ncols=True)
                for batch_idx, train_data in enumerate(train_data_loader_tqdm):
                    step += 1
                    optimizer.zero_grad()
                    (features, truth_data) = train_data
                    features = to_var(features, config['device'])
                    truth_data = to_var(truth_data, config['device'])
                    task_id = int(truth_data['task_id'])
                    # topk MoE
                    outputs, gate_weights_loss = model(features, config, data_feature)
                    loss_raw = criterion(truth=to_var(truth_data, config['device']), predict=outputs)
                    # fthead
                    # outputs = model(features, config, data_feature)
                    # loss_raw = criterion(truth=to_var(truth_data, config['device']), predict=outputs)
                    # 归一化 + 动态权重 + 任务学习率缩放
                    tid = str(task_id)
                    norm = loss_raw / (task_loss_stat[tid] + 1e-6)
                    task_loss_stat[tid] = 0.99 * task_loss_stat[tid] + 0.01 * loss_raw.detach().item()
                    loss = task_weights[tid] * task_lr_scale.get(tid, 1.0) * norm + gate_weights_loss

                    loss.backward()
                    clip_grad_norm_(parameters=model.parameters(), max_norm=50, norm_type=2)
                    optimizer.step()

                    train_loss_records[task_id].append(loss_raw.item())
                    train_norm_loss_records[task_id].append(norm.detach().item())
                    preds_buffer[task_id].append(outputs.detach().cpu())
                    labels_buffer[task_id].append(truth_data['labels'].detach().cpu())
                    if task_id == 3 and 'traj_idxs' in features:
                        idx_buffer[task_id].append(features['traj_idxs'].cpu())

                    train_data_loader_tqdm.set_description(
                        f'Epoch: {epoch + 1}, task: {task_id}, batch: {batch_idx + 1}, loss: {loss.item():.4f}')
                    wandb.log({"run": run + 1, "epoch": epoch + 1, "loss": loss}, step=step)

                # 汇总训练指标
                train_metrics_total = {}
                for task in preds_buffer.keys():
                    metrics = compute_mt_metrics(task, preds_buffer, labels_buffer, idx_buffer, config['metrics_train'])
                    for k, v in metrics.items():
                        train_metrics_total[f"train_task{task}_{k}"] = v
                    train_metrics_total[f"train_task{task}_loss"] = float(np.mean(train_loss_records[task])) if len(train_loss_records[task]) > 0 else 0.0
                wandb.log(train_metrics_total, step=step + 1)
                logger.info(f'Epoch: {epoch + 1}, train duration: {(time.time() - current_time):.2f} seconds')
                for key, val in train_metrics_total.items():
                    logger.info(f"{key}: {val:.4f}")

                # 更新任务权重（动态权重平均，使用归一化后的均值）
                avg_norm = []
                for tid in task_ids:
                    vals = train_norm_loss_records[int(tid)] if int(tid) in train_norm_loss_records else []
                    avg_norm.append(np.mean(vals) if len(vals) > 0 else 0.0)
                avg_norm = np.array(avg_norm)
                exp_scores = np.exp(avg_norm / (loss_balance_temp + 1e-8))
                exp_scores = exp_scores / (np.sum(exp_scores) + 1e-8)
                for i, tid in enumerate(task_ids):
                    task_weights[tid] = float(exp_scores[i])

                # 验证
                val_losses, val_metrics = evaluate_multitask_model(model=model, evaluate_data_loader=dataloader['val'],
                                                                   criterion=criterion, config=config,
                                                                   data_feature=data_feature, mode='val')
                val_metrics_total = val_metrics[0]
                val_metrics_total['val_loss'] = val_losses[0]
                step += 1
                wandb.log(val_metrics_total, step=step)
                logger.info(f'validate loss: {val_losses[0]:.4f}')
                for key, val in val_metrics_total.items():
                    logger.info(f"validate {key}: {val:.4f}")

                # 早停仅使用总验证 loss 作为指标
                early_stop = early_stopping.step([('val_loss', val_losses[0], False)], model, (features, config, data_feature))
                if early_stop:
                    break

                save_model(f"{config['save_model_folder']}/final_model.pkl",
                           **{'model_state_dict': copy.deepcopy(model.state_dict()),
                              'epoch': current_epoch,
                              'run': seed,
                              'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())})

            start_epoch = 0
            early_stopping.load_checkpoint(model)

            # 评估最佳模型
            logger.info(f"get final performance on dataset {config['dataset']}...")
            val_losses, val_metrics = evaluate_multitask_model(model=model, evaluate_data_loader=dataloader['val'],
                                                               criterion=criterion, config=config,
                                                               data_feature=data_feature, mode='val')
            test_losses, test_metrics = evaluate_multitask_model(model=model, evaluate_data_loader=dataloader['test'],
                                                                 criterion=criterion, config=config,
                                                                 data_feature=data_feature, mode='test')

            val_metric_dict = val_metrics[0]
            test_metric_dict = test_metrics[0]

            logger.info(f'validate loss: {val_losses[0]:.4f}')
            for metric_name, metric_val in val_metric_dict.items():
                logger.info(f'validate {metric_name}, {metric_val:.4f}')

            logger.info(f'test loss: {test_losses[0]:.4f}')
            for metric_name, metric_val in test_metric_dict.items():
                logger.info(f'test {metric_name}, {metric_val:.4f}')

            single_run_time = time.time() - run_start_time
            logger.info(f"Run {run + 1} after {current_epoch} epochs cost {single_run_time:.2f} seconds.")

            val_metric_all_runs.append(val_metric_dict)
            test_metric_all_runs.append(test_metric_dict)

            if run < config['num_runs'] - 1:
                logger.removeHandler(fh)
                logger.removeHandler(ch)

            # 保存结果
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            }
            result_json = json.dumps(result_json, indent=4)
            save_result_folder = os.path.join(config['base_dir'], 'repository', 'saved_results', config['model'] ,config['dataset'])
            os.makedirs(save_result_folder, exist_ok=True)
            save_result_path = os.path.join(save_result_folder, f"{config['save_model_name']}_seed{seed}.json")
            with open(save_result_path, 'w') as file:
                file.write(result_json)

        # 跨 run 的平均指标
        logger.info(f"metrics over {config['num_runs']} runs:")
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(
                f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(
                f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')
        for metric_name in test_metric_all_runs[0].keys():
            logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
            logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                        f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
        wandb.finish()
    finally:
        save_model(f"{config['save_model_folder']}/final_model.pkl",
                   **{'model_state_dict': copy.deepcopy(model.state_dict()),
                      'epoch': current_epoch,
                      'run': seed,
                      'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())})


def train_model(config):
    # get segment features
    data_feature = get_feature_data(config)

    model, criterion, optimizer, start_epoch, start_run = create_executor(config)

    if config.get('task') == 'mt_finetune':
        return train_multitask_model(config, data_feature, model, criterion, optimizer, start_epoch, start_run)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=500)

    val_metric_all_runs, test_metric_all_runs,  = [], []

    dummy_input, features = None, None
    step = 0
    current_epoch = 0
    seed = 0

    wandb.log({"#parameters/n": get_parameter_sizes(model)}, step=step)
    try:
        for run in range(start_run, config['num_runs']):
            set_random_seed(seed=run)
            seed = run

            # get dataloader of full data
            dataloader = get_dataset(config, data_feature)

            # set up logger
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            os.makedirs(f"{config['base_dir']}/logs/{config['model']}/{config['dataset']}/{config['save_model_name']}/{seed}/", exist_ok=True)
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

            if config['peft'] is not '' and run == start_run  and 'Embed' not in config['model']:
                model = modify_model_after_init(logger, model, config, len(dataloader['train']))

            run_start_time = time.time()
            logger.info(f"********** Run {run + 1} starts. **********")

            logger.info(f'configuration is {json.dumps(dict(config), indent=4, ensure_ascii=False)}')

            logger.info(f'model -> {model}')

            logger.info(f"model name: {config['model']}, #parameters: {get_parameter_sizes(model) * 4} B, "
                        f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.")

            early_stopping = EarlyStopping(patience=config['patience'], save_model_folder=config['save_model_folder'],
                                           save_model_name=config['save_model_name'], logger=logger, model_name=config['model'],seed=seed)
            if "metrics_train" not in config:
                config['metrics_train'] = config['metrics_eval']
            for epoch in range(start_epoch, config['num_epochs']):
                model.train()
                current_epoch = epoch

                # store train losses and metrics
                current_time = time.time()
                train_losses, train_metrics = [], []
                train_data_loader_tqdm = tqdm(dataloader['train'], ncols=120, dynamic_ncols=True)
                for batch_idx, train_data in enumerate(train_data_loader_tqdm):
                    step+=1
                    optimizer.zero_grad()
                    (features, truth_data) = train_data
                    features = to_var(features, config['device'])
                    # Adapter_moe
                    outputs, gate_weights_loss = model(features, config, data_feature)
                    loss = criterion(truth=to_var(truth_data, config['device']), predict=outputs) + gate_weights_loss
                    
                    # outputs = model(features, config, data_feature)
                    # loss = criterion(truth=to_var(truth_data, config['device']), predict=outputs)

                    train_losses.append(loss.item())
                    train_metrics.append(getattr(import_module('utils.metrics'),
                                        config['metrics_train'])(preds=outputs, labels=truth_data))
                    loss.backward()
                    clip_grad_norm_(parameters=model.parameters(), max_norm=50, norm_type=2)
                    optimizer.step()
                    train_data_loader_tqdm.set_description(
                        # Adapter_moe
                        f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item():.4f}, gate_loss: {gate_weights_loss.item() if type(gate_weights_loss) != int else 0. :.4f}')
                    # Where the magic happens
                    wandb.log({"run": run + 1, "epoch": epoch + 1, "loss": loss}, step=step)

                val_losses, val_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['val'], criterion=criterion, config=config, data_feature= data_feature)
                wandb.log({"time_p_epoch":(time.time() - current_time)}, step=step)
                logger.info(f'Epoch: {epoch + 1}, train and val duration per epoch: {(time.time() - current_time):.2f} seconds, '
                            f'learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')

                train_metrics_total = dict()
                for metric_name in train_metrics[0].keys():
                    train_metrics_total[f'train {metric_name}'] = np.mean([train_metric[metric_name] for train_metric in train_metrics])
                    logger.info(
                        f"train {metric_name}, {train_metrics_total[f'train {metric_name}']:.4f}")
                step+=1
                wandb.log(train_metrics_total,step=step)
                logger.info(f'validate loss: {np.mean(val_losses):.4f}')
                val_metrics_total = dict()
                for metric_name in val_metrics[0].keys():
                    val_metrics_total[f'validate {metric_name}'] = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                    logger.info(
                        f"validate {metric_name}, {val_metrics_total[f'validate {metric_name}']:.4f}")
                step+=1
                val_metrics_total['validate_loss'] = np.mean(val_losses)
                wandb.log(val_metrics_total,step=step)

                # perform testing once after test_interval_epochs
                if (epoch + 1) % config['test_interval_epochs'] == 0:
                    test_losses, test_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['test'], criterion=criterion, config=config, data_feature= data_feature, mode='test')

                    logger.info(f'test loss: {np.mean(test_losses):.4f}')
                    test_metrics_total = dict()
                    for metric_name in test_metrics[0].keys():
                        test_metrics_total[f'test {metric_name}'] = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                        logger.info(
                            f"test {metric_name}, {test_metrics_total[f'test {metric_name}']:.4f}")
                    step+=1
                    wandb.log(test_metrics_total, step=step)

                # select the best model based on all the validate metrics, higher_better: True
                val_metric_indicator = []
                for metric_name in val_metrics[0].keys():
                    if ('recall' in metric_name or 'precision' in metric_name or 'acc' in metric_name or 'f1' in metric_name
                            or 'top' in metric_name or 'auc' in metric_name or 'PEARR' in metric_name or 'kendall_tau' in metric_name
                            or 'spearman_corr' in metric_name or 'r2_score' in metric_name):
                        val_metric_indicator.append(
                            (metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
                    else:
                        val_metric_indicator.append(
                            (metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), False))
                dummy_input = (features, config, data_feature)
                early_stop = early_stopping.step(val_metric_indicator, model, dummy_input)
                if early_stop:
                    break

                save_model(f"{config['save_model_folder']}/final_model.pkl",
                           **{'model_state_dict': copy.deepcopy(model.state_dict()),
                              'epoch': current_epoch,
                              'run': seed,
                              'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())})

            start_epoch = 0
            # load the best model
            early_stopping.load_checkpoint(model)

            # evaluate the best model
            logger.info(f"get final performance on dataset {config['dataset']}...")
            val_losses, val_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['val'], criterion=criterion,
                                                       config=config, data_feature= data_feature, mode='val')
            test_losses, test_metrics = evaluate_model(model=model, evaluate_data_loader=dataloader['test'], criterion=criterion,
                                                       config=config, data_feature= data_feature, mode='test')

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
            logger.info(f"Run {run + 1} after {current_epoch} epochs cost {single_run_time:.2f} seconds.")

            val_metric_all_runs.append(val_metric_dict)
            test_metric_all_runs.append(test_metric_dict)

            # avoid the overlap of logs
            if run < config['num_runs'] - 1:
                logger.removeHandler(fh)
                logger.removeHandler(ch)

            # save model result
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            }
            result_json = json.dumps(result_json, indent=4)

            # save_result_folder = f"./saved_results/{args.model_name}_{args.identify}/{args.dataset_name}"
            save_result_folder = os.path.join(config['base_dir'], 'repository', 'saved_results', config['model'] ,config['dataset'])

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
            logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
            logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                        f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
        wandb.finish()
    finally:
        save_model(f"{config['save_model_folder']}/final_model.pkl",
                   **{'model_state_dict': copy.deepcopy(model.state_dict()),
                      'epoch': current_epoch,
                      'run': seed,
                      'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())})

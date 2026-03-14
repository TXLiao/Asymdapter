import os
import pickle
import pickle as pkl
import numpy as np
import heapq
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import timestamp2timetuple
from importlib import import_module
from tqdm import tqdm
import time


class Floatvalue_StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class TrajectoryProcessingDataset(Dataset):
    def __init__(self, df_folder: str, vocab, add_cls, cache_path, masking_ratio, masking_mode, distribution, avg_mask_len, mode):
        self.data = np.load(df_folder, allow_pickle=True)
        # self.traj_idxs = data[:,0]
        self.adj_seg = self.data[:,1]
        # self.duration = data[:,3]
        self.departure_timestamp = self.data[:,4]
        self.adj_seg_lens = np.asarray([len(x) for x in self.adj_seg])
        self.idxs = np.asarray([x for x in range(len(self.adj_seg))])
        self.add_cls = add_cls
        self.vocab = vocab
        cache_path = cache_path + f"{mode}.pkl"
        if os.path.exists(cache_path):
            self.traj_list = pickle.load(open(cache_path, 'rb'))
            print('Load dataset from {}'.format(cache_path))
        else:
            self.traj_list = self.data_processing([self.adj_seg, self.departure_timestamp, self.adj_seg_lens], cache_path = cache_path)

        self.masking_ratio = masking_ratio
        self.masking_mode = masking_mode
        self.distribution = distribution
        self.avg_mask_len = avg_mask_len
        self.exclude_feats = None

    def __getitem__(self, idx):
        traj_ind = self.traj_list[idx]  # (seq_length, feat_dim)
        mask = self.noise_mask(traj_ind, self.masking_ratio, self.avg_mask_len, self.masking_mode, self.distribution,
                          self.exclude_feats, self.add_cls)  # (seq_length, feat_dim) boolean array
        return torch.LongTensor(traj_ind), torch.LongTensor(mask)

    def noise_mask(self, X, masking_ratio, lm=3, mode='together', distribution='random', exclude_feats=None, add_cls=True):
        if exclude_feats is not None:
            exclude_feats = set(exclude_feats)

        if distribution == 'geometric':  # stateful (Markov chain)
            if mode == 'separate':  # each variable (feature) is independent
                mask = np.ones(X.shape, dtype=bool)
                for m in range(X.shape[1]):  # feature dimension
                    if exclude_feats is None or m not in exclude_feats:
                        mask[:, m] = self.geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
            else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
                mask = np.tile(np.expand_dims(self.geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
        elif distribution == 'random':  # each position is independent Bernoulli with p = 1 - masking_ratio
            if mode == 'separate':
                mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                        p=(1 - masking_ratio, masking_ratio))
            else:
                mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                                p=(1 - masking_ratio, masking_ratio)), X.shape[1])
        else:
            mask = np.ones(X.shape, dtype=bool)
        if add_cls:
            mask[0] = True  # CLS at 0, set mask=1
        mask[1] = False # avoid nan when calculating CELoss
        return mask

    def geom_noise_mask_single(self, L, lm, masking_ratio):
        keep_mask = np.ones(L, dtype=bool)
        p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
        p_u = p_m * masking_ratio / (
                    1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
        p = [p_m, p_u]

        # Start in state 0 with masking_ratio probability
        state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
        for i in range(L):
            keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
            if np.random.rand() < p[state]:
                state = 1 - state

        return keep_mask

    def __len__(self):
        return len(self.traj_list)

    def data_processing(self, origin_data, desc=None, cache_path=None, tmat_path=None):
        print('Processing dataset in TrajectoryProcessingDataset!')
        traj_list = []
        for i in tqdm(range(len(origin_data[0])), desc=desc):
            loc_list = origin_data[0][i]
            new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            tim_list = origin_data[1][i]
            loc_len = origin_data[2][i]
            tim_list = [tim_list] * loc_len
            new_tim_list = [time.localtime(tim) for tim in tim_list]
            minutes = [new_tim.tm_hour * 60 + new_tim.tm_min + 1 for new_tim in new_tim_list]
            weeks = [new_tim.tm_wday + 1 for new_tim in new_tim_list]
            if self.add_cls:
                new_loc_list = [self.vocab.sos_index] + new_loc_list
                minutes = [self.vocab.pad_index] + minutes
                weeks = [self.vocab.pad_index] + weeks
                tim_list = [tim_list[0]] + tim_list
            traj_fea = np.array([new_loc_list, tim_list, minutes, weeks]).transpose((1, 0))
            traj_list.append(traj_fea)
        pickle.dump(traj_list, open(cache_path, 'wb'))
        return traj_list# , temporal_mat_list


class RegFinetuneDataset(Dataset):
    def __init__(self, df_folder: str):
        data = np.load(df_folder, allow_pickle=True)
        self.traj_idxs = data[:,0]
        self.adj_seg = data[:,1]
        self.adj_gps = data[:,2]
        self.duration = data[:,3]
        self.departure_timestamp = data[:,4]
        self.departure_tuple = np.asarray([timestamp2timetuple(x) for x in self.departure_timestamp])
        self.travel_time = data[:,5]
        self.adj_seg_lens = np.asarray([len(x) for x in self.adj_seg])
        self.adj_gps_lens = np.asarray([len(x) for x in self.adj_gps])
        self.idxs = np.asarray([x for x in range(len(self.traj_idxs))])

    def __getitem__(self, idx):
        return [
            self.idxs[idx],
            self.adj_seg[idx],
            self.departure_tuple[idx],
            self.adj_seg_lens[idx],
            self.travel_time[idx]
            ]

    def __len__(self):
        return len(self.traj_idxs)

class SpdRegFinetuneDataset(Dataset):
    def __init__(self, df_folder: str):
        data = np.load(df_folder, allow_pickle=True)
        self.adj_seg = data[:,0:2]
        self.speed = data[:,-1]
        self.departure_timestamp = data[:,2]
        self.departure_tuple = np.asarray([timestamp2timetuple(x) for x in self.departure_timestamp])
        self.adj_seg_lens = np.asarray([len(x) for x in self.adj_seg])
        self.idxs = np.asarray([x for x in range(len(self.adj_seg))])

    def __getitem__(self, idx):
        return [
            self.idxs[idx],
            self.adj_seg[idx],
            self.departure_tuple[idx],
            self.adj_seg_lens[idx],
            self.speed[idx]
            ]

    def __len__(self):
        return len(self.idxs)


class CLSFinetuneDataset(Dataset):
    def __init__(self, df_folder: str, data_feature):
        data = np.load(df_folder, allow_pickle=True)
        sample_rate = 1
        print(f"-----------------------------------------------")
        print(f"Sample the data with scale of 1 / {sample_rate}")
        print(f"-----------------------------------------------")
        data = data[::sample_rate]
        self.traj_idxs = data[:,0]
        self.adj_seg = data[:,1]
        self.edge_data = data_feature['edgeinfo']
        # self.highway = {'living_street':1, 'motorway':2, 'motorway_link':3, 'plannned':4,
        #                 'trunk':5, "secondary":6, "trunk_link":7, "tertiary_link":8,
        #                 "primary":9, "residential":10, "primary_link":11, "unclassified":12,
        #                 "tertiary":13, "secondary_link":14}
        self.highway = {#'out_of_classes': 0,
                    'living_street': 0, 'motorway_link': 1,
                   'trunk': 2, "secondary": 3, "trunk_link": 4, "tertiary_link": 5,
                   "primary": 6, "residential": 7, "primary_link": 8,
                   "tertiary": 9, "secondary_link": 10}
        self.edge_label = self.get_cls_labesls(self.adj_seg, self.edge_data, self.highway)
        self.departure_timestamp = data[:, 4]
        self.departure_tuple = np.asarray([timestamp2timetuple(x) for x in self.departure_timestamp])
        self.adj_seg_lens = np.asarray([len(x) for x in self.adj_seg])
        self.idxs = np.asarray([x for x in range(len(self.traj_idxs))])

    def __getitem__(self, idx):
        return [
            self.idxs[idx],
            self.adj_seg[idx],
            self.adj_seg_lens[idx],
            self.departure_tuple[idx],
            self.edge_label[idx]
            ]

    def get_cls_labesls(self, adj_seg, edge_data, highway):
        edge_labels = []
        for _, segs in enumerate(tqdm(adj_seg, desc='proccess the segment label ')):
            infos = []
            for x in segs:
                # idx= -100 padding idx,  ['motorway', 'motorway_link'] or 'motorway'
                info = edge_data[x][0]
                if info.startswith("['") and info.endswith("']"):
                    info = eval(info)[0]
                infos.append(highway[info] if info in highway.keys() else -100)
            edge_labels.append(infos)
        return edge_labels

    def __len__(self):
        return len(self.traj_idxs)

class PRFinetuneDataset(Dataset):
    def __init__(self, df_folder: str):
        data = np.load(df_folder, allow_pickle=True)
        self.traj_idxs = data[:,0]
        self.adj_seg = data[:,1]
        self.sim = data[:,2]
        self.wt_sim = data[:,3]

        self.adj_seg_lens = np.asarray([len(x) for x in self.adj_seg])
        self.idxs = np.asarray([x for x in range(len(self.traj_idxs))])

    def __getitem__(self, idx):
        return [
            self.idxs[idx],
            self.traj_idxs[idx],
            self.adj_seg[idx],
            self.adj_seg_lens[idx],
            self.wt_sim[idx]
            ]

    def __len__(self):
        return len(self.traj_idxs)


class SegBatchSampler():
    def __init__(self, count, adj_segments_lens, idxs, batch_size, drop_last = True):
        self.count = count
        self.adj_segments_lens = adj_segments_lens
        self.batch_size = batch_size
        self.idxs = list(idxs)
        self.drop_last = drop_last

        np.random.shuffle(self.idxs)
        self.chunk_size = self.batch_size * 100
        self.chunks = (self.count + self.chunk_size - 1) // self.chunk_size
        # re-arrange indices to minimize the padding
        for i in range(self.chunks):
            partial_indices = self.idxs[i * self.chunk_size: (i + 1) * self.chunk_size]
            partial_indices.sort(key=lambda x: self.adj_segments_lens[x], reverse=True)
            self.idxs[i * self.chunk_size: (i + 1) * self.chunk_size] = partial_indices

        # yield batch, and the last has been dropped
        self.batches = (self.count - 1 + self.batch_size) // self.batch_size

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        for i in range(self.batches):
            yield self.idxs[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size




def get_feature_data(config):
    data_feature ={}

    if 'TTE' in config['model'] or 'MultiTaskModel' in config['model']:
        # travel time is normalized with scaler
        # data_feature['time_standard'] = (config['time_mean'], config['time_std'])
        data_feature['time_standard'] = Floatvalue_StandardScaler(config['time_mean'], config['time_std'])

    if 'MLP_PR' in config['model'] or 'MultiTaskModel' in config['model']:
        data_feature['sim_standard'] = Floatvalue_StandardScaler(mean=config['sim_mean'], std=config['sim_std'])
    if 'MLP_REG' in config['model'] or 'MultiTaskModel' in config['model']:
        data_feature['spd_standard'] = Floatvalue_StandardScaler(mean=config['speed_mean'], std=config['speed_std'])


    # basic edge and node attributes in a specific city data
    with open(os.path.join(config['base_dir'], config['edges_dir']), 'rb') as f:
        data_feature['edgeinfo'] = pkl.load(f)
    with open(os.path.join(config['base_dir'], config['nodes_dir']), 'rb') as f:
        data_feature['nodeinfo'] = pkl.load(f)

    return data_feature

def get_dataset(config, data_feature):
    return getattr(import_module('utils.DataLoader'), f"get_{config['task']}_{config['model']}_dataloader")(dict(config), data_feature)


def get_reg_finetune_MLP_REG_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        full_data[phase] = SpdRegFinetuneDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}_REG_timeorder.npy"))
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            shuffle=True if phase == 'train' else False,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            collate_fn = collate_fn)
    return dataloader.copy()

def get_tte_finetune_MLP_TTE_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        full_data[phase] = RegFinetuneDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}.npy"))
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=SegBatchSampler(count=len(full_data[phase]),adj_segments_lens=full_data[phase].adj_seg_lens,idxs=full_data[phase].idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn)
    return dataloader.copy()

def get_cls_finetune_MLP_CLS_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        full_data[phase] = CLSFinetuneDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}.npy"), data_feature=data_feature)
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=SegBatchSampler(count=len(full_data[phase]),adj_segments_lens=full_data[phase].adj_seg_lens,idxs=full_data[phase].idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn)
    return dataloader.copy()

def get_pr_finetune_MLP_PR_dataloader(config, data_feature):
    # load the segment dataset and output the class of segdataset
    full_data = {}
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        full_data[phase] = PRFinetuneDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}_PR_top_4.npy"))
        collate_fn = CollateFnWrapper(config, data_feature)
        dataloader[phase] = DataLoader(
            dataset=full_data[phase],
            batch_sampler=SegBatchSampler(count=len(full_data[phase]),adj_segments_lens=full_data[phase].adj_seg_lens,idxs=full_data[phase].idxs, batch_size=config['batch_size']),
            num_workers=config['num_workers'],
            collate_fn = collate_fn)
    return dataloader.copy()

class CollateFnWrapper:
    def __init__(self, config, data_feature):
        self.config = config
        self.data_feature = data_feature

    def __call__(self, batch):
        return getattr(import_module('utils.model_collate_fn'),self.config['collate_func'])(data=batch, config=self.config, data_feature=self.data_feature)


class MultiTaskDataset(Dataset):
    """
    为多任务训练在原始样本末尾附加 task_id。
    """
    def __init__(self, base_dataset: Dataset, task_id: int):
        self.base_dataset = base_dataset
        self.task_id = task_id

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        # 原始样本为 list，直接在末尾追加 task_id
        return sample + [self.task_id]


class RoundRobinDataLoader:
    """
    轮询多个 DataLoader，保证每个 batch 仅来自单一任务。
    """
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        # 使用“均匀调度”：按照各任务 batch 数的比例在整个 epoch 中均匀穿插
        iters = [iter(dl) for dl in self.dataloaders]
        lengths = [len(dl) for dl in self.dataloaders]
        if len(lengths) == 0:
            return
        max_len = max(lengths)
        # interval 越小越频繁，初始 next_time=0，按 interval 递增
        intervals = [max_len / l if l > 0 else float('inf') for l in lengths]
        heap = []
        for idx, it in enumerate(iters):
            if lengths[idx] > 0:
                heapq.heappush(heap, (0.0, idx, it))

        while heap:
            next_time, idx, it = heapq.heappop(heap)
            try:
                yield next(it)
                next_time += intervals[idx]
                heapq.heappush(heap, (next_time, idx, it))
            except StopIteration:
                continue

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)


def get_mt_finetune_MultiTaskModel_dataloader(config, data_feature):
    dataloader = {}
    # 任务 -> 默认 batch_size 映射
    default_task_bs = {'1': 16, '2': 32, '3': 64, '4': 1024}
    task_bs = config.get('task_batch_size', default_task_bs)
    # 任务 -> 采样比例映射
    default_sample_ratio = {'1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}
    sample_ratio = config.get('task_sample_ratio', default_sample_ratio)
    for phase in ['train', 'val', 'test']:
        # 1: TTE, 2: CLS, 3: PR, 4: REG
        datasets = []
        # TTE
        tte_ds = RegFinetuneDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}.npy"))
        tte_len = int(len(tte_ds) * sample_ratio.get('1', 1.0))
        tte_ds = torch.utils.data.Subset(tte_ds, np.random.choice(len(tte_ds), tte_len, replace=False)) if tte_len < len(tte_ds) else tte_ds
        datasets.append(MultiTaskDataset(tte_ds, task_id=1))
        # CLS
        cls_ds = CLSFinetuneDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}.npy"),
                                   data_feature=data_feature)
        cls_len = int(len(cls_ds) * sample_ratio.get('2', 1.0))
        cls_ds = torch.utils.data.Subset(cls_ds, np.random.choice(len(cls_ds), cls_len, replace=False)) if cls_len < len(cls_ds) else cls_ds
        datasets.append(MultiTaskDataset(cls_ds, task_id=2))
        # PR
        pr_ds = PRFinetuneDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}_PR_top_4.npy"))
        pr_len = int(len(pr_ds) * sample_ratio.get('3', 1.0))
        pr_ds = torch.utils.data.Subset(pr_ds, np.random.choice(len(pr_ds), pr_len, replace=False)) if pr_len < len(pr_ds) else pr_ds
        datasets.append(MultiTaskDataset(pr_ds, task_id=3))
        # REG
        reg_ds = SpdRegFinetuneDataset(df_folder=os.path.join(config['base_dir'], config['data_dir'], f"{phase}_REG_timeorder.npy"))
        reg_len = int(len(reg_ds) * sample_ratio.get('4', 1.0))
        reg_ds = torch.utils.data.Subset(reg_ds, np.random.choice(len(reg_ds), reg_len, replace=False)) if reg_len < len(reg_ds) else reg_ds
        datasets.append(MultiTaskDataset(reg_ds, task_id=4))

        collate_fn = CollateFnWrapper(config, data_feature)
        shuffle_flag = True if phase == 'train' else False
        loaders = []
        for idx, ds in enumerate(datasets):
            loaders.append(
                DataLoader(
                    dataset=ds,
                    batch_size=task_bs.get(str(idx + 1), 32),
                    shuffle=shuffle_flag,
                    num_workers=config['num_workers'],
                    collate_fn=collate_fn
                )
            )
        dataloader[phase] = RoundRobinDataLoader(loaders)
    return dataloader.copy()

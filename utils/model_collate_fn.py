import os
import time
import torch
import numpy as np

highway = {'living_street': 1, 'motorway': 2, 'motorway_link': 3, 'plannned': 4, 'trunk': 5, "secondary": 6,
           "trunk_link": 7, "tertiary_link": 8, "primary": 9, "residential": 10, "primary_link": 11, "unclassified": 12,
           "tertiary": 13, "secondary_link": 14}
node_type = {'turning_circle': 1, 'traffic_signals': 2, 'crossing': 3, 'motorway_junction': 4, "mini_roundabout": 5}


def TTE4SimPRL_collate_fn(data, config, data_feature):
    batch_size = len(data)
    cls_token, pad_teken, mask_token = config['cls_token'], config['pad_token'], config['mask_token']
    seg_id = []
    departure_timestamp = []
    seg_len = []
    time = []
    for ind, l in enumerate(data):
        seg_id.append(l[1])
        # departure_timestamp.append(timestamp2timetuple(l[2]))
        departure_timestamp.append(l[2])
        seg_len.append(l[3])
        time.append(l[4])

    seg_len = np.asarray(seg_len)
    # departure_timestamp = torch.unsqueeze(torch.FloatTensor(departure_timestamp),dim=1)
    departure_timestamp = torch.FloatTensor(np.asarray(departure_timestamp))
    max_seq_length = seg_len.max() if 'moe' not in config['peft'] else config['max_seq_len']

    # construct the matrix with batch_size x max_seq_len
    mask = np.arange(max_seq_length) < seg_len[:, None]
    # pad the linkids with unfixed length into the fixed length
    padded_seg_id = np.full(mask.shape, dtype=np.int16, fill_value=pad_teken)
    padded_seg_id[mask] = np.concatenate(seg_id)
    padded_seg_id = torch.LongTensor(padded_seg_id)
    # construct the mask encoder for the BERT, - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.
    mask_encoder_seg = torch.zeros(mask.shape, dtype=torch.float16)
    mask_encoder_seg.masked_fill_(torch.BoolTensor(mask), value=1)

    if 'cls' in config['pooler_type']:
        # add a [cls] token in the 1st index, [cls] token is set as (config['edges'] + 1)
        padded_seg_id = torch.cat([torch.full((batch_size, 1), fill_value=cls_token), padded_seg_id], dim=1)
        mask_encoder_seg = torch.cat([torch.ones((batch_size, 1)), mask_encoder_seg], dim=1)

    return {'input_ids': padded_seg_id, 'attention_mask_seg': mask_encoder_seg, 'lon_lat': None,
            'attention_mask_gps': None,
            'mlm_input_ids': None, 'mlm_labels': None, 'departure_timestamp': departure_timestamp}, torch.FloatTensor(
        time)

def REG4SimPRL_collate_fn(data, config, data_feature):
    batch_size = len(data)
    cls_token, pad_teken, mask_token = config['cls_token'], config['pad_token'], config['mask_token']
    seg_id = []
    departure_timestamp = []
    seg_len = []
    spd = []
    for ind, l in enumerate(data):
        seg_id.append(l[1])
        departure_timestamp.append(l[2])
        seg_len.append(l[3])
        spd.append(l[4])

    seg_len = np.asarray(seg_len)
    departure_timestamp = torch.FloatTensor(np.asarray(departure_timestamp))
    max_seq_length = seg_len.max()  # if 'moe' not in config['peft'] else config['max_seq_len']

    # construct the matrix with batch_size x max_seq_len
    mask = np.arange(max_seq_length) < seg_len[:, None]
    # pad the linkids with unfixed length into the fixed length
    padded_seg_id = np.full(mask.shape, dtype=np.int16, fill_value=pad_teken)
    padded_seg_id[mask] = np.concatenate(seg_id)
    padded_seg_id = torch.LongTensor(padded_seg_id)
    # construct the mask encoder for the BERT, - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.
    mask_encoder_seg = torch.zeros(mask.shape, dtype=torch.float16)
    mask_encoder_seg.masked_fill_(torch.BoolTensor(mask), value=1)

    if 'cls' in config['pooler_type']:
        # add a [cls] token in the 1st index, [cls] token is set as (config['edges'] + 1)
        padded_seg_id = torch.cat([torch.full((batch_size, 1), fill_value=cls_token), padded_seg_id], dim=1)
        mask_encoder_seg = torch.cat([torch.ones((batch_size, 1)), mask_encoder_seg], dim=1)

    return {'input_ids': padded_seg_id, 'attention_mask_seg': mask_encoder_seg, 'lon_lat': None,
            'attention_mask_gps': None,
            'mlm_input_ids': None, 'mlm_labels': None, 'departure_timestamp': departure_timestamp}, torch.FloatTensor(
        spd)


def CLS4SimPRL_collate_fn(data, config, data_feature):
    batch_size = len(data)
    cls_token, pad_teken, mask_token = config['cls_token'], config['pad_token'], config['mask_token']
    linkids = []
    link_labels = []
    inds = []
    lens = []
    departure_timestamp = []
    for ind, l in enumerate(data):
        linkids.append(np.asarray(l[1]))
        link_labels.append(np.asarray(l[4]))
        inds.append(l[0])
        lens.append(l[2])
        departure_timestamp.append(l[3])

    seg_len = np.asarray(lens)
    # departure_timestamp = torch.unsqueeze(torch.FloatTensor(departure_timestamp),dim=1)
    departure_timestamp = torch.FloatTensor(np.asarray(departure_timestamp))
    max_seq_length = seg_len.max()  # if 'moe' not in config['peft'] else config['max_seq_len']

    # construct the matrix with batch_size x max_seq_len
    mask = np.arange(max_seq_length) < seg_len[:, None]
    # pad the linkids with unfixed length into the fixed length
    padded_seg_id = np.full(mask.shape, dtype=np.int16, fill_value=pad_teken)
    padded_seg_id[mask] = np.concatenate(linkids)
    padded_seg_id = torch.LongTensor(padded_seg_id)
    # construct the mask encoder for the BERT, - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.
    mask_encoder_seg = torch.zeros(mask.shape, dtype=torch.float16)
    mask_encoder_seg.masked_fill_(torch.BoolTensor(mask), value=1)

    padded_linkids_labels = np.full(mask.shape, fill_value=-100, dtype=np.int16)
    padded_linkids_labels[mask] = np.concatenate(link_labels)
    padded_linkids_labels = torch.LongTensor(padded_linkids_labels)

    if 'cls' in config['pooler_type']:
        # add a [cls] token in the 1st index, [cls] token is set as (config['edges'] + 1)
        padded_seg_id = torch.cat([torch.full((batch_size, 1), fill_value=cls_token), padded_seg_id], dim=1)
        mask_encoder_seg = torch.cat([torch.ones((batch_size, 1)), mask_encoder_seg], dim=1)
        padded_linkids_labels = torch.cat([torch.full((batch_size, 1), fill_value=-100), padded_linkids_labels], dim=1)

    return {'input_ids': padded_seg_id, 'attention_mask_seg': mask_encoder_seg, 'lon_lat': None,
            'attention_mask_gps': None,
            'mlm_input_ids': None, 'mlm_labels': None,
            'departure_timestamp': departure_timestamp}, padded_linkids_labels.reshape(-1)


def PR4SimPRL_collate_fn(data, config, data_feature):
    batch_size = len(data)
    cls_token, pad_teken, mask_token = config['cls_token'], config['pad_token'], config['mask_token']
    linkids = []
    link_sim = []
    inds = []
    lens = []
    for ind, l in enumerate(data):
        linkids.append(np.asarray(l[2]))
        link_sim.append(np.asarray(l[4]))
        inds.append(l[1])
        lens.append(l[3])

    seg_len = np.asarray(lens)
    max_seq_length = seg_len.max()  # if 'moe' not in config['peft'] else config['max_seq_len']

    # construct the matrix with batch_size x max_seq_len
    mask = np.arange(max_seq_length) < seg_len[:, None]
    # pad the linkids with unfixed length into the fixed length
    padded_seg_id = np.full(mask.shape, dtype=np.int16, fill_value=pad_teken)
    padded_seg_id[mask] = np.concatenate(linkids)
    padded_seg_id = torch.LongTensor(padded_seg_id)
    # construct the mask encoder for the BERT, - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.
    mask_encoder_seg = torch.zeros(mask.shape, dtype=torch.float16)
    mask_encoder_seg.masked_fill_(torch.BoolTensor(mask), value=1)

    # padded_linkids_sim = np.full(mask.shape, fill_value=-100, dtype=np.int16)
    # padded_linkids_sim[mask] = np.concatenate(link_sim)
    # padded_linkids_sim = torch.LongTensor(padded_linkids_sim)

    if 'cls' in config['pooler_type']:
        # add a [cls] token in the 1st index, [cls] token is set as (config['edges'] + 1)
        padded_seg_id = torch.cat([torch.full((batch_size, 1), fill_value=cls_token), padded_seg_id], dim=1)
        mask_encoder_seg = torch.cat([torch.ones((batch_size, 1)), mask_encoder_seg], dim=1)
        # padded_linkids_sim = torch.cat([torch.full((batch_size,1),fill_value=-100), padded_linkids_sim], dim=1)

    return {'input_ids': padded_seg_id, 'attention_mask_seg': mask_encoder_seg, 'lon_lat': None,
            'attention_mask_gps': None,
            'traj_idxs': torch.IntTensor(np.asarray(inds)), 'mlm_input_ids': None, 'mlm_labels': None,
            'departure_timestamp': None}, torch.FloatTensor(np.asarray(link_sim)).reshape(-1)

def MultiTaskModel_collate_fn(data, config, data_feature):
    task_id = data[0][-1]
    if not all([d[-1] == task_id for d in data]):
        raise ValueError("A multi-task batch must contain only one task_id.")
    if task_id == 1:  # tte
        outputs = TTE4SimPRL_collate_fn(data, config, data_feature)
    elif task_id == 2:  # cls
        outputs = CLS4SimPRL_collate_fn(data, config, data_feature)
    elif task_id == 3:  # pr
        outputs = PR4SimPRL_collate_fn(data, config, data_feature)
    elif task_id == 4:  # reg
        outputs = REG4SimPRL_collate_fn(data, config, data_feature)
    else:
        return ValueError(f"{task_id} not exist!")
    features, labels = outputs
    features['task_id'] = torch.unsqueeze(torch.tensor([task_id] * len(data)), dim=1)
    # 包装标签，便于多任务损失中区分任务类型
    labels = {'labels': labels, 'task_id': torch.tensor(task_id)}
    return features, labels

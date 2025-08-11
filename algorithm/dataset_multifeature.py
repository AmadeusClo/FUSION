# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

def search_recent_data(train, label_start_idx, T_p, T_h):
    """
    T_p: prediction time steps
    T_h: historical time steps
    """
    if label_start_idx + T_p > len(train): return None
    start_idx, end_idx = label_start_idx - T_h, label_start_idx - T_p + T_p
    if start_idx < 0 or end_idx < 0: return None
    return (start_idx, end_idx), (label_start_idx, label_start_idx + T_p)


def search_multihop_neighbor(adj, hops=5):
    node_cnt = adj.shape[0]
    hop_arr = np.zeros((adj.shape[0], adj.shape[0]))
    for h_idx in range(node_cnt):  # refer node idx(n)
        tmp_h_node, tmp_neibor_step = [h_idx], [h_idx]  # save spatial corr node  # 0 step(self) first
        hop_arr[h_idx, :] = -1  # if the value exceed maximum hop, it is set to (hops + 1)
        hop_arr[h_idx, h_idx] = 0  # at begin, the hop of self->self is set to 0
        for hop_idx in range(hops):  # how many spatial steps
            tmp_step_node = []  # neighbor nodes in the previous k step
            tmp_step_node_kth = []  # neighbor nodes in the kth step
            for tmp_nei_node in tmp_neibor_step:
                tmp_neibor_step = list((np.argwhere(adj[tmp_nei_node] == 1).flatten()))  # find the one step neighbor first
                tmp_step_node += tmp_neibor_step
                tmp_step_node_kth += set(tmp_step_node) - set(tmp_h_node)  # the nodes that have appeared in the first k-1 step are no longer needed
                tmp_h_node += tmp_neibor_step
            tmp_neibor_step = tmp_step_node_kth.copy()
            all_spatial_node = list(set(tmp_neibor_step))  # the all spatial node in kth step
            hop_arr[h_idx, all_spatial_node] = hop_idx + 1
    return hop_arr[:, :, np.newaxis]

class TrainCleanDataset_multifeature():
    def __init__(self, config):

        self.data_name = config.data.name
        self.feature_file = config.data.feature_file
        self.val_start_idx = config.data.val_start_idx
        self.adj = np.load(config.data.spatial)
        self.label, self.feature = self.read_data()

        #for stpgcn
        if config.model.get('alpha', None) is not None:
            self.alpha = config.model.alpha
            self.t_size = config.model.t_size
            self.spatial_distance = search_multihop_neighbor(self.adj, hops=self.alpha)
            self.range_mask = self.interaction_range_mask(hops=self.alpha, t_size=self.t_size)

   
    def read_data(self):
        data = np.load(self.feature_file)
        data = data.astype(np.float32)
        
        # if 'PEMS' in self.data_name or 'AIR' in self.data_name or 'Metro' in self.data_name:
        #     data = np.nan_to_num(data, nan=0)
        
        # # 对整个数据进行标准化处理
        # normalized_data = self.normalization(data).astype('float32')
        
        return data, data


    def normalization(self, data):
        # 计算每个通道的均值和标准差
        mean = np.mean(data, axis=(0, 1), keepdims=True)
        std = np.std(data, axis=(0, 1), keepdims=True)
        
        # 标准化处理
        normalized_data = (data - mean) / std
        self.mean = mean
        self.std = std
        
        return normalized_data

    def reverse_normalization(self, data):
        # 使用保存的均值和标准差进行反标准化处理
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        mean = np.array(self.mean)  # 确保mean是numpy数组
        std = np.array(self.std)
        mean_change = mean[:, :, :, np.newaxis, np.newaxis]  # 形状变为 (1, 1, F, 1)
        std_change = std[:, :, :, np.newaxis, np.newaxis]  # 形状变为 (1, 1, F, 1)
        
        # 仅对第2维（通道维）进行逆标准化
        reversed_data = data * std_change + mean_change
        
        reversed_data = np.squeeze(reversed_data, axis=0)
            
        return reversed_data
    
    def reverse_sample_normalization(self, data):
            # 使用保存的均值和标准差进行反标准化处理
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        mean = np.array(self.mean)  # 确保mean是numpy数组
        std = np.array(self.std)
        # 获取data的形状
        B, n_samples, F, V, T = data.shape
        
        # 扩展mean和std的形状以匹配data的形状
        mean_change = mean[:, :, :, np.newaxis, np.newaxis]  # 形状变为 (1, 1, F, 1, 1)
        std_change = std[:, :, :, np.newaxis, np.newaxis]  # 形状变为 (1, 1, F, 1, 1)
        
        # 广播mean_change和std_change以匹配data的形状
        mean_change = np.tile(mean_change, (B, n_samples, 1, V, T))  # 复制mean_change以匹配data的形状
        std_change = np.tile(std_change, (B, n_samples, 1, V, T))  # 复制std_change以匹配data的形状
        
        # 对第3维（F维）进行逆标准化
        reversed_data = data * std_change + mean_change
            
        return reversed_data
    
    # for stpgcn
    def interaction_range_mask(self, hops=2, t_size=3):
        hop_arr = self.spatial_distance
        hop_arr[hop_arr != -1] = 1
        hop_arr[hop_arr == -1] = 0
        return np.concatenate([hop_arr.squeeze()] * t_size, axis=-1)  # V,tV



class TrainprocessDataset(Dataset):
    def __init__(self, clean_data, data_range, config):
        self.T_h = config.model.T_h
        self.T_p = config.model.T_p
        self.V = config.model.V
        self.points_per_hour = config.data.points_per_hour
        self.data_range = data_range
        self.data_name = clean_data.data_name


        self.label = np.array(clean_data.label) # (T_total, V, D), where T_all means the total time steps in the data
        self.feature = np.array(clean_data.feature)  # (T_total, V, D)

        # Prepare samples
        self.idx_lst = self.get_idx_lst()
        print('sample num:', len(self.idx_lst))

    def __getitem__(self, index):

        recent_idx = self.idx_lst[index]

        start, end = recent_idx[1][0], recent_idx[1][1]
        label = self.label[start:end]

        start, end = recent_idx[0][0], recent_idx[0][1]
        node_feature = self.feature[start:end]
        pos_w, pos_d = self.get_time_pos(start)
        pos_w = np.array(pos_w, dtype=np.int32)
        pos_d = np.array(pos_d, dtype=np.int32)
        return label, node_feature, pos_w, pos_d

    def __len__(self):
        return len(self.idx_lst)

    def get_time_pos(self, idx):
        idx = np.array(range(self.T_h)) + idx
        pos_w = (idx // (self.points_per_hour * 24)) % 7  # day of week
        pos_d = idx % (self.points_per_hour * 24)  # time of day
        return pos_w, pos_d

    def get_idx_lst(self):
        idx_lst = []
        start = self.data_range[0]
        end = self.data_range[1] if self.data_range[1] != -1 else self.feature.shape[0]

        for label_start_idx in range(start, end):
            # only 6:00-24:00 for Metro data
            if 'Metro' in self.data_name:
                if label_start_idx % (24 * 6) < (7 * 6):
                    continue
                if label_start_idx % (24 * 6) > (24 * 6) - self.T_p:
                    continue

            recent = search_recent_data(self.feature, label_start_idx, self.T_p, self.T_h)  # recent data

            if recent:
                idx_lst.append(recent)
        return idx_lst

    #################################################


class TestCleanDataset_multifeature(Dataset):
    def __init__(self, config):
        self.dataset_test = np.load(config.data.test_file)
        self.dataset_test = self.dataset_test.astype(np.float32)
        self.t_h = config.model.T_h
        self.t_p = config.model.T_p
        self.segment_length = self.t_h + self.t_p
        self.segments = self.split_segments()
        # self.normalized_segments = self.normalize_segments()
        self.feature = self.combine_segments()
        self.label = self.feature

    def split_segments(self):
        segments = []
        for i in range(0, len(self.dataset_test), self.segment_length):
            segment = self.dataset_test[i:i + self.segment_length]
            if len(segment) == self.segment_length:
                segments.append(segment)
        return segments

    def normalize_segments(self):
        normalized_segments = []
        for segment in self.segments:
            normal_data = segment[:self.t_h]
            mean = np.mean(normal_data, axis=0)
            std = np.std(normal_data, axis=0)
            normalized_segment = (segment - mean) / std
            normalized_segments.append(normalized_segment)
        return normalized_segments
    
    def combine_segments(self):
        combined_data = np.zeros_like(self.dataset_test)
        for i, segment in enumerate(self.segments):
            start_idx = i * self.segment_length
            end_idx = start_idx + self.segment_length
            combined_data[start_idx:end_idx] = segment
        return combined_data

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]

class TestprocessDataset(Dataset):
    def __init__(self, clean_data, data_range, config):
        self.T_h = config.model.T_h
        self.T_p = config.model.T_p
        self.V = config.model.V
        self.points_per_hour = config.data.points_per_hour
        self.data_range = data_range
        self.segment_length = self.T_h + self.T_p


        self.label = np.array(clean_data.label) # (T_total, V, D), where T_all means the total time steps in the data
        self.feature = np.array(clean_data.feature)  # (T_total, V, D)

        # Prepare samples
        self.idx_lst = self.get_idx_lst()
        print('sample num:', len(self.idx_lst))

    def __getitem__(self, index):

        recent_idx = self.idx_lst[index]

        start, end = recent_idx[1][0], recent_idx[1][1]
        label = self.label[start:end]

        start, end = recent_idx[0][0], recent_idx[0][1]
        node_feature = self.feature[start:end]
        pos_w, pos_d = self.get_time_pos(start)
        pos_w = np.array(pos_w, dtype=np.int32)
        pos_d = np.array(pos_d, dtype=np.int32)
        return label, node_feature, pos_w, pos_d

    def __len__(self):
        return len(self.idx_lst)

    def get_time_pos(self, idx):
        idx = np.array(range(self.T_h)) + idx
        pos_w = (idx // (self.points_per_hour * 24)) % 7  # day of week
        pos_d = idx % (self.points_per_hour * 24)  # time of day
        return pos_w, pos_d
    
    def get_idx_lst(self):
        idx_list = list(range(0, self.feature.shape[0] - self.segment_length + 1, self.segment_length))
        idx_lst = []
        for idx in idx_list:
            recent = search_recent_data(self.feature, idx, self.T_p, self.T_h)
            # 根据需要处理recent数据
            
            if recent:
                idx_lst.append(recent)
        return idx_lst
    
if __name__ == '__main__':
    pass

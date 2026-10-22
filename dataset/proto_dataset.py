import torch
import torch.utils.data as data
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from .Probe_dataset import topk_TrainDataset 

class PKSampler(Sampler):
    def __init__(self, data_source, p=8, k=8):
        self.data_source = data_source
        self.p = p
        self.k = k
        self.batch_size = p * k
        
        self.label_to_indices = defaultdict(list)
                
        for idx in range(len(data_source)):
            vid_str = data_source.id_list[idx]
            prefix = vid_str.split('_')[0]
            
            if 'hvu' in prefix: offset = 0
            elif 'ucf' in prefix: offset = 200
            elif 'chsims' in prefix: offset = 300
            else: offset = 0 # Fallback
            
            # 2. Get local label
            item = data_source.neighbors_data[idx]
            if item is None:
                local_label = 0 
            else:
                local_label = item['kmeans_label']
            
            if hasattr(local_label, 'item'):
                local_label = local_label.item()
            
            global_label = local_label + offset
            
            self.label_to_indices[global_label].append(idx)
            
        self.valid_labels = [l for l, idxs in self.label_to_indices.items()] 
        # print(f"PKSampler: cluster numbers: {len(self.valid_labels)}")

    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        
        for _ in range(num_batches):
            batch_indices = []
            
            if len(self.valid_labels) < self.p:
                chosen_labels = np.random.choice(self.valid_labels, len(self.valid_labels), replace=False)
            else:
                chosen_labels = np.random.choice(self.valid_labels, self.p, replace=False)
            
            for label in chosen_labels:
                indices = self.label_to_indices[label]
                replace = len(indices) < self.k
                chosen_indices = np.random.choice(indices, self.k, replace=replace)
                batch_indices.extend(chosen_indices)

            yield batch_indices

    def __len__(self):
        return len(self.data_source) // self.batch_size


def get_proto_loader(cfg, p=8, k=8):
    dataset = topk_TrainDataset(cfg)
    sampler = PKSampler(dataset, p=p, k=k)
    
    return data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=cfg.workers,
        pin_memory=True
    )
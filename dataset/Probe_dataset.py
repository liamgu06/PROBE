import multiprocessing as mp
import pdb
import pickle
import random
from multiprocessing import Process, Queue
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import torch
import torch.utils.data as data
import numpy as np
import h5py
from transformers import CLIPTokenizer,CLIPModel
from configs.Probe_config import TrainingConfig
from utils.env import logging
import json
import torch.nn as nn
import torch.nn.functional as F
from model.Probe_model import ProbeModel
import os

class RandomMaskingGenerator:
    """_summary_
    Random masking process on the dataset.
    """
    def __init__(self, max_frames, mask_ratio):
        self.num_patches = max_frames
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        idx = [i for i in range(self.num_patches)]
        random.shuffle(idx)
        idx1 = idx[:(self.num_patches - self.num_mask)]
        idx2 = idx[-(self.num_patches - self.num_mask):]
        mask1 = np.ones(self.num_patches)
        mask2 = np.ones(self.num_patches)
        mask1[idx1] = 0.
        mask2[idx2] = 0.
        return [mask1, mask2]



class topk_TrainDataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.cross_task_config = {
            "event":     ("event_conditional_reasoning",     "EVENT_TEST_PROMPTS"),
            "attribute": ("attribute_conditional_reasoning", "ATTRIBUTE_TEST_PROMPTS"),
            "concept":   ("concept_conditional_reasoning",   "CONCEPT_TEST_PROMPTS"),
            "object":    ("object_conditional_reasoning",    "OBJECT_TEST_PROMPTS")
        }
        self.valid_indices = {}
               
        print(f"CrossDataset: Loading Teacher Features from {cfg.teacher_feat_path}...")
        self.teacher_h5 = h5py.File(cfg.teacher_feat_path, 'r')
        
        print("CrossDataset: Scanning sparse valid indices...")
        for name, (h5_key, _) in self.cross_task_config.items():
            if h5_key in self.teacher_h5:
                dset = self.teacher_h5[h5_key][:] 
                if dset.ndim == 3: dset = dset.squeeze(1)
                
                norms = np.linalg.norm(dset, axis=1)
                valid_idxs = np.where(norms > 1e-5)[0].tolist()
                
                if len(valid_idxs) > 0:
                    self.valid_indices[name] = valid_idxs
                    print(f" -> '{name}': {len(valid_idxs)} valid samples.")
                else:
                    print(f" -> '{name}': Empty.")
        
        print(f"Dataset: Loading Student Features from {cfg.train_feat_path}...")
        self.h5_file = h5py.File(cfg.train_feat_path, 'r')
        self.student_feats = self.h5_file['feats']
            
        print(f"Dataset: Loading ID List from {cfg.video_ids_path}...")
        with open(cfg.video_ids_path, 'r') as f:
            self.id_list = json.load(f)
            
        print(f"Dataset: Loading Neighbor Map from {cfg.train_neighbor_path}...")
        self.neighbors_data = torch.load(cfg.train_neighbor_path)
        
        self.mask_generator = RandomMaskingGenerator(cfg.max_frames, cfg.mask_ratio)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.tokenizer_name)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # pretrained gate_head
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        temp_model = ProbeModel(cfg, hash_bits=cfg.hash_bits).to(device)
        temp_model.eval()
        if hasattr(cfg, 'gate_ckpt_path') and os.path.exists(cfg.gate_ckpt_path):
            print(f" -> Loading Gate Checkpoint from: {cfg.gate_ckpt_path}")
            gate_state_dict = torch.load(cfg.gate_ckpt_path, map_location=device)
            if any(k.startswith('gate_head.') for k in gate_state_dict.keys()):
                temp_model.load_state_dict(gate_state_dict, strict=False)
            else:
                temp_model.gate_head.load_state_dict(gate_state_dict)
        else:
            print("CRITICAL WARNING: No Gate Checkpoint found!")
        
        
        prompts_map = {
            'hvu': cfg.SCENE_TEST_PROMPTS, 
            'ucf': cfg.ACT_TEST_PROMPTS, 
            'chsims': cfg.EMO_TEST_PROMPTS
        }
        
        self.prompt_masks = {}    
        self.pre_tokenized = {}   
        dataset_ids_map = {'hvu': 0, 'ucf': 1, 'chsims': 2}
        self.cross_masks = {}
        print("AuxDataset: Pre-computing Masks...")
        with torch.no_grad():
            for name, (_, prompt_attr) in self.cross_task_config.items():
                if name not in self.valid_indices: continue
                text = getattr(cfg, prompt_attr, f"A video about {name}")
                tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors="pt")
                input_tokens = {k: v.to(device) for k, v in tokens.items()}
                m_p, _, __ = temp_model.forward_prompt(input_tokens)
                self.cross_masks[name] = m_p.cpu().squeeze(0)
                
                
        with torch.no_grad():
            for key, text in prompts_map.items():
                tokens = self.tokenizer(
                    text, padding='max_length', truncation=True, 
                    max_length=77, return_tensors="pt"
                )
                input_tokens = {k: v.to(device) for k, v in tokens.items()}
                m_p, _ ,__= temp_model.forward_prompt(input_tokens)
                
                mask = m_p.cpu().squeeze(0) # [hash_bits]
                self.prompt_masks[key] = mask
                
                d_id = dataset_ids_map[key]
                self.pre_tokenized[d_id] = {
                    'input_ids': tokens.input_ids.squeeze(0),
                    'attention_mask': tokens.attention_mask.squeeze(0)
                }
                
                print(f" -> Generated gate mask for '{key}' (Active bits: {mask.sum().item()})")

        del temp_model
        torch.cuda.empty_cache()
        
        print(f"Dataset: Tokenized Keys Available: {list(self.pre_tokenized.keys())}")
        
        self.SAVED_TOP_CUTOFF = 512 
        
    def __len__(self):
        return len(self.id_list)
    
    def sample_cross_batch(self, batch_size=128):
        
        # task_name = np.random.choice(list(self.cross_task_config.keys()))
        task_name = np.random.choice(list(self.cross_task_config.keys()), p=self.task_probs)       
        valid_idxs = self.valid_indices.get(task_name, [])
        if len(valid_idxs) == 0:
            return self.sample_cross_batch(batch_size)
            
        replace_flag = len(valid_idxs) < batch_size
        selected_indices = np.random.choice(valid_idxs, size=batch_size, replace=replace_flag)
        selected_indices = np.sort(selected_indices) 
        
        h5_key, _ = self.cross_task_config[task_name]
        
        
        teacher_data = self.teacher_h5[h5_key][selected_indices]
        if teacher_data.ndim == 3: 
            teacher_data = teacher_data.squeeze(1)
            
        student_data = self.student_feats[selected_indices]
        
        mask = self.cross_masks[task_name] # [Bits]
        mask_expanded = mask.unsqueeze(0).repeat(batch_size, 1) # [B, Bits]
        
        return {
            "teacher_feat": torch.from_numpy(teacher_data).float(),
            "student_feat": torch.from_numpy(student_data).float(),
            "cross_mask":   mask_expanded,
            "task_name":    task_name
        }

    
    def get_mask_and_id(self, vid_str, total_bits=512):
        prefix = vid_str.split('_')[0]
        mask = torch.ones(total_bits)
        dataset_id = 0

        if 'hvu' in prefix:
            dataset_id = 0 
            key = 'hvu'
        elif 'ucf' in prefix:
            dataset_id = 1 
            key = 'ucf'
        elif 'chsims' in prefix:
            dataset_id = 2 
            key = 'chsims'
        else:
            logging.warning(f"Unknown prefix: {prefix}, using ACT mask as fallback.")
            dataset_id = 0
        mask = self.prompt_masks[key]
        return mask, dataset_id

    def __getitem__(self, index):
        
        def read_h5_safe(indices):
            # read_h5 in order
            unique_sorted_idx, inverse_idx = np.unique(indices, return_inverse=True)
            
            unique_feats = self.student_feats[unique_sorted_idx]
            
            original_feats = unique_feats[inverse_idx]
            
            return torch.from_numpy(original_feats).float()
        
        vid_str = self.id_list[index]
        
        range_mask, dataset_id = self.get_mask_and_id(vid_str, total_bits=self.cfg.hash_bits)
        
        item_data = self.neighbors_data[index]  # ['indices','scores','kmeans_label']
        all_neighbors = item_data['indices']  
        k_means_label = item_data['kmeans_label']
        POS_POOL_SIZE = 30 
        NUM_POS_SAMPLE = 5
        
        real_pos_pool_size = min(len(all_neighbors) - 1, POS_POOL_SIZE)
        pos_pool = all_neighbors[1 : real_pos_pool_size + 1]
        
        if len(pos_pool) >= NUM_POS_SAMPLE:
            pos_indices = np.random.choice(pos_pool, size=NUM_POS_SAMPLE, replace=False)
        else:
            pos_indices = np.random.choice(pos_pool, size=NUM_POS_SAMPLE, replace=True)
            
        pos_indices = pos_indices.astype(np.int64)
        
        pos_feats = read_h5_safe(pos_indices)
        
        NUM_NEG_SAMPLE = 5
        total_len = len(all_neighbors)
        
        bottom_start_idx = total_len // 2
        
        neg_pool = all_neighbors[bottom_start_idx : ]
        
        if len(neg_pool) >= NUM_NEG_SAMPLE:
            neg_indices = np.random.choice(neg_pool, size=NUM_NEG_SAMPLE, replace=False)
        else:
            neg_indices = np.random.choice(neg_pool, size=NUM_NEG_SAMPLE, replace=True)
            
        neg_indices = neg_indices.astype(np.int64)
        
        neg_feats = read_h5_safe(neg_indices) 
        
        q_feat = torch.from_numpy(self.student_feats[index]).float() 
        random_masks = self.mask_generator()
        mask_tensor = torch.tensor(np.array(random_masks)).float()
        if dataset_id not in self.pre_tokenized:
            token_data = self.pre_tokenized[0]
        else:
            token_data = self.pre_tokenized[dataset_id]
        
        return {
            "own_feat": q_feat,          # [T, D]
            "pos_feats": pos_feats,      # [5, T, D]
            "neg_feats": neg_feats,      # [5, T, D]
            
            "input_ids": token_data['input_ids'], 
            "attention_mask": token_data['attention_mask'],
            
            "meta": vid_str,
            "range_mask": range_mask,
            "random_mask": mask_tensor,
            
            "index": index,
            "dataset_id": dataset_id,
            "kmeans_label": k_means_label
        }

def get_train_loader(cfg):
    dataset = topk_TrainDataset(cfg)
    return data.DataLoader(
        dataset=dataset,
        batch_size=64, 
        shuffle=True,        
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True
    )

class TestDataset(data.Dataset):
    def __init__(self, cfg, dataset_name, mode='database'):
        self.cfg = cfg
        self.mode = mode
        self.dataset_name = dataset_name
        
        ds_conf = cfg.EVAL_DATASETS[dataset_name]
        self.ds_type = str(ds_conf['type']).strip()
        
        if self.mode == 'query':
            feat_path = ds_conf['query_feat']
        else: 
            feat_path = ds_conf['test_feat']
            
        print(f"[{dataset_name}] Loading {mode} features from: {feat_path}")
        
        self.h5_file = h5py.File(feat_path, 'r')
        self.feats = self.h5_file['feats'] 
        self.length = self.feats.shape[0]

        if self.ds_type == 'act':
            self.prompt_text = cfg.ACT_TEST_PROMPTS
        elif self.ds_type == 'emo':
            self.prompt_text = cfg.EMO_TEST_PROMPTS
        elif self.ds_type == 'scene':
            self.prompt_text = cfg.SCENE_TEST_PROMPTS
        elif self.ds_type == 'event':
            self.prompt_text = cfg.EVENT_TEST_PROMPTS
        elif self.ds_type == 'concept':
            self.prompt_text = cfg.CONCEPT_TEST_PROMPTS
        elif self.ds_type == 'attribute':
            self.prompt_text = cfg.ATTRIBUTE_TEST_PROMPTS
        elif self.ds_type == 'object':
            self.prompt_text = cfg.OBJECT_TEST_PROMPTS
        else:
            self.prompt_text = "A video."
            print(f"Warning: Unknown dataset type '{self.ds_type}', using default prompt.")
            print(f"Warning: Unknown dataset type {repr(self.ds_type)}...")

        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.tokenizer_name)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        video_feat = torch.from_numpy(self.feats[index]).float() 
        
        if self.mode == 'database':
            return {
                "pixel_values": video_feat
            }
            
        elif self.mode == 'query':
            text_inputs = self.tokenizer(
                self.prompt_text,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": video_feat,
                "input_ids": text_inputs.input_ids.squeeze(0),
                "attention_mask": text_inputs.attention_mask.squeeze(0)
            }
            
class GatePretrainDataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.tokenizer_name)
        
        self.text_list = [] 
        base_prompts = [
            cfg.ACT_TEST_PROMPTS, 
            cfg.EMO_TEST_PROMPTS, 
            cfg.SCENE_TEST_PROMPTS, 
            cfg.REASONING_TEST_PROMPTS,
            cfg.ENTITY_TEST_PROMPTS,
            cfg.ATTRIBUTE_TEST_PROMPTS,
            cfg.CONCEPT_TEST_PROMPTS,
            cfg.EVENT_TEST_PROMPTS,
            cfg.OBJECT_TEST_PROMPTS
        ]
        self.text_list.extend(base_prompts*5)
        
        new_path = "path/to/prompts/generated_prompts.json"
        with open(new_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
            for item in data:
                if 'prompt' in item:
                    self.text_list.append(item['prompt'])
                        
        random.shuffle(self.text_list)

        print(f"GatePretrainDataset: Initialization Complete. Total samples: {len(self.text_list)}")
        
        if len(self.text_list) < 100:
            print("CRITICAL WARNING: Dataset size is extremely small (<100). Model will overfit!")


    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        text = self.text_list[index]
        
        inputs = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "raw_text": text
        }
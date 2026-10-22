import torch
import numpy as np
import scipy.io as sio
import os
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.Probe_dataset import TestDataset
from utils import mAP
from configs.Probe_config import TrainingConfig
from model.Probe_model import ProbeModel

def calc_masked_hamming_dist_fast(db_codes, q_codes, q_masks, device='cuda'):
    db_codes = torch.as_tensor(db_codes, device=device, dtype=torch.float32)
    q_codes = torch.as_tensor(q_codes, device=device, dtype=torch.float32)
    q_masks = torch.as_tensor(q_masks, device=device, dtype=torch.float32)
    # sum(m * |q-d|) = sum(m) - sum((m*q) * d)
    term1 = q_masks.sum(dim=1, keepdim=True)  # [N_q, 1]
    
    weighted_q = q_codes * q_masks
    term2 = torch.matmul(weighted_q, db_codes.t())  # [N_q, N_db]
    
    dist = term1 - term2
    return dist.t().cpu().numpy()  # [N_db, N_q]



def load_labels_from_mat(path):
    try:
        mat = sio.loadmat(path)
        possible_keys = ['q_label', 'labels', 'label', 'test_label', 'query_label']
        for k in possible_keys:
            if k in mat:
                return mat[k].astype(np.float32)
        raise ValueError(f"Cannot find valid label key in {path}")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def evaluate_one_dataset(cfg, device, gate_model, hash_model, dataset_name, logger):
    ds_conf = cfg.EVAL_DATASETS[dataset_name]
    
    logger.info(f">>> Evaluating {dataset_name} ({ds_conf['type']}) <<<")
    
    # Database loader
    db_dataset = TestDataset(cfg, dataset_name, mode='database')
    db_loader = DataLoader(db_dataset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=4)
    
    db_codes_list = []
    with torch.no_grad():
        for batch in tqdm(db_loader, desc=f"[{dataset_name}] DB"):  
            feats = batch['pixel_values'].to(device)
            b_v = hash_model.inference_video(feats)
            db_codes_list.append(torch.sign(b_v).cpu())
            
    db_codes = torch.cat(db_codes_list, dim=0)  # [N_db, K]
    
    # Query loader
    q_dataset = TestDataset(cfg, dataset_name, mode='query')
    q_loader = DataLoader(q_dataset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=4)
    first_item = q_dataset[0]
    
    prompt_input_ids = first_item['input_ids'].unsqueeze(0).to(device)
    prompt_att_mask = first_item['attention_mask'].unsqueeze(0).to(device)
    gate_model.eval()
    with torch.no_grad():
        # Compute the dataset-specific mask [1, K]
        prompt_inputs = {
            'input_ids': prompt_input_ids, 
            'attention_mask': prompt_att_mask
        }
        hard_mask,__,__ = gate_model.forward_prompt(prompt_inputs)
        
    hash_model.eval()
    q_codes_list = []
    with torch.no_grad():
        for batch in tqdm(q_loader, desc=f"[{dataset_name}] Query"):
            feats = batch['pixel_values'].to(device)
            b_q = hash_model.inference_video(feats)
            q_codes_list.append(torch.sign(b_q).cpu())

    q_codes = torch.cat(q_codes_list, dim=0)  # [N_q, K]
    q_masks = hard_mask.repeat(q_codes.shape[0], 1)
    
    # Compute distances
    dist_matrix = calc_masked_hamming_dist_fast(db_codes, q_codes, q_masks, device=device)
    sorted_indices = np.argsort(dist_matrix, axis=0)
    
    # Load labels
    # Note: the database side may use `test_label` or `database_label`, depending on dataset definition.
    db_labels = load_labels_from_mat(ds_conf['test_label'])
    q_labels = load_labels_from_mat(ds_conf['query_label'])
    
    if db_labels is None or q_labels is None:
        logger.error(f"Label loading failed for {dataset_name}")
        return {}

    gt_sim = np.dot(db_labels, q_labels.transpose())
    cateTrainTest = (gt_sim > 0)

    topk_list = [5, 20, 40, 60, 80, 100]
    results = {}
    
    logger.info("-" * 50)
    for k in topk_list:
        current_map, _, _ = mAP(cateTrainTest, sorted_indices, num_return_NN=k)
        results[f'mAP@{k}'] = current_map
        logger.info(f"{dataset_name:<15} | {k:<5} | {current_map:.4f}")
        
    map_all, _, _ = mAP(cateTrainTest, sorted_indices, num_return_NN=None)
    results['mAP@All'] = map_all
    logger.info(f"{dataset_name:<15} | {'All':<5} | {map_all:.4f}")
    logger.info("-" * 50)
    
    return results


def main():
    cfg = TrainingConfig()
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger("Standalone_Eval")

    logger.info(">>> Model loading <<<")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate_model = ProbeModel(cfg, hash_bits=cfg.hash_bits).to(device)
    
    hash_model = ProbeModel(cfg, hash_bits=cfg.hash_bits).to(device)
    
    
    logger.info(f"Loading GATE weights from: {cfg.gate_ckpt_path}")
    if os.path.exists(cfg.gate_ckpt_path):
        gate_ckpt = torch.load(cfg.gate_ckpt_path, map_location=device)
        try:
            msg = gate_model.gate_head.load_state_dict(gate_ckpt, strict=True)
            logger.info(f"Gate Head loaded successfully. Msg: {msg}")
        except Exception as e:
            logger.error(f"Failed to load gate head: {e}")
            logger.error("Tip: Check if the checkpoint only contains 'gate_head' keys.")
            return
    else:
        logger.error(f"Gate checkpoint not found at: {cfg.gate_ckpt_path}")
        return
        
    logger.info(f"Loading HASH weights from: {cfg.best_model_path}")
    if os.path.exists(cfg.best_model_path):
        hash_ckpt = torch.load(cfg.best_model_path, map_location=device)

        if isinstance(hash_ckpt, dict) and 'model_state_dict' in hash_ckpt:
            state_dict = hash_ckpt['model_state_dict']
        else:
            state_dict = hash_ckpt  # Compatible with directly-saved state_dict
            
        msg = hash_model.load_state_dict(state_dict, strict=True)
        logger.info(f"Hash Model loaded. Missing keys: {len(msg.missing_keys)}")
    else:
        logger.error(f"Hash model checkpoint not found at: {cfg.best_model_path}")
        return

    gate_model.float().eval()
    hash_model.float().eval()

    all_scores = {}
    
    for ds_name in cfg.CROSS_EVAL_DATASET_LIST:
        try:
            scores = evaluate_one_dataset(cfg, device, gate_model, hash_model, ds_name, logger)
            all_scores[ds_name] = scores
        except Exception as e:
            logger.error(f"Error evaluating {ds_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()


import torch
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.Probe_model import ProbeModel
from dataset.Probe_dataset import topk_TrainDataset
from dataset.proto_dataset import PKSampler       
from configs.Probe_config import TrainingConfig
from Probe_loss import CrossConsistencyLoss, SubspaceGeometryLoss, TemporalLoss
from utils.env import set_log , set_seed
from eval import evaluate_all
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='probe_training')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')
    args = parser.parse_args()
    return args


def train():
    args = parse_args()
    cfg = TrainingConfig()
    
    cfg.seed = 59
    cfg.tau_c = 0.20
    cfg.tau_p = 0.50
    cfg.pp_weight = 0.50
    cfg.lambda_proto = 0.11
    cfg.lambda_contra = 1.35
    cfg.contra_temp = 0.70
    cfg.tau_plus = 0.15
    cfg.mask_ratio = 0.40
    cfg.lambda_recon = 1.50
    cfg.lambda_cross = 0.13
    cfg.tau_s = 0.2
    cfg.cross_gap = 10
    cfg.learning_rate = 1e-4

    cfg.log_path = "path/to/logs"
    cfg.save_path = "path/to/checkpoints"
    
    set_seed(cfg)
    if not os.path.exists(cfg.log_path): os.makedirs(cfg.log_path)
    if not os.path.exists(cfg.save_path): os.makedirs(cfg.save_path)

    logger = set_log(cfg, 'your_txt_name') 
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device 
    logger.info(f"Using device: {device} | Params Loaded.")
    
    train_ds = topk_TrainDataset(cfg)
    rank_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    
    proto_sampler = PKSampler(train_ds, p=16, k=8)
    proto_loader = torch.utils.data.DataLoader(train_ds, batch_sampler=proto_sampler, num_workers=4)
    proto_iter = iter(proto_loader)
    
    model = ProbeModel(cfg, hash_bits=cfg.hash_bits).to(device)
    
    criterion_temporal = TemporalLoss(cfg=cfg).to(device)
    
    criterion_proto = SubspaceGeometryLoss(
        total_videos=len(train_ds), cfg = cfg, device=device
    ).to(device)
    
    global_cross = CrossConsistencyLoss(cfg).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-2)

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (cfg.num_epochs - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(cfg.num_epochs):

        model.train()
        
        for batch_idx, rank_batch in enumerate(rank_loader):
            
            optimizer.zero_grad()
            
            q_raw = rank_batch['own_feat'].to(device)
            pos_raw = rank_batch['pos_feats'].to(device)
            neg_raw = rank_batch['neg_feats'].to(device)

            range_mask = rank_batch['range_mask'].to(device) 
            
            random_mask = rank_batch['random_mask'].to(device) 
            
            indices = rank_batch['index'].to(device)
            
            all_raw = torch.cat([q_raw.unsqueeze(1), pos_raw, neg_raw], dim=1)
            B, N, T, D = all_raw.shape
            
            _, all_seq_hash, all_seq_logits = model.forward_video(all_raw.view(-1, T, D))
            
            all_seq_hash = all_seq_hash.view(B, N, T, -1)
            all_seq_logits = all_seq_logits.view(B, N, T, -1)
            
            all_hash = torch.mean(all_seq_hash, dim=2)     
            all_logits = torch.mean(all_seq_logits, dim=2)
            
            q_hash, q_logits = all_hash[:, 0], all_logits[:, 0]
            pos_hash, neg_hash = all_hash[:, 1:6], all_hash[:, 6:]
            
            mask1 = random_mask[:, 0, :].bool() 
            mask2 = random_mask[:, 1, :].bool()
            recon1, v1_seq_hash, v1_seq_logits = model.forward_video(q_raw, mask1)
            recon2, v2_seq_hash, v2_seq_logits = model.forward_video(q_raw, mask2)
            
            view1_hash = torch.mean(v1_seq_hash, dim=1)
            view2_hash = torch.mean(v2_seq_hash, dim=1)  
            
            
            target1 = q_raw[mask1].reshape(B, -1, D)
            target2 = q_raw[mask2].reshape(B, -1, D)
            
            final_mask = range_mask
            
            loss_temporal_total, temporal_stats = criterion_temporal(
                q_hash,
                view1_hash=view1_hash, view2_hash=view2_hash,
                recon_1=recon1, target_1=target1,             
                recon_2=recon2, target_2=target2
            )
            
            criterion_proto.update_ema(indices, 
                                       q_logits)

            try:
                proto_batch = next(proto_iter)
            except StopIteration:
                proto_iter = iter(proto_loader)
                proto_batch = next(proto_iter)
                
            p_raw = proto_batch['own_feat'].to(device)
            p_range_mask = proto_batch['range_mask'].to(device)

            p_mask = p_range_mask
            p_labels = proto_batch['kmeans_label'].to(device).long()
            p_dids = proto_batch['dataset_id'].to(device)
            
            _, __, p_seq_logits = model.forward_video(p_raw)
            p_logits = torch.mean(p_seq_logits, dim=1)
            
            loss_proto_total , prototype_status = criterion_proto(p_logits, p_labels, p_dids, p_mask)
            
            loss_cross_total = torch.tensor(0.0, device=device)
            val_lcross = 0.0
    
            if (batch_idx + 1) % cfg.cross_gap == 0:
                loss_cross_total, val_lcross = global_cross.run_training_step(
                model, train_ds, device
            )
            total_loss = loss_temporal_total + loss_proto_total + loss_cross_total
            total_loss.backward()
            
            optimizer.step()    
            val_lalign = temporal_stats['loss_align']
            val_lrc = temporal_stats['loss_recon']
            val_lintra= prototype_status['loss_intra']
            val_linter = prototype_status['loss_iner']
            val_lcross = val_lcross
        
            if (batch_idx + 1) % 20 == 0:
                val_act = final_mask.float().mean().item()
                val_bit1 = q_hash.gt(0).float().mean().item()
            
                msg = (
                    f"Epoch[{epoch+1}/{cfg.num_epochs}] "
                    f"Step[{batch_idx+1}/{len(rank_loader)}] "
                    f"| L_Align: {val_lalign:.4f} "
                    f"| L_Recon: {val_lrc:.4f}"
                    f"| L_intra: {val_lintra:.4f} "
                    f"| L_Cross: {val_lcross:.4f} "
                    f"| L_inter: {val_linter:.4f} "
                    f"| Gate_Act: {val_act:.4f}"
                    f"| Bit-1_ratio: {val_bit1:.4f}"
                )   
                logger.info(msg)
            
        scheduler.step()
        logger.info(f"--- Epoch {epoch+1} Done. Current LR: {optimizer.param_groups[0]['lr']:.8e} ---")
        
        if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.num_epochs or (epoch + 1) == 1:
             logger.info(f"--- Running Evaluation at Epoch {epoch+1} ---")
             model.eval()
             with torch.no_grad():
                results = evaluate_all(cfg, model, logger, args, prompt_masks=train_ds.prompt_masks )
                current_avg_mAP = results
                logger.info(f"Current average_mAP: {current_avg_mAP:.4f} (at Epoch {epoch}) ")

             save_path = os.path.join(cfg.save_path, f'your_txt_name_{epoch+1}.pth')
             torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'avg_mAP': current_avg_mAP}, save_path)
             logger.info(f"Checkpoint saved: {save_path}")

             
if __name__ == "__main__":
    train()
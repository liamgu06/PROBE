import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset.Probe_dataset import topk_TrainDataset


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def dcl(out_1, out_2, batch_size, temperature=0.5, tau_plus=0.1):
    out_1 = F.normalize(out_1, dim=1)
    out_2 = F.normalize(out_2, dim=1)
    
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = get_negative_mask(batch_size).to(out_1.device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    if True:
        N = batch_size * 2 - 2
        Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
        Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
    else:
        Ng = neg.sum(dim=-1)

    loss = (- torch.log(pos / (pos + Ng) )).mean()
    return loss


class TemporalLoss(nn.Module):
    def __init__(self, cfg): 
        super().__init__()
        self.cfg = cfg
        self.temp = self.cfg.contra_temp
        self.tau_plus = self.cfg.tau_plus
        self.lambda_recon = self.cfg.lambda_recon
        self.lambda_contra = self.cfg.lambda_contra

    def forward(self, q_hash, view1_hash=None, view2_hash=None, recon_1=None, target_1=None, recon_2=None, target_2=None):
        
        if view1_hash is not None and view2_hash is not None:
            v1 = F.normalize(view1_hash, p=2, dim=1)
            v2 = F.normalize(view2_hash, p=2, dim=1)
            L_align = dcl(v1, v2, 
                        batch_size=v1.size(0), 
                        temperature=self.temp, 
                        tau_plus=self.tau_plus)
        else:
            print("Warning: Contrastive loss not computed due to missing view logits.")   
        
        L_recon = torch.tensor(0.0, device=q_hash.device)
        
        if recon_1 is not None and target_1 is not None:
            mask_lr_1 = F.mse_loss(recon_1, target_1)
            mask_lr_2 = F.mse_loss(recon_2, target_2)
            L_recon = mask_lr_1 + mask_lr_2 
            
        total_loss = self.lambda_contra * L_align + self.lambda_recon * L_recon
        return total_loss, {
            "loss_align": L_align.item(),
            "loss_recon": L_recon.item(),
        }



class SubspaceGeometryLoss(nn.Module):
    def __init__(self, total_videos, cfg, device='cuda'):
        super().__init__()
        self.num_clusters_list = cfg.num_clusters_list
        self.mu = cfg.mu
        self.tau_c = cfg.tau_c
        self.tau_p = cfg.tau_p
        hash_bits = cfg.hash_bits
        self.pp_weight = cfg.pp_weight
        self.register_buffer("ema_feats", torch.zeros(total_videos, hash_bits))
        self.register_buffer("ema_initialized", torch.zeros(total_videos, dtype=torch.bool))
        for i,k in enumerate(self.num_clusters_list):
            self.register_buffer(f"prototypes_{i}", torch.zeros(k, hash_bits))

    def update_ema(self, indices, current_feats):
        with torch.no_grad():
            current_feats = current_feats.detach()
            old_ema = self.ema_feats[indices]
            not_init = ~self.ema_initialized[indices]
            if not_init.any():
                old_ema[not_init] = current_feats[not_init]
                self.ema_initialized[indices] = True
            new_ema = self.mu * old_ema + (1.0 - self.mu) * current_feats
            self.ema_feats[indices] = new_ema
        
        
    def forward(self, q_logits, labels, dataset_ids, final_mask):
        q_raw = F.normalize(q_logits, p=2, dim=1)
        with torch.no_grad():
            batch_size = len(labels)
            for i in range(batch_size):
                d_id = dataset_ids[i].item()
                label_idx = labels[i].item() 
                proto_bank = getattr(self, f"prototypes_{d_id}")
                
                new_proto = self.mu * proto_bank[label_idx] + (1 - self.mu) * q_raw[i].detach()
                proto_bank[label_idx] = new_proto
                
            for d_id in range(len(self.num_clusters_list)):
                proto_bank = getattr(self, f"prototypes_{d_id}")
                setattr(self, f"prototypes_{d_id}", F.normalize(proto_bank, p=2, dim=1))
        
        total_loss = 0.0
        loss_intra = 0.0
        loss_iner = 0.0
        valid_cnt = 0
        for d_id in range(len(self.num_clusters_list)):
            dataset_selector = (dataset_ids == d_id)
            if not dataset_selector.any(): 
                continue
            
            sub_q_raw = q_logits[dataset_selector]
            sub_label = labels[dataset_selector]
            
            proto_bank = getattr(self, f"prototypes_{d_id}")
            sub_protos_raw = proto_bank.detach()
            
            sub_mask = final_mask[dataset_selector]
            sub_q_masked = F.normalize(sub_q_raw * sub_mask, p=2, dim=1)
            
            protos_expanded = sub_protos_raw.unsqueeze(0)
            mask_expanded = sub_mask.unsqueeze(1)
            protos_masked_3d = protos_expanded * mask_expanded
            sub_protos_masked = F.normalize(protos_masked_3d, p=2, dim=2)
                   
            logits = (sub_q_masked.unsqueeze(1) * sub_protos_masked).sum(dim=2) / self.tau_c
            loss_intra += F.cross_entropy(logits, sub_label)
            
            unique_labels = torch.unique(sub_label)
            if len(unique_labels) >= 2:
                batch_protos_list = []
                for k in unique_labels:
                    k_proto = sub_q_masked[sub_label == k].mean(dim=0)
                    k_proto = F.normalize(k_proto, p=2, dim=0)
                    batch_protos_list.append(k_proto)
                batch_protos = torch.stack(batch_protos_list)
                sim_matrix = torch.matmul(batch_protos, batch_protos.t()) / self.tau_p
                diag_mask = torch.eye(len(unique_labels), device=q_logits.device)
                sum_exp_neg = (torch.exp(sim_matrix) * (1 - diag_mask)).sum(dim=1)
                loss_iner += torch.log(sum_exp_neg + 1e-6).mean()
            valid_cnt += 1

        if valid_cnt > 0:
            loss_intra /= valid_cnt
            loss_iner /= valid_cnt
            
        total_loss = loss_intra + self.pp_weight * loss_iner
        return total_loss , {
            "loss_intra": loss_intra, 
            "loss_iner" : loss_iner
        }


class CrossConsistencyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tau_s = getattr(cfg, 'tau_s', 0.2)         
        self.lambda_cross = getattr(cfg, 'LAMBDA_CROSS', 1)
        self.batch_size = 128

    def forward(self, s_logits, t_feat, mask):
        z_s = s_logits * mask 
        z_s = F.normalize(z_s, p=2, dim=1)
        
        z_t = F.normalize(t_feat, p=2, dim=1)
        
        sim_s = torch.mm(z_s, z_s.t()) / self.tau_s
        sim_t = torch.mm(z_t, z_t.t()) / self.tau_s
        
        log_prob_s = F.log_softmax(sim_s, dim=1)
        prob_t     = F.softmax(sim_t, dim=1)
        
        log_prob_t = F.log_softmax(sim_t, dim=1)
        prob_s = F.softmax(sim_s, dim=1)
        
        kl_st = F.kl_div(log_prob_s, prob_t, reduction='batchmean')
        kl_ts = F.kl_div(log_prob_t, prob_s, reduction='batchmean')
        consistency_loss = 0.5 * (kl_st + kl_ts)
        
        return consistency_loss
    
    
    def run_training_step(self, model, train_dataset, device):
        batch_data = train_dataset.sample_cross_batch(batch_size=self.batch_size)
        
        s_feat = batch_data['student_feat'].to(device)
        t_feat = batch_data['teacher_feat'].to(device)
        c_mask = batch_data['cross_mask'].to(device)

        _, __, s_seq_logits = model.forward_video(s_feat)
        s_logits = torch.mean(s_seq_logits, dim=1) 

        loss_val = self.forward(s_logits, t_feat, c_mask)
        
        return loss_val, loss_val.item()
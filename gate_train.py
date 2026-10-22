import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import os
import sys
    
from dataset.Probe_dataset import GatePretrainDataset
from model.Probe_model import ProbeModel 
from configs.Probe_config import TrainingConfig
from utils.env import set_log 


def evaluate_gate(model, dataloader, device, logger, epoch, cfg):
    model.eval()
    all_mae, all_acc1, all_acck = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            prompt_tokens = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            m_p, u_logits, z_p = model.forward_prompt(prompt_tokens)
            
            sim_matrix_teacher = torch.mm(z_p, z_p.t())
            
            gate_norm = F.normalize(u_logits, p=2, dim=1)
            sim_matrix_gate = torch.mm(gate_norm, gate_norm.t())
            
            B = sim_matrix_teacher.shape[0]
            eye_mask = torch.eye(B, device=device).bool()
            off_diag_mask = ~eye_mask
            
            mae = (torch.abs(sim_matrix_teacher - sim_matrix_gate)[off_diag_mask]).mean().item()
            all_mae.append(mae)
            
            s_t = sim_matrix_teacher.clone().masked_fill_(eye_mask, -1.0)
            s_g = sim_matrix_gate.clone().masked_fill_(eye_mask, -1.0)
            
            acc1 = (s_t.max(dim=1)[1] == s_g.max(dim=1)[1]).float().mean().item()
            all_acc1.append(acc1)
            
            k = 10
            _, idx_t = torch.topk(s_t, k, dim=1) 
            _, idx_g = torch.topk(s_g, k, dim=1)
            intersect = (idx_g.unsqueeze(2) == idx_t.unsqueeze(1)).any(dim=-1).any(dim=-1)
            acck = intersect.float().mean().item()
            all_acck.append(acck)
            
    avg_mae = sum(all_mae) / len(all_mae)
    avg_acc1 = sum(all_acc1) / len(all_acc1)
    avg_acck = sum(all_acck) / len(all_acck)
    
    logger.info(f"EVAL [Epoch {epoch}] | MAE: {avg_mae:.4f} | Acc@1: {avg_acc1:.4f} | Acc@k: {avg_acck:.4f}")
    
    model.train()
    return avg_acck
    
    
def train_gate_only():
    cfg = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg.log_path = "path/to/logs"
    cfg.save_path = "path/to/checkpoints"
    if not os.path.exists(cfg.log_path): os.makedirs(cfg.log_path)
    if not os.path.exists(cfg.save_path): os.makedirs(cfg.save_path)
        
    logger = set_log(cfg, 'gate_train.log') 
    model = ProbeModel(cfg, hash_bits=cfg.hash_bits).to(device)
    
    for name, param in model.named_parameters():
        if "gate_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    
    dataset = GatePretrainDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    mse_loss = nn.MSELoss()
    
    logger.info("Start Training Gate Head...")
    
    epochs = 100
    for epoch in range(epochs):
        model.train() 
        
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            prompt_tokens = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            optimizer.zero_grad()
            
            m_p, u_logits, clip_emb = model.forward_prompt(prompt_tokens)
            
            gate_norm = F.normalize(u_logits, p=2, dim=1) 
            sim_matrix_gate = torch.mm(gate_norm, gate_norm.t())
            sim_matrix_clip = torch.mm(clip_emb, clip_emb.t())
            loss_structure = mse_loss(sim_matrix_gate, sim_matrix_clip)
            
            usage = m_p.mean(dim=0) + 1e-9
            p = usage / usage.sum()
            entropy = - (p * torch.log(p)).sum()
            
            max_entropy = torch.log(torch.tensor(cfg.hash_bits).float()).to(device)
            loss_balance = 1.0 - (entropy / max_entropy)

            total_loss = loss_structure + cfg.lambda_balance * loss_balance
            
            total_loss.backward()
            optimizer.step()
            if (step + 1) % 20 == 0:
                logger.info(f"Epoch[{epoch+1}/{epochs}] Step[{step+1}/{len(dataloader)}] | L_total: {total_loss.item():.4f} | L_stru: {loss_structure.item():.4f} | L_balance: {loss_balance.item():.4f}")

        scheduler.step()
        if (epoch + 1) % 2 == 0:
            
            evaluate_gate(model, dataloader, device, logger, epoch + 1, cfg)
            
            save_path = os.path.join(cfg.save_path, f"pretrained_gate_epoch_{epoch+1}.pth")
            torch.save(model.gate_head.state_dict(), save_path)
            logger.info(f"Gate Head saved to {save_path}")


if __name__ == "__main__":
    train_gate_only()
    

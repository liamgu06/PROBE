import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import math
import warnings
import numpy as np
from functools import partial
import math
from typing import Optional
from thop import profile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from mamba_ssm import Mamba
from transformers import CLIPTokenizer, CLIPTextModel

from .RWKV.vrwkv6 import RWKVBLOCK
from .RetNet.retention import MultiScaleRetention
from .Transformer.transformerBlock import TransformerBlock
import h5py


def __call_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor

    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p = {}'.format(self.drop_prob)

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class Block(nn.Module):
    """
    Bidirectional Mamba to Extract Semantic Information
    """
    def __init__(self, dim,  mlp_ratio=4.,drop=0., layer_id=0.,n_layer=12,
                    drop_path=0., init_values=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.mlp_ratio = mlp_ratio
        self.layer_id = layer_id
        self.n_head = dim//64
        self.n_layer = n_layer
        self.init_values = init_values
        self.rwkv_drop_path = drop_path
        self.mamba1 = Mamba(
                d_model=dim,d_state=16,d_conv=4,expand=2
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.dim = dim
        self.mamba2 =Mamba(
                d_model=dim,d_state=16,d_conv=4,expand=2
        )
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None


class ProbeEncoder(nn.Module):
    def __init__(self,feature_num=4096, embed_dim=256, max_frame=25, nbits=64, depth=12, mlp_ratio=4.,
                    drop_rate=0.,drop_path_rate=0., 
                    norm_layer=nn.LayerNorm, init_values=None, use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        num_patches = max_frame
        self.num_patches = num_patches
        self.nbits = nbits
        self.patch_embed = nn.Linear(feature_num, embed_dim)
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
        
    def forward_features(self, x, mask, model_name="Probe"):
        x = self.patch_embed(x)
        
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) 

        for blk in self.blocks:
            x_vis = blk(x=x_vis, model_name = model_name)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask, model_name="Probe"):
        x = self.forward_features(x, mask, model_name)
        x = self.head(x)
        return x
    

class ProbeDecoder(nn.Module):
    def __init__(self, feature_num=2048, embed_dim=256, max_frame=25, depth=12,
                  mlp_ratio=4., drop_rate=0., 
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        num_patches = max_frame
        self.num_patches = num_patches

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,mlp_ratio=mlp_ratio, 
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, feature_num)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, return_token_num, model_name="Probe"):
        for blk in self.blocks:
            x = blk(x=x, model_name=model_name)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) 
        else:
            x = self.head(self.norm(x))

        return x
    

class ProbeModel(nn.Module):
    def __init__(self, 
                 cfg,
                 vision_tower_name = None,    
                 hash_bits=512):
        super().__init__()
        
        self.cfg = cfg
        if vision_tower_name is None:
            vision_tower_name = cfg.local_clip_path
        
        self.video_encoder = ProbeEncoder(
            feature_num=cfg.feature_size, 
            embed_dim=cfg.hidden_size, 
            max_frame=cfg.max_frames,
            nbits=cfg.hash_bits,
            depth=cfg.encoder_depth,
            mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            norm_layer=nn.LayerNorm,
            init_values=cfg.init_values,
            use_learnable_pos_emb=cfg.use_learnable_pos_emb
        )
        
        self.video_feature_dim = cfg.hidden_size 
        
        self.decoder_embed_dim = cfg.decoder_embed_dim 
        
        self.hash_head = nn.Linear(self.video_feature_dim, hash_bits)
        self.encoder_to_decoder = nn.Linear(hash_bits, self.decoder_embed_dim, bias=False)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, cfg.max_frames, self.decoder_embed_dim))
        trunc_normal_(self.decoder_pos_embed, std=.02)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_embed_dim))
        trunc_normal_(self.mask_token, std=.02)

        self.decoder = ProbeDecoder(
            feature_num=cfg.feature_size, 
            embed_dim=cfg.decoder_embed_dim, 
            max_frame=cfg.max_frames,
            depth=cfg.decoder_depth,
            mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            norm_layer=nn.LayerNorm,
            init_values=cfg.init_values)
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            vision_tower_name,
            torch_dtype=torch.float32,
        )
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.text_feature_dim = self.text_encoder.config.hidden_size

        self.gate_head = nn.Linear(self.text_feature_dim, hash_bits)

    def forward_video(self, video_features, mask=None):
        
        if mask is None:
            mask = torch.zeros((video_features.size(0), video_features.size(1)), 
                               dtype=torch.bool, device=video_features.device)
        
        z_sequence = self.video_encoder(video_features, mask, model_name=self.cfg.model_name)
        h_logits_seq = self.hash_head(z_sequence) 
        
        if self.training:
            b_v_hard = torch.sign(h_logits_seq) 
            b_v_soft = torch.tanh(h_logits_seq)
            b_v_seq = b_v_hard + b_v_soft - b_v_soft.detach() 
        else:
            b_v_seq = torch.sign(h_logits_seq)
        
        output = None
        
        if self.training and mask.any():
            x_vis = self.encoder_to_decoder(b_v_seq)
            B,N,C = x_vis.shape
            expand_pos_embed = self.decoder_pos_embed.expand(B, -1, -1).type_as(video_features).to(video_features.device)
            C = x_vis.shape[-1]
            pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
            pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
            x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
            output = self.decoder(x_full, pos_emd_mask.shape[1], model_name=self.cfg.model_name)

        return output, b_v_seq, h_logits_seq
    
    def inference_video(self, video_features):
        B, T, _ = video_features.shape
        device = video_features.device
        mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        z_sequence = self.video_encoder(video_features, mask, model_name=self.cfg.model_name)
        z_v = torch.mean(z_sequence, dim=1) 
        
        h_logits = self.hash_head(z_v)
        
        z = self.hash_head(z_v)
        
        b_v = torch.sign(h_logits) 
 
        return b_v
    
    @staticmethod
    def gumbel_topk(logits, k, tau=1.0, hard=True, deterministic=False):
        if deterministic:
            y = logits / tau 
        else:
            unif = torch.rand_like(logits)
            gumbels = -torch.log(-torch.log(unif + 1e-9) + 1e-9)
            y = (logits + gumbels) / tau
        
        _, indices = torch.topk(y, k, dim=-1)
        y_hard = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
        
        y_soft = torch.sigmoid(y) 
        if hard:
            return y_hard - y_soft.detach() + y_soft
        return y_soft
    

    def forward_prompt(self, prompt_tokens):
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=prompt_tokens['input_ids'], 
                attention_mask=prompt_tokens['attention_mask']
            )
            
            last_hidden = outputs.last_hidden_state  
            eos_indices = prompt_tokens['attention_mask'].sum(dim=1) - 1
            batch_indices = torch.arange(last_hidden.shape[0], device=last_hidden.device)
            z_p_raw = last_hidden[batch_indices, eos_indices] 
            z_p = torch.nn.functional.normalize(z_p_raw, p=2, dim=-1)

        u_logits = self.gate_head(z_p.float()) 
        
        m_p = self.gumbel_topk(u_logits, k=self.cfg.gate_topk, tau=self.cfg.k_tau, hard=True, deterministic=not self.training)
        
        u_logits = u_logits * m_p

        return m_p, u_logits,z_p

    def forward(self, video, prompt_tokens, video_mask=None):
        b_v, h_logits = self.forward_video(video, mask=video_mask)
        m_p, u_logits, __ = self.forward_prompt(prompt_tokens)
        
        return {
            "hash_code": b_v,       
            "mask": m_p,           
            "hash_logits": h_logits,
            "mask_logits": u_logits
        }
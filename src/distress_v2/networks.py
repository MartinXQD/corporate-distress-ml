"""Legacy-compatible capacity and official numerical/TabM/FT modules."""
import math
import numpy as np
import torch
from torch import nn

class LegacyResidual(nn.Module):
    """Re-expressed legacy architecture; original team source is credited."""
    def __init__(self,d,width=192,depth=4,dropout=.25):
        super().__init__()
        self.input_projection=nn.Sequential(nn.Linear(d,width),nn.GELU(),nn.Dropout(dropout))
        self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,width),nn.Dropout(dropout)) for _ in range(depth)])
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
    def forward(self,x):
        h=self.input_projection(x)
        for block in self.blocks:h=h+block(h)
        return self.head(h).squeeze(-1)

class RegularMLP(nn.Module):
    def __init__(self,d,width=128,depth=2,dropout=.15):
        super().__init__();layers=[]
        for _ in range(depth):
            layers.extend([nn.Linear(d,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(dropout)]);d=width
        self.layers=nn.Sequential(*layers,nn.Linear(d,1))
    def forward(self,x):return self.layers(x).squeeze(-1)

class NumericNetwork(nn.Module):
    def __init__(self,d,n_cont,cfg,x_for_bins=None,bins=None):
        super().__init__();self.n_cont=n_cont;self.family=cfg['family'];self.embedding=None
        mode=cfg.get('embedding','none');embed_dim=cfg.get('embedding_dim',8)
        if mode!='none':
            import rtdl_num_embeddings as ne
            if mode=='piecewise':
                if bins is None:bins=ne.compute_bins(torch.as_tensor(x_for_bins[:,:n_cont]),n_bins=cfg.get('bins',16))
                self.embedding=ne.PiecewiseLinearEmbeddings(bins,embed_dim,activation=False,version='B')
            elif mode=='periodic':
                self.embedding=ne.PeriodicEmbeddings(n_cont,embed_dim,n_frequencies=16,frequency_init_scale=cfg.get('frequency_scale',.1),activation=True,lite=True)
            else:raise ValueError(mode)
            d=n_cont*embed_dim+(d-n_cont)
        if self.family=='tabm':
            from tabm import TabM
            self.body=TabM.make(n_num_features=d,d_out=1,k=cfg.get('k',16),n_blocks=cfg.get('depth',2),d_block=cfg.get('width',128),dropout=cfg.get('dropout',.1))
        elif self.family=='official_ft':
            from rtdl_revisiting_models import FTTransformer
            self.body=FTTransformer(n_cont_features=d,cat_cardinalities=[],d_out=1,n_blocks=cfg.get('depth',2),d_block=cfg.get('token_dim',32),attention_n_heads=8,attention_dropout=cfg.get('dropout',.1),ffn_d_hidden=None,ffn_d_hidden_multiplier=4/3,ffn_dropout=cfg.get('dropout',.1),residual_dropout=0.)
        elif cfg.get('backbone','mlp')=='residual' or self.family=='legacy_residual':
            self.body=LegacyResidual(d,cfg.get('width',192),cfg.get('depth',4),cfg.get('dropout',.25))
        elif self.family=='v1_residual':
            from distress_ml.neural import TabularNet
            self.body=TabularNet('residual_mlp',d,cfg)
        else:self.body=RegularMLP(d,cfg.get('width',128),cfg.get('depth',2),cfg.get('dropout',.15))
    def forward(self,x):
        if self.embedding is not None:
            continuous=self.embedding(x[:,:self.n_cont]).flatten(1)
            x=torch.cat([continuous,x[:,self.n_cont:]],dim=1)
        if self.family=='official_ft':return self.body(x,None).squeeze(-1)
        out=self.body(x)
        return out.squeeze(-1) if self.family=='tabm' else out

def logits(model,x,batch=2048):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch):out.append(model(torch.as_tensor(x[start:start+batch],dtype=torch.float32)).cpu().numpy())
    return np.concatenate(out)

def probabilities(model,x):
    from scipy.special import expit
    z=logits(model,x)
    return expit(z).mean(axis=1) if z.ndim==2 else expit(z)


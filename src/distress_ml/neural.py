"""Small independently implemented PyTorch tabular networks and training loop."""
from __future__ import annotations
import copy
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(2)
except RuntimeError:
    pass

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

class ResidualBlock(nn.Module):
    def __init__(self,width,dropout):
        super().__init__()
        self.block=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.ReLU(),nn.Dropout(dropout),nn.Linear(width*2,width),nn.Dropout(dropout))
    def forward(self,x):
        return x+self.block(x)

class TabularNet(nn.Module):
    def __init__(self,family,n_features,config):
        super().__init__()
        self.family=family
        w=config.get('width',64); drop=config.get('dropout',.15); depth=config.get('depth',2)
        if family=='mlp':
            layers=[]; n=n_features
            for _ in range(depth):
                layers.extend([nn.Linear(n,w),nn.ReLU(),nn.Dropout(drop)]); n=w
            self.network=nn.Sequential(*layers,nn.Linear(w,1))
        elif family=='residual_mlp':
            self.network=nn.Sequential(nn.Linear(n_features,w),*[ResidualBlock(w,drop) for _ in range(depth)],nn.LayerNorm(w),nn.ReLU(),nn.Linear(w,1))
        elif family=='ft_transformer':
            d=config.get('token_dim',16)
            self.token_weight=nn.Parameter(torch.randn(n_features,d)*.02)
            self.token_bias=nn.Parameter(torch.zeros(n_features,d))
            self.cls=nn.Parameter(torch.randn(1,1,d)*.02)
            layer=nn.TransformerEncoderLayer(d_model=d,nhead=2,dim_feedforward=d*2,dropout=drop,activation='relu',batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,num_layers=depth,enable_nested_tensor=False)
            self.head=nn.Sequential(nn.LayerNorm(d),nn.ReLU(),nn.Linear(d,1))
        elif family=='gru':
            self.gru=nn.GRU(n_features,w,num_layers=1,batch_first=True)
            self.head=nn.Sequential(nn.Dropout(drop),nn.Linear(w,1))
        else:
            raise ValueError(family)
    def forward(self,x):
        if self.family=='gru':
            output,_=self.gru(x)
            return self.head(output[:,-1]).squeeze(-1)
        if self.family=='ft_transformer':
            tokens=x.unsqueeze(-1)*self.token_weight+self.token_bias
            tokens=torch.cat([self.cls.expand(len(x),-1,-1),tokens],dim=1)
            return self.head(self.encoder(tokens)[:,0]).squeeze(-1)
        return self.network(x).squeeze(-1)

def predict_logits(model,x,batch_size=2048):
    model.eval(); scores=[]
    with torch.inference_mode():
        for i in range(0,len(x),batch_size):
            scores.append(model(torch.as_tensor(x[i:i+batch_size],dtype=torch.float32)).cpu().numpy())
    return np.concatenate(scores)

def fit_neural(family,x,y,config,seed,pos_weight=1,stop=None,epochs=None):
    seed_everything(seed)
    model=TabularNet(family,x.shape[-1],config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('learning_rate',.001),weight_decay=config.get('weight_decay',.0001))
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(y,dtype=torch.float32)),batch_size=config.get('batch_size',1024),shuffle=True,generator=generator,num_workers=0)
    history=[]; best=-np.inf; best_epoch=1; best_state=None; stale=0
    limit=epochs or config.get('max_epochs',25)
    started=time.perf_counter()
    for epoch in range(1,limit+1):
        model.train(); total=0.; n=0
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(xb)
            assert logits.shape==yb.shape
            loss=criterion(logits,yb)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite BCE loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite gradient')
            nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step(); total+=loss.item()*len(yb); n+=len(yb)
        tr=predict_logits(model,x)
        row=dict(epoch=epoch,train_loss=total/n,train_AP=float(average_precision_score(y,tr)),elapsed_seconds=time.perf_counter()-started)
        if stop is not None:
            sx,sy=stop; sl=predict_logits(model,sx)
            with torch.no_grad():
                row['stop_loss']=float(nn.functional.binary_cross_entropy_with_logits(torch.tensor(sl),torch.tensor(sy,dtype=torch.float32)))
            row['stop_AP']=float(average_precision_score(sy,sl))
            if row['stop_AP']>best+1e-6:
                best=row['stop_AP']; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); stale=0
            else: stale+=1
        history.append(row)
        if stop is not None and stale>=config.get('patience',4): break
    if stop is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch=limit
    model.eval()
    return model,history,best_epoch

"""Train-only numerical transforms; mask columns bypass quantile maps."""
import warnings
import numpy as np
from sklearn.preprocessing import QuantileTransformer

class NumericTransform:
    def __init__(self,kind='std01',indicators=True):self.kind=kind;self.indicators=indicators
    def fit(self,x):
        self.columns=list(x);a=x.to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');self.median=np.nan_to_num(np.nanmedian(a,axis=0))
        self.numeric=np.array([not set(np.unique(v[np.isfinite(v)])).issubset({0.,1.}) for v in a.T])
        self.missing=np.flatnonzero(np.isnan(a).any(axis=0)) if self.indicators else np.array([],int)
        z=np.where(np.isfinite(a),a,self.median)
        q=.01 if self.kind=='std01' else .005
        self.low,self.high=np.quantile(z,[q,1-q],axis=0) if self.kind in ['std01','std005'] else (np.full(a.shape[1],-np.inf),np.full(a.shape[1],np.inf))
        self.low[~self.numeric]=-np.inf;self.high[~self.numeric]=np.inf
        z=np.clip(z,self.low,self.high)
        self.center=np.zeros(z.shape[1]);self.scale=np.ones(z.shape[1]);self.quantile=None
        if self.kind in ['std01','std005','robust','quantile']:
            if self.kind=='quantile':
                self.quantile=QuantileTransformer(n_quantiles=min(1000,len(z)),output_distribution='normal',subsample=None,random_state=42).fit(z[:,self.numeric])
                self.train_min=z[:,self.numeric].min(axis=0);self.train_max=z[:,self.numeric].max(axis=0)
            elif self.kind=='robust':
                self.center[self.numeric]=np.median(z[:,self.numeric],axis=0)
                v=np.quantile(z[:,self.numeric],[.25,.75],axis=0);self.scale[self.numeric]=v[1]-v[0]
            else:
                self.center[self.numeric]=z[:,self.numeric].mean(axis=0);self.scale[self.numeric]=z[:,self.numeric].std(axis=0)
        self.scale=np.where(self.scale>0,self.scale,1.)
        self.n_cont=int(self.numeric.sum());self.binary=np.flatnonzero(~self.numeric)
        self.output_columns=[self.columns[i] for i in np.flatnonzero(self.numeric)]+[self.columns[i] for i in self.binary]+[self.columns[i]+'_missing' for i in self.missing]
        return self
    def transform(self,x):
        assert list(x)==self.columns
        a=x.to_numpy(float);missing=(~np.isfinite(a)).astype(float)
        z=np.where(np.isfinite(a),a,self.median)
        if self.kind=='native':z=np.where(np.isfinite(a),a,np.nan)
        z=np.clip(z,self.low,self.high)
        if self.quantile is not None:z[:,self.numeric]=self.quantile.transform(z[:,self.numeric])
        else:z=(z-self.center)/self.scale
        z=np.column_stack([z[:,self.numeric],z[:,self.binary],missing[:,self.missing]])
        if self.kind!='native':assert np.isfinite(z).all()
        return z.astype(np.float32)
    def saturation(self,x):
        if self.quantile is None:return 0.
        a=x.to_numpy(float)[:,self.numeric]
        return float(np.mean((a<self.train_min)|(a>self.train_max)))

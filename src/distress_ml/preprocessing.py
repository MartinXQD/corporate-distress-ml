"""Train-role-only imputation, optional clipping, scaling and resampling."""
import warnings
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.neighbors import NearestNeighbors

class Preprocessor:
    def __init__(self, clip=True, scale=True):
        self.clip=clip
        self.scale=scale

    def fit(self, x):
        self.input_columns=list(x.columns)
        a=x.to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            self.medians=np.nanmedian(a,axis=0)
        self.medians=np.nan_to_num(self.medians)
        z=np.where(np.isfinite(a),a,self.medians)
        self.lower,self.upper=np.quantile(z,[.005,.995],axis=0) if self.clip else (np.full(z.shape[1],-np.inf),np.full(z.shape[1],np.inf))
        z=np.clip(z,self.lower,self.upper)
        # Masks are offered for every column, then constants/duplicates are removed in fit only.
        z=np.column_stack([z,(~np.isfinite(a)).astype(float)])
        names=self.input_columns+[c+"_missing" for c in self.input_columns]
        frame=pd.DataFrame(z,columns=names)
        variable=frame.nunique(dropna=False).to_numpy()>1
        duplicate=frame.T.duplicated().to_numpy()
        self.keep=np.flatnonzero(variable & ~duplicate)
        self.columns=[names[i] for i in self.keep]
        z=z[:,self.keep]
        self.means=z.mean(axis=0) if self.scale else np.zeros(z.shape[1])
        self.stds=z.std(axis=0) if self.scale else np.ones(z.shape[1])
        self.stds=np.where(self.stds>0,self.stds,1)
        return self

    def transform(self,x):
        assert list(x.columns)==self.input_columns, "Feature order mismatch"
        a=x.to_numpy(dtype=float)
        z=np.where(np.isfinite(a),a,self.medians)
        z=np.clip(z,self.lower,self.upper)
        z=np.column_stack([z,(~np.isfinite(a)).astype(float)])[:,self.keep]
        z=(z-self.means)/self.stds
        if not np.isfinite(z).all(): raise ValueError("Nonfinite transformed input")
        return z.astype(np.float32)

    def fit_transform(self,x):
        return self.fit(x).transform(x)

def positive_weight(y,alpha):
    pos=int(np.sum(y)); neg=len(y)-pos
    return (neg/pos)**alpha

def resample_train(x,y,method,seed):
    if method=="none": return x,y
    cls={"ros":RandomOverSampler,"smote":SMOTE}[method]
    # Exact KD-tree neighbors avoid the bundled OpenMP pairwise dispatcher,
    # which crashes on this macOS Python environment when PyTorch is loaded.
    sampler=cls(random_state=seed,k_neighbors=NearestNeighbors(n_neighbors=6,algorithm='kd_tree',n_jobs=1)) if method=='smote' else cls(random_state=seed)
    return sampler.fit_resample(x,y)

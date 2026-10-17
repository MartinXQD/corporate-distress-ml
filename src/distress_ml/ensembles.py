"""One pre-registered equal-rank tree/ML-DL candidate, chosen only on OOF years."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .common import ART,PUBLIC,ROOT,dump,now

def annual_rank(years,scores):
    t=pd.DataFrame({'year':np.asarray(years),'score':np.asarray(scores)})
    return t.groupby('year').score.rank(method='average',pct=True).to_numpy()

def choose_ensemble(representatives):
    tables={}
    for kind,cid in representatives.items():
        paths=sorted((ART/'predictions').glob(f'dev_{cid}_D*.csv.gz'))
        tables[kind]=pd.concat([pd.read_csv(p) for p in paths]).sort_values('row_id').reset_index(drop=True)
    a,b=tables['ML'],tables['DL']
    assert np.array_equal(a.row_id,b.row_id) and np.array_equal(a.y_true,b.y_true)
    ml=annual_rank(a.year,a.raw_score); dl=annual_rank(b.year,b.raw_score)
    pred={'ML':ml,'DL':dl,'equal_rank_ensemble':.5*(ml+dl)}
    rows=[]
    for name,score in pred.items():
        for year in sorted(a.year.unique()):
            ix=a.year.to_numpy()==year
            rows.append(dict(candidate=name,year=int(year),AP=float(average_precision_score(a.y_true[ix],score[ix])),prevalence=float(a.y_true[ix].mean())))
    t=pd.DataFrame(rows); t.to_csv(PUBLIC/'ensemble_development.csv',index=False)
    m=t.groupby('candidate').AP.mean()
    choice=dict(registered_candidate='equal-rank ML + DL, fixed 0.5/0.5',chosen_on='development OOF macro-year AP only',selected=bool(m['equal_rank_ensemble']>max(m['ML'],m['DL'])),macro_year_AP=m.to_dict(),representatives=representatives,selected_at_utc=now())
    dump(ROOT/'configs/ensemble_choice.json',choice)
    return choice

class RankEnsemblePipeline:
    def __init__(self,ml,dl):
        self.ml=ml; self.dl=dl
        self.cfg=dict(family='equal_rank_ensemble',feature_set='ALL',seed=42,clip=True,sampling='none',selector='none',weight_alpha=0)
        self.columns=sorted(set(ml.columns or [])|set(dl.columns or []))
        self.selected=self.columns
    def predict_raw(self,df,all_features=None):
        from .features import build_features
        f=all_features if all_features is not None else build_features(df,'ALL')
        a=self.ml.predict_raw(df,f); b=self.dl.predict_raw(df,f)
        return .5*(annual_rank(df.year,a)+annual_rank(df.year,b))

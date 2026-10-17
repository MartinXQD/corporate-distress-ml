"""Selectors whose validation and all fitted transforms stay inside training years."""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from .common import fit_event,ART
from .preprocessing import Preprocessor
from .features import feature_group
from .classical import make_model

def select_features(x,df,strategy,seed,tag):
    if strategy=='none': return list(x.columns),[]
    end=int(df.year.max()); train=(df.year<end).to_numpy(); val=(df.year==end).to_numpy()
    y=df.y.to_numpy(); details=[]
    if strategy=='elastic_net':
        pp=Preprocessor(clip=True,scale=True).fit(x.loc[train]); a=pp.transform(x.loc[train]); b=pp.transform(x.loc[val])
        candidates=[]
        for c in [.01,.1]:
            for ratio in [.5,.9]:
                call=fit_event(tag+'/EN_inner',int(train.sum()),df.loc[train,'company_name'],C=c,l1_ratio=ratio,max_year=end-1)
                model=LogisticRegression(C=c,l1_ratio=ratio,penalty='elasticnet',solver='saga',max_iter=1500,tol=1e-3,random_state=seed)
                model.fit(a,y[train]); ap=average_precision_score(y[val],model.predict_proba(b)[:,1])
                joblib.dump(dict(model=model,preprocessor=pp),ART/f'models/selector_fit_{call:04d}.joblib',compress=3)
                candidates.append((ap,c,ratio)); details.append(dict(strategy=strategy,stage='inner_candidate',C=c,l1_ratio=ratio,AP=float(ap)))
        _,c,ratio=max(candidates)
        pp=Preprocessor(clip=True,scale=True).fit(x)
        call=fit_event(tag+'/EN_full',len(df),df.company_name,C=c,l1_ratio=ratio,max_year=end)
        model=LogisticRegression(C=c,l1_ratio=ratio,penalty='elasticnet',solver='saga',max_iter=1500,tol=1e-3,random_state=seed).fit(pp.transform(x),y)
        joblib.dump(dict(model=model,preprocessor=pp),ART/f'models/selector_fit_{call:04d}.joblib',compress=3)
        chosen=[]
        for name,coef in zip(pp.columns,model.coef_[0]):
            source=name.removesuffix('_missing')
            if abs(coef)>1e-7 and source in x: chosen.append(source)
            details.append(dict(strategy=strategy,stage='full_coefficient',feature=name,coefficient=float(coef),selected=abs(coef)>1e-7))
        return sorted(set(chosen)) or list(x.columns),details
    if strategy=='model_aware':
        groups={g:[c for c in x if feature_group(c)==g] for g in sorted({feature_group(c) for c in x})}
        importance={g:[] for g in groups}; rng=np.random.default_rng(seed)
        # Both cross-fitted ranking years precede the candidate-selection year.
        for score_year in [end-2,end-1]:
            tr=(df.year<score_year).to_numpy(); va=(df.year==score_year).to_numpy()
            pp=Preprocessor(clip=True,scale=False).fit(x.loc[tr]); a=pp.transform(x.loc[tr]); b=pp.transform(x.loc[va])
            call=fit_event(tag+'/permutation_rank',int(tr.sum()),df.loc[tr,'company_name'],max_year=score_year-1,score_year=score_year)
            m=make_model('xgboost',dict(n_estimators=70,max_depth=3),seed).fit(a,y[tr])
            joblib.dump(dict(model=m,preprocessor=pp),ART/f'models/selector_fit_{call:04d}.joblib',compress=3)
            baseline=average_precision_score(y[va],m.predict_proba(b)[:,1])
            for group,cols in groups.items():
                indices=[i for i,c in enumerate(pp.columns) if c.removesuffix('_missing') in cols]
                pert=b.copy(); order=rng.permutation(len(b)); pert[:,indices]=b[order][:,indices]
                drop=baseline-average_precision_score(y[va],m.predict_proba(pert)[:,1]); importance[group].append(float(drop))
                details.append(dict(strategy=strategy,stage='cross_fitted_importance',year=score_year,group=group,AP_drop=float(drop)))
        ranking=sorted(groups,key=lambda g:(-np.mean(importance[g]),g))
        candidates=[]
        for k in sorted(set([min(2,len(groups)),min(4,len(groups)),len(groups)])):
            cols=[c for c in x if feature_group(c) in ranking[:k]]
            pp=Preprocessor(clip=True,scale=False).fit(x.loc[train,cols])
            call=fit_event(tag+'/group_candidate',int(train.sum()),df.loc[train,'company_name'],top_groups=k,max_year=end-1)
            m=make_model('xgboost',dict(n_estimators=70,max_depth=3),seed).fit(pp.transform(x.loc[train,cols]),y[train])
            joblib.dump(dict(model=m,preprocessor=pp,columns=cols),ART/f'models/selector_fit_{call:04d}.joblib',compress=3)
            ap=average_precision_score(y[val],m.predict_proba(pp.transform(x.loc[val,cols]))[:,1])
            candidates.append((ap,-k,cols)); details.append(dict(strategy=strategy,stage='inner_candidate',top_groups=k,AP=float(ap)))
        return max(candidates,key=lambda a:(a[0],a[1]))[2],details
    raise ValueError(strategy)

"""Disjoint 2013 Platt fitting and 2014 fixed-threshold selection."""
import numpy as np
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .common import fit_event
from .evaluation import choose_threshold
from .splits import assert_disjoint,row_hash

def fit_calibration(p,fit_df,calibration,decision,features_cal=None,features_dec=None,tag='final'):
    assert_disjoint(fit_df,calibration,decision)
    assert fit_df.year.max()<=2012
    assert set(calibration.year)=={2013} and set(decision.year)=={2014}
    raw=p.predict_raw(calibration,features_cal)
    fit_event(tag+'/Platt2013',len(calibration),calibration.company_name,max_year=2013,row_hash=row_hash(calibration),family='Platt_sigmoid')
    cal=make_pipeline(StandardScaler(),LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000))
    cal.fit(raw.reshape(-1,1),calibration.y.to_numpy())
    p.calibrator=cal
    decraw=p.predict_raw(decision,features_dec)
    decscore=cal.predict_proba(decraw.reshape(-1,1))[:,1]
    p.threshold=choose_threshold(decision.y.to_numpy(),decscore)
    p.metadata.update(calibration_year=2013,calibration_row_hash=row_hash(calibration),decision_year=2014,decision_row_hash=row_hash(decision),fixed_threshold=p.threshold,calibration_coefficient=float(cal[-1].coef_[0,0]),budget_rule='Annual ceil(1/5/10 percent), label-free SHA256 tie break',history_window=3 if p.cfg['feature_set'] in ['S2','LAGS'] else 1)
    return p

def calibrated_predict(p,df,features=None):
    raw=p.predict_raw(df,features)
    return raw,p.calibrator.predict_proba(raw.reshape(-1,1))[:,1]

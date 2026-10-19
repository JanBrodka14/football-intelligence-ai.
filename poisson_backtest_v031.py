"""Independent, strictly season-held-out Poisson goals baseline; research, not betting advice."""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from backtest import load, features, scores, LABELS, TRAIN, TEST, ROOT

OUT = ROOT / 'output'
MAX_GOALS = 12

def goal_features(matches):
    # Two records per fixture, home scoring and away scoring.
    cats=[]; venue=[]; goals=[]
    for m in matches:
        cats.extend([(m['league']+'|'+m['home'],m['league']+'|'+m['away']),
                     (m['league']+'|'+m['away'],m['league']+'|'+m['home'])])
        venue.extend([1,0]);goals.extend([m['hg'],m['ag']])
    return cats,np.array(venue),np.array(goals)

def matrix(enc,cats,venue):
    import scipy.sparse as sp
    return sp.hstack([enc.transform(cats),sp.csr_matrix(venue.reshape(-1,1))],format='csr')

def probs(h,a):
    # Include tail mass in final bin to ensure probabilities sum to one.
    ph=poisson.pmf(np.arange(MAX_GOALS),h);pa=poisson.pmf(np.arange(MAX_GOALS),a)
    ph=np.r_[ph,max(0,1-ph.sum())];pa=np.r_[pa,max(0,1-pa.sum())]
    joint=np.outer(ph,pa)
    H=float(np.tril(joint,-1).sum());D=float(np.trace(joint));A=float(np.triu(joint,1).sum())
    under=float(joint[:3,:3][np.add.outer(np.arange(3),np.arange(3))<=2].sum())
    btts=float((1-ph[0])*(1-pa[0]))
    return np.array([H,D,A])/joint.sum(),under,btts

def main():
    games=load()
    tr=[m for m in games if m['season'] in TRAIN];te=[m for m in games if m['season'] in TEST]
    if len(tr)<200 or len(te)<100: raise SystemExit('Insufficient data for holdout')
    # Tune regularization using 2023/24 -> 2024/25 ONLY.
    # alpha=1.0 was excessive for sparse team one-hot features, shrinking
    # team-specific attack/defense coefficients almost to zero.
    inner_train=[m for m in tr if m['season']=='2324']
    inner_valid=[m for m in tr if m['season']=='2425']
    if not inner_train or not inner_valid:
        raise SystemExit('Missing inner train/validation seasons')
    ic,iv,iy=goal_features(inner_train)
    vc,vv,vy=goal_features(inner_valid)
    inner_enc=OneHotEncoder(handle_unknown='ignore',sparse_output=True)
    inner_enc.fit(ic)
    inner_X=matrix(inner_enc,ic,iv)
    inner_V=matrix(inner_enc,vc,vv)
    candidates=[0.0001,0.001,0.01,0.1]
    validation=[]
    for alpha in candidates:
        candidate=PoissonRegressor(alpha=alpha,max_iter=1000,tol=1e-7).fit(inner_X,iy)
        mu=np.clip(candidate.predict(inner_V),0.05,7.0)
        dev=float(np.mean(mu-vy*np.log(mu)))  # Poisson NLL without constant log(y!)
        validation.append({'alpha':alpha,'validation_poisson_nll_no_constant':dev})
    best=min(validation,key=lambda r:r['validation_poisson_nll_no_constant'])['alpha']
    cat,venue,y=goal_features(tr)
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=True)
    # Column 1 encodes attack, column 2 encodes opponent defense.
    enc.fit(cat)
    X=matrix(enc,cat,venue)
    reg=PoissonRegressor(alpha=best,max_iter=1000,tol=1e-7).fit(X,y)
    tc,tv,_=goal_features(te)
    lambdas=np.clip(reg.predict(matrix(enc,tc,tv)),0.05,7.0).reshape(-1,2)
    predicted=[probs(h,a) for h,a in lambdas]
    p=np.array([x[0] for x in predicted]);labels=np.array([m['result'] for m in te])
    # Recompute original baseline on the identical test rows.
    all_x,all_y=features(games)
    train_idx=[i for i,m in enumerate(games) if m['season'] in TRAIN]
    test_idx=[i for i,m in enumerate(games) if m['season'] in TEST]
    base=LogisticRegression(max_iter=1000).fit(all_x[train_idx],all_y[train_idx])
    bp0=base.predict_proba(all_x[test_idx]);bc=list(base.classes_)
    bp=np.array([[row[bc.index(l)] for l in LABELS] for row in bp0])
    report={'version':'0.3 research prototype','train_matches':len(tr),'test_matches':len(te),
            'poisson':scores(labels,p),'elo_logistic_02':scores(labels,bp),
            'model':'Poisson goals regression; regularization chosen on 2023/24 -> 2024/25 validation; no Dixon-Coles',
            'selected_alpha':best,'alpha_validation':validation,
            'diagnostics':{'predicted_home_wins':int((p.argmax(axis=1)==0).sum()),
                           'predicted_draws':int((p.argmax(axis=1)==1).sum()),
                           'predicted_away_wins':int((p.argmax(axis=1)==2).sum()),
                           'expected_home_goals_std':float(lambdas[:,0].std()),
                           'expected_away_goals_std':float(lambdas[:,1].std())},
            'limitations':['Season holdout 2025/26 is reused from 0.2: not an independent untouched test.',
              'New/promoted teams have unseen categorical coefficients; evaluate them separately.',
              'No injuries, xG or lineup features; no claim of real-time forecasts.',
              'Over/under and BTTS have no independently validated market-odds backtest here.',
              'Odds availability/timing not verified; no profitable-betting claim.',
              'Tuning is exploratory; 2025/26 has been inspected in previous versions and is not untouched.']}
    OUT.mkdir(exist_ok=True)
    (OUT/'metrics_v03.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for i,m in enumerate(te):
        row={'date':str(m['date']),'league':m['league'],'home':m['home'],'away':m['away'],
          'result':m['result'],'home_goals':m['hg'],'away_goals':m['ag'],
          'expected_home_goals':round(float(lambdas[i,0]),5),'expected_away_goals':round(float(lambdas[i,1]),5),
          'p_home':round(float(p[i,0]),6),'p_draw':round(float(p[i,1]),6),'p_away':round(float(p[i,2]),6),
          'p_under_2_5':round(float(predicted[i][1]),6),'p_over_2_5':round(float(1-predicted[i][1]),6),
          'p_btts_yes':round(float(predicted[i][2]),6), 'p_btts_no':round(float(1-predicted[i][2]),6),
          'baseline_p_home':round(float(bp[i,0]),6),'baseline_p_draw':round(float(bp[i,1]),6),
          'baseline_p_away':round(float(bp[i,2]),6)}
        rows.append(row)
    with (OUT/'predictions_v03.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

"""Chronological 1X2 evaluation, historical odds, and fixed-stake paper betting."""
import csv
import datetime as dt
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
TRAIN = {'2324', '2425'}
TEST = {'2526'}
LABELS = ['H', 'D', 'A']
ODDS_SETS = [('B365H','B365D','B365A'), ('AvgH','AvgD','AvgA'), ('MaxH','MaxD','MaxA')]

def parse_date(s):
    for fmt in ('%d/%m/%Y','%d/%m/%y','%Y-%m-%d'):
        try: return dt.datetime.strptime(s.strip(),fmt).date()
        except ValueError: pass
    return None

def valid_odds(row):
    for cols in ODDS_SETS:
        try:
            values=[float(row[k]) for k in cols]
            if all(np.isfinite(v) and v > 1.0 for v in values):
                return dict(zip(LABELS,values)), '/'.join(cols)
        except (KeyError,ValueError,TypeError): pass
    return None, None

def load():
    games=[]
    for f in sorted((ROOT/'data').glob('*.csv')):
        try: season,league=f.stem.split('_',1)
        except ValueError: continue
        if season not in TRAIN|TEST: continue
        with f.open(encoding='utf-8-sig',errors='replace',newline='') as handle:
            for row in csv.DictReader(handle):
                date=parse_date(row.get('Date',''))
                if not date or row.get('FTR') not in LABELS or not row.get('HomeTeam') or not row.get('AwayTeam'): continue
                try: hg,ag=int(row['FTHG']),int(row['FTAG'])
                except (ValueError,TypeError): continue
                odds,source=valid_odds(row)
                games.append(dict(date=date,season=season,league=league,home=row['HomeTeam'].strip(),away=row['AwayTeam'].strip(),result=row['FTR'],hg=hg,ag=ag,odds=odds,odds_source=source))
    return sorted(games,key=lambda x:(x['date'],x['league'],x['home']))

def features(games):
    elo=defaultdict(lambda:1500.0); form=defaultdict(list)
    X=[]; Y=[]
    for m in games:
        h=(m['league'],m['home']);a=(m['league'],m['away'])
        eh,ea=elo[h],elo[a];fh,fa=form[h][-5:],form[a][-5:]
        avg=lambda x,j:sum(v[j] for v in x)/len(x) if x else 1.3
        X.append([eh-ea,avg(fh,0)-avg(fa,0),avg(fh,1)-avg(fa,1),len(fh),len(fa)])
        Y.append(m['result'])
        expected=1/(1+10**((ea-eh-65)/400))
        actual={'H':1,'D':0.5,'A':0}[m['result']]
        delta=22*(actual-expected)
        elo[h]+=delta;elo[a]-=delta
        form[h].append((m['hg'],m['ag']));form[a].append((m['ag'],m['hg']))
    return np.asarray(X),np.asarray(Y)

def scores(y,p):
    oh=np.array([[float(v==lab) for lab in LABELS] for v in y])
    # Explicit H,D,A ordering avoids sklearn.log_loss alphabetical-label mismatch.
    return dict(accuracy=float(accuracy_score(y,np.asarray(LABELS)[p.argmax(1)])),
                multiclass_brier=float(np.mean(np.sum((p-oh)**2,axis=1))),
                log_loss=float(-np.mean(np.sum(oh*np.log(np.clip(p,1e-15,1)),axis=1))))

def main():
    games=load()
    X,Y=features(games)
    tr=[i for i,m in enumerate(games) if m['season'] in TRAIN]
    te=[i for i,m in enumerate(games) if m['season'] in TEST]
    if len(tr)<200 or len(te)<100: raise SystemExit('Not enough matches')
    model=LogisticRegression(max_iter=1000).fit(X[tr],Y[tr])
    raw=model.predict_proba(X[te]); classes=list(model.classes_)
    p=np.asarray([[row[classes.index(lab)] for lab in LABELS] for row in raw])
    metrics={'model':'Elo + form + logistic regression (v0.2)', 'train_matches':len(tr),'test_matches':len(te),**scores(Y[te],p)}
    # Predeclared illustrative EV thresholds, not optimized on test set.
    thresholds=[0.0,0.03,0.05,0.10]
    bets={str(t):[] for t in thresholds}
    market=[]; records=[]
    for j,idx in enumerate(te):
        m=games[idx]; odds=m['odds']; pp=dict(zip(LABELS,p[j]))
        rec=dict(date=str(m['date']),league=m['league'],home=m['home'],away=m['away'],result=m['result'],p_home=pp['H'],p_draw=pp['D'],p_away=pp['A'],odds_source=m['odds_source'] or '')
        for lab in LABELS: rec['odds_'+lab.lower()]=odds[lab] if odds else ''
        if odds:
            implied=np.array([1/odds[lab] for lab in LABELS]); market.append((m['result'],implied/implied.sum()))
            options=[(pp[lab]*odds[lab]-1,lab) for lab in LABELS]
            ev,choice=max(options)
            rec.update(best_ev=ev,best_bet=choice,best_bet_odds=odds[choice])
            for t in thresholds:
                if ev>t:
                    profit=(odds[choice]-1) if m['result']==choice else -1.0
                    bets[str(t)].append((profit,choice==m['result']))
        else: rec.update(best_ev='',best_bet='',best_bet_odds='')
        records.append(rec)
    metrics['market_comparison']={'matches_with_odds':len(market),'coverage':len(market)/len(te)}
    if market:
        my=np.array([v[0] for v in market]); mp=np.array([v[1] for v in market])
        metrics['market_comparison'].update(scores(my,mp))
    metrics['paper_betting']={}
    for t,items in bets.items():
        profits=np.array([v[0] for v in items]); n=len(items)
        curve=np.r_[0,profits.cumsum()] if n else np.array([0.])
        metrics['paper_betting'][t]={'bets':n,'win_rate':float(np.mean([v[1] for v in items])) if n else None,
          'net_units':float(profits.sum()),'roi':float(profits.mean()) if n else None,
          'max_drawdown_units':float(np.max(np.maximum.accumulate(curve)-curve))}
    metrics['warnings']=['Historical odds columns may be closing odds, not guaranteed available at prediction timestamp.',
      'Odds are for evaluation only, never training features.',
      'Positive backtest ROI is not proof of future profitability.',
      'Thresholds are illustrative and must not be selected based on this test set.',
      'Market baseline uses normalized implied probabilities; margin removed.']
    out=ROOT/'output';out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'historical_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps(metrics,indent=2,ensure_ascii=False))

if __name__=='__main__': main()

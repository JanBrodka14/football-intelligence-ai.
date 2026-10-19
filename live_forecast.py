"""Pre-match daily forecasts from public league CSVs. No bookmaker edge claims."""
import csv
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from backtest import date_of, fit_goals, goal_means, outcome_probs, elo_features, LABELS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
LEDGER = ROOT / 'live_predictions.csv'
OUT = ROOT / 'output'
FIELDS = ['forecast_id','created_utc','match_date','league','home','away','p_home','p_draw','p_away','p_over_2_5','p_under_2_5','p_btts_yes','p_btts_no','elo_p_home','elo_p_draw','elo_p_away','selection','confidence','result','home_goals','away_goals','settled_utc']


def load_all(asof):
    past, future = [], []
    for file in sorted(DATA.glob('*.csv')):
        try:
            season, league = file.stem.split('_',1)
        except ValueError:
            continue
        if season not in ('2324','2425','2526','2627'):
            continue
        with file.open(encoding='utf-8-sig',errors='replace',newline='') as stream:
            for r in csv.DictReader(stream):
                day = date_of(r.get('Date',''))
                h,a = (r.get('HomeTeam') or '').strip(),(r.get('AwayTeam') or '').strip()
                if not day or not h or not a:
                    continue
                m = dict(date=day,season=season,league=league,home=h,away=a)
                try:
                    hg,ag = int(r['FTHG']),int(r['FTAG'])
                    valid = r.get('FTR') in LABELS and {'H':hg>ag,'D':hg==ag,'A':hg<ag}[r['FTR']]
                except (ValueError,TypeError,KeyError):
                    valid = False
                if valid:
                    m.update(hg=hg,ag=ag,result=r['FTR'])
                    if day < asof:
                        past.append(m)
                    else:
                        # Results on/after asof are NEVER training data, but can settle older forecasts.
                        future.append(m)
                elif day >= asof:
                    future.append(m)
    key=lambda m:(m['date'],m['league'],m['home'],m['away'])
    return sorted(past,key=key), sorted(future,key=key)


def read_ledger():
    if not LEDGER.exists():
        return []
    with LEDGER.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def save_ledger(rows):
    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    temp = LEDGER.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');w.writeheader();w.writerows(rows)
    temp.replace(LEDGER)


def match_id(m):
    return '|'.join((m['date'].isoformat(),m['league'],m['home'],m['away']))


def settle(rows, all_results, now):
    lookup={match_id(m):m for m in all_results if 'result' in m}
    count=0
    for row in rows:
        if row.get('result'):
            continue
        m=lookup.get(row['forecast_id'])
        if m and m['date'] < now:
            row.update(result=m['result'],home_goals=str(m['hg']),away_goals=str(m['ag']),settled_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            count+=1
    return count


def future_elo_features(past, fixtures):
    # Compute Elo/form state exclusively from completed games before forecast date.
    elo=defaultdict(lambda:1500.)
    form=defaultdict(list)
    for m in past:
        h,a=(m['league'],m['home']),(m['league'],m['away'])
        eh,ea=elo[h],elo[a]
        expected=1/(1+10**((ea-eh-65)/400))
        change=22*({'H':1.,'D':.5,'A':0.}[m['result']]-expected)
        elo[h]+=change;elo[a]-=change
        form[h].append((m['hg'],m['ag']))
        form[a].append((m['ag'],m['hg']))
    x=[]
    for m in fixtures:
        h,a=(m['league'],m['home']),(m['league'],m['away'])
        fh,fa=form[h][-5:],form[a][-5:]
        avg=lambda v,j:sum(z[j] for z in v)/len(v) if v else 1.3
        x.append([elo[h]-elo[a],avg(fh,0)-avg(fa,0),avg(fh,1)-avg(fa,1),len(fh),len(fa)])
    return np.array(x)


def generate(asof, horizon=3):
    past,future=load_all(asof)
    rows=read_ledger()
    settled=settle(rows,past,asof)
    existing={r['forecast_id'] for r in rows}
    fixtures=[m for m in future if 'result' not in m and asof<m['date']<=asof+dt.timedelta(days=horizon) and match_id(m) not in existing]
    # Keep a single fixture per match, even if the source repeats it.
    fixtures=list({match_id(m):m for m in fixtures}.values())
    fixtures.sort(key=lambda m:(m['date'],m['league'],m['home']))
    if len(past)<500:
        raise ValueError(f'Insufficient past data: {len(past)} matches')
    if fixtures:
        # Fixed, previously validated hyperparameters; no tuning on future outcomes.
        mu=goal_means(fit_goals(past,alpha=.001),fixtures)
        xpast=elo_features(past)
        model=LogisticRegression(max_iter=1500).fit(xpast,[m['result'] for m in past])
        ep=model.predict_proba(future_elo_features(past,fixtures))
        ep=ep[:,[list(model.classes_).index(c) for c in LABELS]]
        for i,m in enumerate(fixtures):
            p,under,btts=outcome_probs(*mu[i],rho=-.1)
            choice=int(np.argmax(p))
            rows.append(dict(forecast_id=match_id(m),created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                match_date=m['date'].isoformat(),league=m['league'],home=m['home'],away=m['away'],
                p_home=f'{p[0]:.6f}',p_draw=f'{p[1]:.6f}',p_away=f'{p[2]:.6f}',
                p_over_2_5=f'{1-under:.6f}',p_under_2_5=f'{under:.6f}',
                p_btts_yes=f'{btts:.6f}',p_btts_no=f'{1-btts:.6f}',
                elo_p_home=f'{ep[i,0]:.6f}',elo_p_draw=f'{ep[i,1]:.6f}',elo_p_away=f'{ep[i,2]:.6f}',
                selection=LABELS[choice],confidence=f'{p[choice]:.6f}',result='',home_goals='',away_goals='',settled_utc=''))
    save_ledger(rows)
    today=[r for r in rows if asof.isoformat()<r['match_date']<=(asof+dt.timedelta(days=horizon)).isoformat()]
    top=sorted(today,key=lambda r:float(r['confidence']),reverse=True)[:5]
    settled_rows=[r for r in rows if r.get('result')]
    acc=sum(r['selection']==r['result'] for r in settled_rows)/len(settled_rows) if settled_rows else None
    report={'date':asof.isoformat(),'training_matches':len(past),'new_forecasts':len(fixtures),'settled_today':settled,
            'all_forecasts':len(rows),'settled_forecasts':len(settled_rows),'settled_accuracy':acc,
            'top_5_upcoming':top,'warning':'Ranked by model confidence, NOT validated value bets or guaranteed winners.',
            'source_limitations':'Fixtures are only as complete and timely as football-data.co.uk CSVs. No guaranteed kickoff time.'}
    OUT.mkdir(exist_ok=True)
    (OUT/'daily_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[f'# Football Intelligence AI — {asof.isoformat()}',f'New forecasts: {len(fixtures)} | Settled: {len(settled_rows)} | Accuracy: {acc:.1%}' if acc is not None else f'New forecasts: {len(fixtures)} | No settled predictions yet', '', '## Top 5 upcoming (confidence, not betting value)']
    for r in top:
        lines.append(f"- {r['league']} {r['home']} – {r['away']}: {r['selection']} ({float(r['confidence']):.1%}); 1/X/2 {float(r['p_home']):.1%}/{float(r['p_draw']):.1%}/{float(r['p_away']):.1%}")
    if not top:lines.append('- No upcoming fixtures available in downloaded CSVs.')
    lines.append('\nModel is experimental; no demonstrated bookmaker advantage.')
    (OUT/'daily_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    asof=dt.date.fromisoformat(os.environ.get('FORECAST_DATE',dt.datetime.now(dt.timezone.utc).date().isoformat()))
    print(json.dumps(generate(asof),ensure_ascii=False,indent=2))

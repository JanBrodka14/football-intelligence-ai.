"""Conservative coupon screening against timestamped decimal odds; no invented prices."""
import csv
import datetime as dt
import json
import math
from pathlib import Path

MARKETS = {'H':'p_home','D':'p_draw','A':'p_away',
           'OVER_2_5':'p_over_2_5','UNDER_2_5':'p_under_2_5',
           'BTTS_YES':'p_btts_yes','BTTS_NO':'p_btts_no'}
ODDS_FIELDS = ['forecast_id','market','decimal_odds','captured_utc','bookmaker']

def read_odds(path):
    path=Path(path)
    if not path.exists():
        return [], f'No odds file: {path.name}. No value bets can be verified.'
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not set(ODDS_FIELDS).issubset(reader.fieldnames or []):
            return [], 'Odds CSV missing required columns.'
        return list(reader), None

def select(ledger, odds, now=None, min_odds=1.50, max_odds=1.75, min_edge=0.03, max_legs=5):
    """Screen unstarted matches only; one leg per match, 4-5 legs only when qualified.

    No kickoff timestamps in the public fixture feed: restrict to future calendar dates.
    Odds must have an explicit timestamp within 24h of evaluation.
    """
    now=now or dt.datetime.now(dt.timezone.utc)
    forecasts={r['forecast_id']:r for r in ledger if not r.get('result') and r.get('match_date','')>now.date().isoformat()}
    best={}
    rejected={'invalid':0,'stale':0,'no_forecast':0,'outside_range':0,'insufficient_edge':0}
    for quote in odds:
        try:
            fid=quote['forecast_id']; market=quote['market'].strip().upper()
            if market not in MARKETS or fid not in forecasts:
                rejected['no_forecast']+=1;continue
            price=float(quote['decimal_odds'])
            if not math.isfinite(price) or price<=1:
                raise ValueError('invalid odds')
            stamp=dt.datetime.fromisoformat(quote['captured_utc'].strip().replace('Z','+00:00'))
            if stamp.tzinfo is None:raise ValueError('missing timezone')
            age=(now-stamp.astimezone(dt.timezone.utc)).total_seconds()
            if age<0 or age>86400:
                rejected['stale']+=1;continue
            if not min_odds<=price<=max_odds:
                rejected['outside_range']+=1;continue
            row=forecasts[fid]; p=float(row[MARKETS[market]])
            if not 0<=p<=1 or not math.isfinite(p):raise ValueError('invalid probability')
            edge=p-1/price
            if edge<min_edge:
                rejected['insufficient_edge']+=1;continue
            candidate={'forecast_id':fid,'match_date':row['match_date'],'league':row['league'],
                       'home':row['home'],'away':row['away'],'market':market,'odds':price,
                       'model_probability':round(p,6),'break_even_probability':round(1/price,6),
                       'estimated_edge_pp':round(100*edge,2),'estimated_ev_per_unit':round(p*price-1,5),
                       'captured_utc':quote['captured_utc'],'bookmaker':quote['bookmaker']}
            # Select one best opportunity per match, prioritizing expected value.
            if fid not in best or candidate['estimated_ev_per_unit']>best[fid]['estimated_ev_per_unit']:
                best[fid]=candidate
        except (KeyError,TypeError,ValueError,OverflowError):
            rejected['invalid']+=1
    candidates=sorted(best.values(),key=lambda x:(-x['estimated_ev_per_unit'],-x['estimated_edge_pp']))
    legs=candidates[:max_legs]
    # No independence assertion; multiplication is only an illustrative product of model probabilities.
    coupon={'status':'CANDIDATES_ONLY' if len(legs)>=4 else 'INSUFFICIENT_QUALIFIED_LEGS',
            'legs':legs,'number_of_legs':len(legs),
            'combined_decimal_odds':round(math.prod(x['odds'] for x in legs),3) if legs else None,
            'warning':'Model edge is unverified against the market; not proof of positive EV. Match correlations and uncertainty are not modeled.',
            'rejected':rejected,'total_qualified_matches':len(candidates)}
    return coupon

def write_report(ledger_path='live_predictions.csv',odds_path='odds_snapshot.csv',output_dir='output',now=None):
    now=now or dt.datetime.now(dt.timezone.utc)
    ledger_path=Path(ledger_path)
    if ledger_path.exists():
        with ledger_path.open(encoding='utf-8',newline='') as f:ledger=list(csv.DictReader(f))
    else:ledger=[]
    odds,error=read_odds(odds_path)
    result=select(ledger,odds,now=now)
    result['odds_status']=error or f'Loaded {len(odds)} timestamped quotes; no live feed is connected.'
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'coupon_candidates.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Football Intelligence AI — AKO candidate screening','',result['odds_status'],'',
           f"Status: {result['status']}; qualifying matches: {result['total_qualified_matches']}",'']
    for x in result['legs']:
        lines.append(f"- {x['home']} – {x['away']}: {x['market']} @ {x['odds']:.2f}; model {x['model_probability']:.1%}; estimated edge {x['estimated_edge_pp']:+.2f} pp")
    if not result['legs']:lines.append('- No qualified selections. Do not invent odds or fill a coupon by force.')
    lines.extend(['',result['warning'],'','Odds must be captured within the last 24h and the match must be on a future calendar date.'])
    (output_dir/'coupon_candidates.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    print(json.dumps(write_report(),ensure_ascii=False,indent=2))

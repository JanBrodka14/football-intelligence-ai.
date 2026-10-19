"""API-FOOTBALL v3 ingestion. Standard library only; never prints API key."""
import csv
import datetime as dt
import json
import os
import re
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEAGUES = {'E0':39,'SP1':140,'I1':135,'D1':78,'F1':61,'N1':88,'P1':94,'B1':144,'SC0':179}
# Exact normalized aliases only; ambiguous teams are skipped, never guessed.
ALIASES = {'manchester united':'man united','manchester city':'man city','tottenham hotspur':'tottenham',
 'newcastle united':'newcastle','nottingham forest':'nottm forest','wolverhampton wanderers':'wolves',
 'brighton and hove albion':'brighton','west ham united':'west ham','leicester city':'leicester',
 'leeds united':'leeds','ipswich town':'ipswich','sheffield united':'sheffield united',
 'internazionale':'inter','inter milan':'inter','ac milan':'milan','as roma':'roma',
 'napoli':'napoli','juventus':'juventus','bayern munich':'bayern munich',
 'bayern munchen':'bayern munich','borussia dortmund':'dortmund',
 'bayer leverkusen':'leverkusen','rb leipzig':'rb leipzig',
 'paris saint germain':'paris sg','psg':'paris sg',
 'olympique marseille':'marseille','olympique lyonnais':'lyon',
 'atletico madrid':'ath madrid','athletic club':'ath bilbao',
 'real sociedad':'sociedad','real betis':'betis','sporting cp':'sporting',
 'sporting lisbon':'sporting','fc porto':'porto','sl benfica':'benfica',
 'ajax amsterdam':'ajax','psv eindhoven':'psv','feyenoord rotterdam':'feyenoord',
 'club brugge kv':'club brugge','rsc anderlecht':'anderlecht',
 'celtic glasgow':'celtic','rangers glasgow':'rangers'}

def norm(name):
    name=unicodedata.normalize('NFKD',name).encode('ascii','ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+',' ',name).strip()

def api(path, params):
    key=os.environ.get('API_FOOTBALL_KEY','')
    if not key: raise RuntimeError('Missing API_FOOTBALL_KEY secret')
    url='https://v3.football.api-sports.io/'+path+'?'+urllib.parse.urlencode(params)
    req=urllib.request.Request(url,headers={'x-apisports-key':key,'User-Agent':'FootballIntelligence/0.6'})
    with urllib.request.urlopen(req,timeout=25) as response:
        data=json.load(response)
    if data.get('errors'): raise RuntimeError(f'API {path}: {data["errors"]}')
    return data

def historical_names(code):
    names={}
    for f in (ROOT/'data').glob('*_'+code+'.csv'):
        with f.open(encoding='utf-8-sig',errors='replace',newline='') as stream:
            for row in csv.DictReader(stream):
                for col in ('HomeTeam','AwayTeam'):
                    name=(row.get(col) or '').strip()
                    if name: names[norm(name)]=name
    return names

def match_team(api_name, names):
    key=norm(api_name)
    key=ALIASES.get(key,key)
    return names.get(key)

def ingest_fixtures(days=3):
    today=dt.datetime.now(dt.timezone.utc).date()
    season=today.year if today.month>=7 else today.year-1
    mapping={}; stats={}; skipped=[]
    for code,league_id in LEAGUES.items():
        names=historical_names(code)
        data=api('fixtures',{'league':league_id,'season':season,'from':(today+dt.timedelta(days=1)).isoformat(),
                             'to':(today+dt.timedelta(days=days)).isoformat()})
        additions=[]
        for item in data.get('response',[]):
            status=item['fixture']['status']['short']
            if status!='NS':continue
            home=match_team(item['teams']['home']['name'],names)
            away=match_team(item['teams']['away']['name'],names)
            if not home or not away or home==away:
                skipped.append({'league':code,'home':item['teams']['home']['name'],'away':item['teams']['away']['name']})
                continue
            day=item['fixture']['date'][:10]
            fid='|'.join((day,code,home,away))
            mapping[fid]=item['fixture']['id']
            additions.append({'Date':dt.date.fromisoformat(day).strftime('%d/%m/%Y'),'HomeTeam':home,'AwayTeam':away,'FTHG':'','FTAG':'','FTR':''})
        target=ROOT/'data'/f'{str(season)[-2:]}{str(season+1)[-2:]}_{code}.csv'
        # Keep existing historical rows, add only previously unseen fixtures.
        if target.exists():
            with target.open(encoding='utf-8-sig',errors='replace',newline='') as f:
                reader=csv.DictReader(f);fields=reader.fieldnames or [];rows=list(reader)
        else: fields=['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR'];rows=[]
        existing={(r.get('Date'),r.get('HomeTeam'),r.get('AwayTeam')) for r in rows}
        count=0
        for row in additions:
            key=(row['Date'],row['HomeTeam'],row['AwayTeam'])
            if key not in existing:
                rows.append(row);existing.add(key);count+=1
        if count:
            with target.open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        stats[code]={'api_fixtures':data.get('results',0),'matched':len(additions),'appended':count}
    (ROOT/'api_fixture_map.json').write_text(json.dumps(mapping,indent=2))
    (ROOT/'output').mkdir(exist_ok=True)
    (ROOT/'output'/'api_ingestion.json').write_text(json.dumps({'leagues':stats,'unmatched':skipped,'fixture_ids':len(mapping)},indent=2))
    print('Fixture mapping:',len(mapping),'unmatched:',len(skipped))

def classify(bet,value):
    b=norm(bet);v=norm(value)
    if b in ('match winner','1x2','fulltime result'):
        return {'home':'H','draw':'D','away':'A','1':'H','x':'D','2':'A'}.get(v)
    if b in ('goals over under','over under 2 5','goals over under 2 5'):
        return {'over 2 5':'OVER_2_5','under 2 5':'UNDER_2_5'}.get(v)
    if b in ('both teams score','both teams to score','btts'):
        return {'yes':'BTTS_YES','no':'BTTS_NO'}.get(v)
    return None

def ingest_odds(max_calls=35):
    mapping=json.loads((ROOT/'api_fixture_map.json').read_text())
    # Fetch only fixtures forecasted by our model, with a strict quota budget.
    ledger=ROOT/'live_predictions.csv'
    if not ledger.exists(): raise RuntimeError('Missing forecast ledger')
    with ledger.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    now=dt.datetime.now(dt.timezone.utc)
    eligible=[r['forecast_id'] for r in rows if r.get('forecast_id') in mapping and r.get('match_date','')>now.date().isoformat()]
    quotes=[];checked=0
    for fid in eligible[:max_calls]:
        data=api('odds',{'fixture':mapping[fid]});checked+=1
        for fixture in data.get('response',[]):
            for bookmaker in fixture.get('bookmakers',[]):
                for bet in bookmaker.get('bets',[]):
                    for value in bet.get('values',[]):
                        market=classify(bet.get('name',''),value.get('value',''))
                        if not market:continue
                        try: price=float(value['odd'])
                        except (ValueError,TypeError,KeyError):continue
                        if not 1.5<=price<=1.75:continue
                        stamp=fixture.get('update') or bookmaker.get('update')
                        # Never invent timestamps: API response without update is unusable.
                        if not stamp:continue
                        quotes.append({'forecast_id':fid,'market':market,'decimal_odds':price,
                                       'captured_utc':stamp,'bookmaker':bookmaker.get('name','Unknown')})
    with (ROOT/'odds_snapshot.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['forecast_id','market','decimal_odds','captured_utc','bookmaker'])
        writer.writeheader();writer.writerows(quotes)
    (ROOT/'output'/'api_odds.json').write_text(json.dumps({'eligible':len(eligible),'queried':checked,'quotes':len(quotes),'quota_cap':max_calls},indent=2))
    print('Odds API calls:',checked,'usable quotes:',len(quotes))

if __name__=='__main__':
    import sys
    if len(sys.argv)!=2 or sys.argv[1] not in ('fixtures','odds'):
        raise SystemExit('Usage: python api_football.py fixtures|odds')
    if sys.argv[1]=='fixtures':ingest_fixtures()
    else:ingest_odds(int(os.environ.get('MAX_ODDS_CALLS','35')))

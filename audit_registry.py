"""Validate manually curated forecast registry; never invent or scrape missing outcomes."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def audit(path=ROOT/'forecast_registry.csv'):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate forecast IDs')
    settled=[]; pending=[]; excluded=[]
    for row in rows:
        p=row['prawdopodobienstwo'].strip()
        actual=row['trafiony'].strip()
        classification=row['klasyfikacja'].lower()
        if 'selekcja raportu' not in classification:
            excluded.append(row['id']);continue
        if not p:
            excluded.append(row['id']);continue
        prob=float(p)/100
        if not 0<=prob<=1:
            raise ValueError('Invalid probability '+row['id'])
        if actual not in ('0','0.0','1','1.0'):
            pending.append(row['id']);continue
        outcome=float(actual)
        settled.append(dict(id=row['id'],probability=prob,hit=outcome,brier=(prob-outcome)**2))
    return dict(total=len(rows),settled_selected=len(settled),pending_selected=len(pending),
                excluded=len(excluded),selected_accuracy=sum(v['hit'] for v in settled)/len(settled) if settled else None,
                selected_binary_brier=sum(v['brier'] for v in settled)/len(settled) if settled else None,
                settled_ids=[v['id'] for v in settled],pending_ids=pending,excluded_ids=excluded,
                warning='Only approved selections count. External sources are not proprietary model performance. Missing outcomes require verification and manual update.')

if __name__=='__main__':
    report=audit()
    out=ROOT/'output';out.mkdir(exist_ok=True)
    (out/'registry_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

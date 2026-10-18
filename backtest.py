"""Chronological out-of-sample 1X2 backtest with pre-match Elo features.

Train seasons: 2023/24 and 2024/25; test: 2025/26.
2026/27 is never used to fit or score the historical test.
"""
import csv
import datetime as dt
import json
import pathlib
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'output'
TRAIN = {'2324', '2425'}
TEST = {'2526'}
LABELS = ['H', 'D', 'A']


def date_parse(s):
    for fmt in ('%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d'):
        try:
            return dt.datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    return None


def matches():
    result = []
    for file in sorted(DATA.glob('*.csv')):
        season, league = file.stem.split('_', 1)
        if season not in TRAIN | TEST:
            continue
        with file.open(encoding='utf-8-sig', errors='replace', newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get('FTR') not in LABELS:
                    continue
                date = date_parse(row.get('Date', ''))
                if not date or not row.get('HomeTeam') or not row.get('AwayTeam'):
                    continue
                try:
                    hg, ag = int(row['FTHG']), int(row['FTAG'])
                except (TypeError, ValueError):
                    continue
                result.append({'season': season, 'league': league, 'date': date,
                               'home': row['HomeTeam'].strip(), 'away': row['AwayTeam'].strip(),
                               'hg': hg, 'ag': ag, 'result': row['FTR']})
    return sorted(result, key=lambda m: (m['date'], m['league'], m['home']))


def build_features(games):
    elo = defaultdict(lambda: 1500.0)
    form = defaultdict(list)
    x, y, meta = [], [], []
    # Pre-match features are constructed BEFORE updating ratings or form.
    for m in games:
        h, a = (m['league'], m['home']), (m['league'], m['away'])
        eh, ea = elo[h], elo[a]
        fh, fa = form[h][-5:], form[a][-5:]
        avg = lambda v, index: sum(t[index] for t in v) / len(v) if v else 1.3
        features = [eh - ea, avg(fh, 0) - avg(fa, 0),
                    avg(fh, 1) - avg(fa, 1), len(fh), len(fa)]
        x.append(features)
        y.append(m['result'])
        meta.append(m)
        expected = 1 / (1 + 10 ** ((ea - (eh + 65)) / 400))
        actual = 1 if m['result'] == 'H' else (0.5 if m['result'] == 'D' else 0)
        delta = 22 * (actual - expected)
        elo[h] += delta
        elo[a] -= delta
        form[h].append((m['hg'], m['ag']))
        form[a].append((m['ag'], m['hg']))
    return np.asarray(x), np.asarray(y), meta


def main():
    games = matches()
    if not games:
        raise SystemExit('No CSV data. Run python download_data.py first.')
    x, y, meta = build_features(games)
    train_idx = [i for i, m in enumerate(meta) if m['season'] in TRAIN]
    test_idx = [i for i, m in enumerate(meta) if m['season'] in TEST]
    if len(train_idx) < 200 or len(test_idx) < 100:
        raise SystemExit(f'Insufficient data: train={len(train_idx)}, test={len(test_idx)}')
    model = LogisticRegression(max_iter=1000)
    model.fit(x[train_idx], y[train_idx])
    probs = model.predict_proba(x[test_idx])
    classes = list(model.classes_)
    p = np.asarray([[row[classes.index(label)] for label in LABELS] for row in probs])
    actual = y[test_idx]
    onehot = np.asarray([[float(z == label) for label in LABELS] for z in actual])
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    metrics = {'model': 'Elo + last-five goal form + multinomial logistic regression',
               'train_seasons': sorted(TRAIN), 'test_seasons': sorted(TEST),
               'train_matches': len(train_idx), 'test_matches': len(test_idx),
               'accuracy': float(accuracy_score(actual, np.asarray(LABELS)[p.argmax(axis=1)])),
               'multiclass_brier': brier,
               'log_loss': float(log_loss(actual, p, labels=LABELS)),
               'note': 'Prototype baseline, not yet calibrated; no xG or bookmaker odds.'}
    OUT.mkdir(exist_ok=True)
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    with (OUT / 'historical_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'league', 'home', 'away', 'result', 'p_home', 'p_draw', 'p_away'])
        for j, idx in enumerate(test_idx):
            m = meta[idx]
            writer.writerow([m['date'], m['league'], m['home'], m['away'], m['result'],
                             *[f'{v:.6f}' for v in p[j]]])
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print('Saved: output/metrics.json and output/historical_predictions.csv')


if __name__ == '__main__':
    main()

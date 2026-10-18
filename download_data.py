"""Download historical match results from football-data.co.uk (public CSVs)."""
import csv
import io
import pathlib
import urllib.request

BASE = 'https://www.football-data.co.uk/mmz4281'
LEAGUES = {'E0': 'Premier League', 'SP1': 'La Liga', 'I1': 'Serie A',
           'D1': 'Bundesliga', 'F1': 'Ligue 1', 'N1': 'Eredivisie',
           'P1': 'Primeira Liga', 'B1': 'Belgian Pro League', 'SC0': 'Scottish Premiership'}
SEASONS = ('2324', '2425', '2526', '2627')
DEST = pathlib.Path(__file__).resolve().parent / 'data'


def main():
    DEST.mkdir(exist_ok=True)
    good, failed = 0, []
    for season in SEASONS:
        for code, name in LEAGUES.items():
            url = f'{BASE}/{season}/{code}.csv'
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'FootballIntelligenceResearch/0.1'})
                with urllib.request.urlopen(req, timeout=20) as response:
                    raw = response.read()
                decoded = raw.decode('utf-8-sig', errors='replace')
                rows = list(csv.DictReader(io.StringIO(decoded)))
                if not rows or 'HomeTeam' not in rows[0] or 'FTR' not in rows[0]:
                    raise ValueError('Missing expected columns')
                (DEST / f'{season}_{code}.csv').write_bytes(raw)
                good += 1
                print(f'OK {season} {code}: {len(rows)} rows')
            except Exception as exc:
                failed.append((season, code, str(exc)))
                print(f'SKIP {season} {code}: {exc}')
    print(f'Files downloaded: {good}; skipped: {len(failed)}')
    if good == 0:
        raise SystemExit('No usable data downloaded')


if __name__ == '__main__':
    main()

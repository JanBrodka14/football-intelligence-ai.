"""Football Intelligence AI 0.4: research-only historical evaluation."""
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
LABELS = ("H", "D", "A")
TRAIN_SEASONS = ("2324", "2425")
TEST_SEASON = "2526"

ODDS_SETS = (
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
    ("MaxH", "MaxD", "MaxA"),
)


def date_of(s):
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s.strip(), fmt).date()
        except (ValueError, AttributeError):
            pass
    return None


def load(root=ROOT):
    matches = []

    for f in sorted((root / "data").glob("*.csv")):
        try:
            season, league = f.stem.split("_", 1)
        except ValueError:
            continue

        if season not in set(TRAIN_SEASONS) | {TEST_SEASON}:
            continue

        with f.open(
            encoding="utf-8-sig",
            errors="replace",
            newline=""
        ) as handle:

            for row in csv.DictReader(handle):
                date = date_of(row.get("Date", ""))

                if (
                    date is None
                    or row.get("FTR") not in LABELS
                    or not row.get("HomeTeam")
                    or not row.get("AwayTeam")
                ):
                    continue

                try:
                    hg = int(row["FTHG"])
                    ag = int(row["FTAG"])
                except (TypeError, ValueError):
                    continue

                if (
                    hg < 0
                    or ag < 0
                    or (hg > ag and row["FTR"] != "H")
                    or (hg == ag and row["FTR"] != "D")
                    or (hg < ag and row["FTR"] != "A")
                ):
                    continue

                odds, source = None, None

                for columns in ODDS_SETS:
                    try:
                        vals = [float(row[c]) for c in columns]

                        if all(
                            np.isfinite(v) and v > 1
                            for v in vals
                        ):
                            odds = vals
                            source = "/".join(columns)
                            break

                    except (KeyError, ValueError, TypeError):
                        pass

                matches.append(
                    dict(
                        date=date,
                        season=season,
                        league=league,
                        home=row["HomeTeam"].strip(),
                        away=row["AwayTeam"].strip(),
                        hg=hg,
                        ag=ag,
                        result=row["FTR"],
                        odds=odds,
                        odds_source=source,
                    )
                )

    return sorted(
        matches,
        key=lambda m: (
            m["date"],
            m["league"],
            m["home"],
            m["away"],
        ),
    )


def elo_features(games):
    """Calculate features without same-day result leakage."""

    elo = defaultdict(lambda: 1500.0)
    form = defaultdict(list)
    x = np.zeros((len(games), 5))

    cursor = 0

    while cursor < len(games):
        end = cursor

        while (
            end < len(games)
            and games[end]["date"] == games[cursor]["date"]
        ):
            end += 1

        for i in range(cursor, end):
            m = games[i]

            h = (m["league"], m["home"])
            a = (m["league"], m["away"])

            fh = form[h][-5:]
            fa = form[a][-5:]

            def avg(v, j):
                return (
                    sum(z[j] for z in v) / len(v)
                    if v else 1.3
                )

            x[i] = [
                elo[h] - elo[a],
                avg(fh, 0) - avg(fa, 0),
                avg(fh, 1) - avg(fa, 1),
                len(fh),
                len(fa),
            ]

        for i in range(cursor, end):
            m = games[i]

            h = (m["league"], m["home"])
            a = (m["league"], m["away"])

            eh, ea = elo[h], elo[a]

            expected = 1 / (
                1 + 10 ** ((ea - eh - 65) / 400)
            )

            actual = {
                "H": 1.0,
                "D": 0.5,
                "A": 0.0,
            }[m["result"]]

            change = 22 * (actual - expected)

            elo[h] += change
            elo[a] -= change

            form[h].append((m["hg"], m["ag"]))
            form[a].append((m["ag"], m["hg"]))

        cursor = end

    return x


def goal_rows(matches):
    cats, venues, targets = [], [], []

    for m in matches:
        home = m["league"] + "|" + m["home"]
        away = m["league"] + "|" + m["away"]

        cats.extend(
            (
                (home, away),
                (away, home),
            )
        )

        venues.extend((1, 0))
        targets.extend((m["hg"], m["ag"]))

    return (
        cats,
        np.array(venues),
        np.array(targets),
    )


def goal_matrix(enc, cats, venue):
    from scipy.sparse import hstack, csr_matrix

    return hstack(
        [
            enc.transform(cats),
            csr_matrix(venue.reshape(-1, 1)),
        ],
        format="csr",
    )


def fit_goals(matches, alpha):
    cats, venue, y = goal_rows(matches)

    enc = OneHotEncoder(
        handle_unknown="ignore"
    )

    enc.fit(cats)

    model = PoissonRegressor(
        alpha=alpha,
        max_iter=1000,
        tol=1e-6,
    )

    model.fit(
        goal_matrix(enc, cats, venue),
        y,
    )

    return enc, model


def goal_means(fitted, matches):
    enc, model = fitted

    cats, venue, _ = goal_rows(matches)

    return np.clip(
        model.predict(
            goal_matrix(enc, cats, venue)
        ),
        0.05,
        7.0,
    ).reshape(-1, 2)


def outcome_probs(lh, la, rho=0.0):
    ph = poisson.pmf(
        np.arange(17), lh
    )

    pa = poisson.pmf(
        np.arange(17), la
    )

    grid = np.outer(ph, pa)

    # Dixon-Coles adjustment
    grid[0, 0] *= 1 - lh * la * rho
    grid[0, 1] *= 1 + lh * rho
    grid[1, 0] *= 1 + la * rho
    grid[1, 1] *= 1 - rho

    if np.min(grid) < 0:
        raise ValueError(
            "Invalid Dixon-Coles rho for goal rates"
        )

    grid /= grid.sum()

    one_x_two = np.array(
        [
            np.tril(grid, -1).sum(),
            np.trace(grid),
            np.triu(grid, 1).sum(),
        ]
    )

    under = float(
        sum(
            grid[i, j]
            for i in range(3)
            for j in range(3 - i)
        )
    )

    btts = float(
        grid[1:, 1:].sum()
    )

    return one_x_two, under, btts


def score(y, p):
    p = np.asarray(p, dtype=float)

    assert (
        p.ndim == 2
        and p.shape[1] == 3
        and np.allclose(
            p.sum(axis=1),
            1,
            atol=1e-8,
        )
    )

    truth = np.array(
        [
            [int(v == label) for label in LABELS]
            for v in y
        ]
    )

    return dict(
        accuracy=float(
            np.mean(
                np.array(LABELS)[
                    p.argmax(axis=1)
                ] == y
            )
        ),
        multiclass_brier=float(
            np.mean(
                np.sum(
                    (p - truth) ** 2,
                    axis=1,
                )
            )
        ),
        log_loss=float(
            -np.mean(
                np.sum(
                    truth * np.log(
                        np.clip(p, 1e-15, 1)
                    ),
                    axis=1,
                )
            )
        ),
        actual_draw_rate=float(
            np.mean(
                np.asarray(y) == "D"
            )
        ),
        mean_predicted_draw=float(
            np.mean(p[:, 1])
        ),
        top_pick_draws=int(
            np.sum(
                p.argmax(axis=1) == 1
            )
        ),
    )


def evaluate(games):
    train = [
        m for m in games
        if m["season"] in TRAIN_SEASONS
    ]

    test = [
        m for m in games
        if m["season"] == TEST_SEASON
    ]

    inner_train = [
        m for m in train
        if m["season"] == TRAIN_SEASONS[0]
    ]

    inner_valid = [
        m for m in train
        if m["season"] == TRAIN_SEASONS[1]
    ]

    if min(
        len(inner_train),
        len(inner_valid),
        len(test),
    ) < 100:
        raise ValueError(
            "Insufficient training, validation or test rows"
        )

    val_y = np.array(
        [m["result"] for m in inner_valid]
    )

    trials = []

    for alpha in (
        0.0001,
        0.001,
        0.01,
        0.1,
    ):
        model = fit_goals(
            inner_train,
            alpha,
        )

        mu = goal_means(
            model,
            inner_valid,
        )

        for rho in (
            -0.1,
            -0.05,
            0.0,
            0.01,
            0.02,
        ):
            probabilities = np.array(
                [
                    outcome_probs(h, a, rho)[0]
                    for h, a in mu
                ]
            )

            metrics = score(
                val_y,
                probabilities,
            )

            trials.append(
                dict(
                    alpha=alpha,
                    rho=rho,
                    validation_log_loss=metrics[
                        "log_loss"
                    ],
                )
            )

    best = min(
        trials,
        key=lambda t: t[
            "validation_log_loss"
        ],
    )

    mu = goal_means(
        fit_goals(
            train,
            best["alpha"],
        ),
        test,
    )

    goal_p = [
        outcome_probs(
            h,
            a,
            best["rho"],
        )
        for h, a in mu
    ]

    poisson_p = np.array(
        [v[0] for v in goal_p]
    )

    x = elo_features(games)

    tr_idx = [
        i for i, m in enumerate(games)
        if m["season"] in TRAIN_SEASONS
    ]

    te_idx = [
        i for i, m in enumerate(games)
        if m["season"] == TEST_SEASON
    ]

    y = np.array(
        [m["result"] for m in games]
    )

    elo_model = LogisticRegression(
        max_iter=1500
    ).fit(
        x[tr_idx],
        y[tr_idx],
    )

    raw = elo_model.predict_proba(
        x[te_idx]
    )

    elo_p = raw[
        :,
        [
            list(elo_model.classes_).index(c)
            for c in LABELS
        ],
    ]

    market_y, market_p = [], []
    rows = []

    for i, m in enumerate(test):
        if m["odds"]:
            implied = 1 / np.array(
                m["odds"]
            )

            market_y.append(
                m["result"]
            )

            market_p.append(
                implied / implied.sum()
            )

        rows.append(
            dict(
                date=str(m["date"]),
                league=m["league"],
                home=m["home"],
                away=m["away"],
                result=m["result"],
                home_goals=m["hg"],
                away_goals=m["ag"],
                expected_home_goals=float(
                    mu[i, 0]
                ),
                expected_away_goals=float(
                    mu[i, 1]
                ),
                p_home=float(
                    poisson_p[i, 0]
                ),
                p_draw=float(
                    poisson_p[i, 1]
                ),
                p_away=float(
                    poisson_p[i, 2]
                ),
                p_under_2_5=float(
                    goal_p[i][1]
                ),
                p_over_2_5=float(
                    1 - goal_p[i][1]
                ),
                p_btts_yes=float(
                    goal_p[i][2]
                ),
                p_btts_no=float(
                    1 - goal_p[i][2]
                ),
                baseline_p_home=float(
                    elo_p[i, 0]
                ),
                baseline_p_draw=float(
                    elo_p[i, 1]
                ),
                baseline_p_away=float(
                    elo_p[i, 2]
                ),
            )
        )

    report = dict(
        version="0.4 research release",
        training_matches=len(train),
        test_matches=len(test),
        selected_alpha=best["alpha"],
        selected_rho=best["rho"],
        tuning_metric=(
            "1X2 log loss, "
            "2024/25 validation only"
        ),
        validation_trials=trials,
        poisson_dc=score(
            y[te_idx],
            poisson_p,
        ),
        elo_baseline=score(
            y[te_idx],
            elo_p,
        ),
        market=dict(
            matches_with_odds=len(market_y),
            metrics=(
                score(
                    np.array(market_y),
                    np.array(market_p),
                )
                if market_y

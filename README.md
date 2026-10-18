# Football Intelligence AI — etap 1 (bezpłatny prototyp)

To **działający prototyp historycznego backtestu**, a nie gotowy system typów na dziś.

## Co robi
- Pobiera publiczne CSV z football-data.co.uk (9 lig, jeśli źródło udostępnia dane).
- Tworzy cechy **przed** aktualizacją ratingu Elo i formy po meczu.
- Trenuje regresję logistyczną na sezonach 2023/24–2024/25.
- Testuje na odłożonym sezonie 2025/26 (nie używa go do trenowania).
- Raportuje accuracy, wieloklasowy Brier Score i log loss.
- Zapisuje szczegóły prognoz historycznych do CSV.

## Czego JESZCZE nie robi
- Nie pobiera xG, absencji, składów, kursów ani terminarza przyszłych meczów.
- Nie generuje prognoz na dziś ani nie aktualizuje wag automatycznie.
- Nie ma jeszcze porównania z rynkiem ani pełnej kalibracji.
- Nie obejmuje Ekstraklasy (potrzebne inne legalne źródło danych).
- Nie zapisuje wyników między uruchomieniami w repozytorium; pliki wynikowe są artefaktami GitHub Actions przez 30 dni.

## Uruchomienie w GitHub
1. Prześlij do repozytorium zawartość paczki **z zachowaniem folderu `.github/workflows/`**.
2. Wejdź w **Actions** → **Football AI - historical backtest** → **Run workflow**.
3. Poczekaj na zielony znacznik. Wejdź w wykonanie → **Artifacts** → pobierz `football-ai-backtest`.
4. Otwórz `metrics.json` i `historical_predictions.csv`.
5. Jeśli workflow nie pojawia się: upewnij się, że plik `.github/workflows/backtest.yml` jest na domyślnej gałęzi i Actions są włączone.

## Lokalnie (opcjonalnie)
```bash
python -m pip install -r requirements.txt
python download_data.py
python backtest.py
```

## Zastrzeżenia metodologiczne
- Podział według sezonów ogranicza leakage między treningiem a testem. Przy tej wersji test ma jeden sezon i nie stanowi jeszcze dowodu przewagi nad rynkiem.
- Elo jest aktualizowane chronologicznie również dla spotkań w sezonie testowym: to dozwolone, ponieważ wyniki wcześniejszych meczów były znane przed kolejnymi spotkaniami.
- Cechy obejmują tylko wyniki i formę bramkową, bez xG; nie należy na tej podstawie obstawiać.
- Dostępność danych i warunki automatycznego pobierania zależą od zewnętrznego dostawcy.

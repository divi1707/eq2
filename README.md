# Himalayan earthquake forecasting with ConvLSTM

Student research project (IIT Roorkee). Adapts the ConvLSTM approach of Zhang and Wang (IEEE TGRS, 2023) to the Central Himalaya (26.5 to 31.5 N, 78 to 88 E).

This is probabilistic forecasting research. It is not a way to predict the exact time, place or size of a real earthquake. No skill is claimed without held-out testing.

## Status
- Stage 1 done: config, USGS download and validation, inspection report, catalogue explorer app, tests.
- Not done yet: cleaning, monthly maps, model, training, evaluation, forecasts.
- No real catalogue has been run through this code yet. Results: none.

## Setup (macOS, Apple Silicon)
```bash
cd himalayan_convlstm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Stage 1 commands
```bash
# 1. Run the tests (they use mock data and do not touch the internet)
python -m pytest -q

# 2. Download the catalogue from USGS (needs internet)
python -m src.download_catalogue --config config.yaml

# 3. Inspect what came back (real column names, counts, magnitude types, rough completeness)
python -m src.inspect_catalogue --config config.yaml
```
Send me the full terminal output of step 3 and I will finalise the cleaning code against the real columns.

## Catalogue explorer app
A Streamlit app for checking the catalogue: events per year, magnitude types, rough completeness, a map and a table. It shows no forecasts.
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
It opens in your browser. It can load the saved catalogue, a CSV you upload, or run the USGS download for you. The sidebar lets you change the study area to explore. Changing it there does not change `config.yaml`.

Hosting on Streamlit Community Cloud: upload the whole project folder to the GitHub repo (so `src/`, `config.yaml` and `app/` sit together) and set the main file to `app/app.py`. If `app.py` sits at the top of the repo, the app still finds `src/` as long as `src/` and `config.yaml` are in the same repo. The cloud disk is temporary, so the downloaded catalogue is lost when the app restarts. `requirements.txt` is kept light for this reason (no PyTorch).

In Google Colab, Streamlit needs a tunnel to open in the browser. I have not tested that route, so the Mac route is the supported one.

## If the live download fails
- Run `python -m src.download_catalogue --print-urls` to get one URL per time window.
- Open each URL in a browser and save the CSV. Merge them into one file (keep one header line).
- Run `python -m src.download_catalogue --from-csv path/to/merged.csv`.
- The metadata file will record that this data was supplied manually and not verified.

## Common errors
- `HTTP 400 ... exceeds search limit`: a window holds more than 20000 events. The downloader splits windows automatically. If you see this in manual mode, use shorter windows.
- `received an HTML page instead of CSV`: the service returned an error page. Wait and retry.
- `incomplete download`: fewer rows arrived than the service counted. Run again with `--force`.
- `Could not reach ...`: no internet or a firewall. Use the manual route above.

## Files written by stage 1
- `data/raw/usgs_windows/`: one CSV per time window
- `data/raw/usgs_catalogue_raw.csv`: merged raw catalogue
- `data/raw/usgs_catalogue_raw.meta.json`: how the data was obtained
- `outputs/metrics/catalogue_inspection.json`: inspection report

## Key settings (config.yaml)
All settings are initial choices. Magnitude 4.0 is a candidate threshold only. The download goes down to 2.5 so completeness can be studied before the threshold is fixed.

See `docs/methodology.md` for what the paper did and where this project differs.

# Himalayan earthquake forecasting with ConvLSTM

Student research project (IIT Roorkee). Adapts the ConvLSTM approach of Zhang and Wang (IEEE TGRS, 2023) to a Himalayan study area. The area in `config.yaml` (26.5 to 31.5 N, 78 to 88 E) comes from the project brief and is not final.

This is probabilistic forecasting research, not a way to predict the exact time, place or size of a real earthquake. No skill is claimed without held-out testing.

## Status
- Stage 1 done: config, USGS download and validation, inspection report, catalogue explorer app, tests.
- Not done yet: cleaning, monthly maps, model, training, evaluation, forecasts.
- No real catalogue has been run through this code yet. Results: none.

## Files (all in one folder, no subfolders)
- `config.yaml`: study area, dates, magnitude floor, grid, settings
- `common.py`: config loading, date windows, logging
- `download_catalogue.py`: downloads the USGS catalogue and checks it
- `inspect_catalogue.py`: prints columns, counts, magnitude types, rough completeness
- `explore.py`: data functions used by the app
- `app.py`: the Streamlit catalogue explorer
- `test_*.py`: unit tests (mock data only)
- `methodology.md`: what the paper did and where this project differs
- `requirements.txt`, `requirements-dev.txt`: packages

Folders such as `data/raw/` are created automatically when you run the download.

## Run on your computer
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python download_catalogue.py --config config.yaml
python inspect_catalogue.py --config config.yaml
streamlit run app.py
```
Run these from inside the folder that holds the files. Send me the full output of `inspect_catalogue.py` so the cleaning code can be finalised against the real columns.

## Run the app on Streamlit Community Cloud
1. Put ALL files from this folder at the top level of your GitHub repo.
2. Set the main file to `app.py` and reboot the app.
3. Use the "Download from USGS now" button or upload a CSV in the sidebar.
The cloud disk is temporary, so downloaded data disappears on restart. `requirements.txt` is light (no PyTorch) for this reason.

## If the live download fails
- Run `python download_catalogue.py --print-urls` to get one URL per time window.
- Open each URL in a browser and save the CSV. Merge them into one file (keep one header line).
- Run `python download_catalogue.py --from-csv path/to/merged.csv`.
- The metadata file records that this data was supplied manually and not verified.

## Common errors
- `HTTP 400 ... exceeds search limit`: a window holds more than 20000 events. The downloader splits windows automatically. In manual mode use shorter windows.
- `received an HTML page instead of CSV`: the service returned an error page. Wait and retry.
- `incomplete download`: fewer rows arrived than the service counted. Run again with `--force`.
- `Could not reach ...`: no internet or a firewall. Use the manual route above.
